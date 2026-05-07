import json
import os
from typing import AsyncGenerator

import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.models import ChatRequest
from app.tools.weather import WEATHER_TOOL_DEFINITION, get_weather

router = APIRouter()

_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


async def _stream_claude(req: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream Claude response with weather tool support."""
    client = _get_anthropic()
    messages = [{"role": "user", "content": req.message}]

    # ── First pass: may trigger tool use ──────────────────────────────────
    async with client.messages.stream(
        model=req.model,
        max_tokens=req.max_tokens,
        system=req.system,
        messages=messages,
        tools=[WEATHER_TOOL_DEFINITION],
    ) as stream:
        tool_use_block = None
        text_buffer = []

        async for event in stream:
            if event.type == "content_block_start":
                if hasattr(event.content_block, "type"):
                    if event.content_block.type == "tool_use":
                        tool_use_block = event.content_block
                        yield _sse("tool_call", {"tool": tool_use_block.name})

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    text_buffer.append(delta.text)
                    yield _sse("text", {"chunk": delta.text})

            elif event.type == "message_stop":
                pass

        final_message = await stream.get_final_message()

    # ── Handle tool use if triggered ──────────────────────────────────────
    if final_message.stop_reason == "tool_use":
        tool_block = next(
            (b for b in final_message.content if b.type == "tool_use"), None
        )
        if tool_block and tool_block.name == "get_weather":
            city = tool_block.input.get("city", "")
            yield _sse("tool_result_start", {"city": city})

            try:
                weather_data = await get_weather(city)
            except Exception as exc:
                weather_data = {"error": str(exc)}

            yield _sse("tool_result", weather_data)

            # ── Second pass: final answer with tool result ─────────────────
            messages = [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": final_message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": json.dumps(weather_data),
                        }
                    ],
                },
            ]

            async with client.messages.stream(
                model=req.model,
                max_tokens=req.max_tokens,
                system=req.system,
                messages=messages,
                tools=[WEATHER_TOOL_DEFINITION],
            ) as stream2:
                async for event in stream2:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield _sse("text", {"chunk": event.delta.text})

    yield _sse("done", {})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Streaming chat endpoint (SSE).

    Supports Claude models. Automatically calls the weather tool when
    the user asks about weather.

    **SSE event types:**
    - `text` — `{"chunk": "..."}` streamed text token
    - `tool_call` — `{"tool": "get_weather"}` tool invocation started
    - `tool_result_start` — `{"city": "..."}` fetching weather for city
    - `tool_result` — `{temperature_c, condition, ...}` raw tool data
    - `done` — stream finished
    """
    if not req.model.startswith("claude"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' not supported. Use a claude-* model ID.",
        )

    return StreamingResponse(
        _stream_claude(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
