import json
import os
from typing import AsyncGenerator

import anthropic
from groq import AsyncGroq
from openai import AsyncOpenAI
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.models import ChatRequest
from app.tools.weather import WEATHER_TOOL_DEFINITION, get_weather

router = APIRouter()

# ── Provider routing ───────────────────────────────────────────────────────

# Models whose prefix routes to each provider
_GROQ_PREFIXES = ("llama", "mixtral", "gemma", "whisper")
_XAI_PREFIXES  = ("grok",)

# Models (across OpenAI-compat providers) that support tool/function calling
_TOOL_CAPABLE_MODELS = {
    # Groq
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    # xAI Grok
    "grok-2-latest",
    "grok-2-mini-latest",
    "grok-beta",
}

# ── Lazy singletons ────────────────────────────────────────────────────────

_anthropic_client: anthropic.AsyncAnthropic | None = None
_groq_client: AsyncGroq | None = None
_xai_client: AsyncOpenAI | None = None


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        _anthropic_client = anthropic.AsyncAnthropic(api_key=key)
    return _anthropic_client


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        _groq_client = AsyncGroq(api_key=key)
    return _groq_client


def _get_xai() -> AsyncOpenAI:
    global _xai_client
    if _xai_client is None:
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="XAI_API_KEY not configured")
        _xai_client = AsyncOpenAI(api_key=key, base_url="https://api.x.ai/v1")
    return _xai_client


# ── Anthropic / Claude ─────────────────────────────────────────────────────

async def _stream_claude(req: ChatRequest) -> AsyncGenerator[str, None]:
    client = _get_anthropic()
    messages = [{"role": "user", "content": req.message}]

    async with client.messages.stream(
        model=req.model,
        max_tokens=req.max_tokens,
        system=req.system,
        messages=messages,
        tools=[WEATHER_TOOL_DEFINITION],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_start":
                if getattr(event.content_block, "type", None) == "tool_use":
                    yield _sse("tool_call", {"tool": event.content_block.name})
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield _sse("text", {"chunk": event.delta.text})

        final = await stream.get_final_message()

    if final.stop_reason == "tool_use":
        tool_block = next((b for b in final.content if b.type == "tool_use"), None)
        if tool_block and tool_block.name == "get_weather":
            city = tool_block.input.get("city", "")
            yield _sse("tool_result_start", {"city": city})
            weather_data = await _fetch_weather(city)
            yield _sse("tool_result", weather_data)

            messages = [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": final.content},
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


# ── OpenAI-compatible (Groq + xAI) ────────────────────────────────────────

# Both Groq and xAI use the OpenAI tool format — convert our definition once.
_OAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": WEATHER_TOOL_DEFINITION["name"],
        "description": WEATHER_TOOL_DEFINITION["description"],
        "parameters": WEATHER_TOOL_DEFINITION["input_schema"],
    },
}


async def _stream_openai_compat(
    client: AsyncGroq | AsyncOpenAI,
    req: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Shared streaming loop for any OpenAI-compatible provider (Groq, xAI)."""
    supports_tools = req.model in _TOOL_CAPABLE_MODELS
    messages = [
        {"role": "system", "content": req.system},
        {"role": "user", "content": req.message},
    ]

    kwargs: dict = dict(model=req.model, max_tokens=req.max_tokens, messages=messages, stream=True)
    if supports_tools:
        kwargs["tools"] = [_OAI_WEATHER_TOOL]

    # Accumulate tool-call fields across streamed chunks
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_args = ""
    finish_reason: str | None = None

    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        delta = choice.delta

        if delta.content:
            yield _sse("text", {"chunk": delta.content})

        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.id:
                    tool_call_id = tc.id
                if tc.function.name:
                    tool_call_name = tc.function.name
                    yield _sse("tool_call", {"tool": tool_call_name})
                if tc.function.arguments:
                    tool_call_args += tc.function.arguments

    if finish_reason == "tool_calls" and tool_call_name == "get_weather":
        city = json.loads(tool_call_args).get("city", "")
        yield _sse("tool_result_start", {"city": city})
        weather_data = await _fetch_weather(city)
        yield _sse("tool_result", weather_data)

        messages = [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.message},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_call_name, "arguments": tool_call_args},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(weather_data),
            },
        ]

        stream2 = await client.chat.completions.create(
            model=req.model, max_tokens=req.max_tokens, messages=messages, stream=True
        )
        async for chunk in stream2:
            delta = chunk.choices[0].delta
            if delta.content:
                yield _sse("text", {"chunk": delta.content})

    yield _sse("done", {})


# ── Shared helpers ─────────────────────────────────────────────────────────

async def _fetch_weather(city: str) -> dict:
    try:
        return await get_weather(city)
    except Exception as exc:
        return {"error": str(exc)}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Route ──────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Streaming chat endpoint (SSE).

    Supports **Claude** (Anthropic), **Groq** (free), and **xAI Grok** models.
    Automatically calls the weather tool when the user asks about weather.

    **Supported models:**
    - Claude: `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`
    - Groq (free): `llama-3.3-70b-versatile` *(tool calling)*, `llama-3.1-8b-instant`,
      `mixtral-8x7b-32768`, `gemma2-9b-it`
    - xAI Grok: `grok-2-latest` *(tool calling)*, `grok-2-mini-latest`, `grok-beta`

    **SSE event types:**
    - `text` — `{"chunk": "..."}` streamed token
    - `tool_call` — `{"tool": "get_weather"}` tool triggered
    - `tool_result_start` — `{"city": "..."}` fetching weather
    - `tool_result` — `{temperature_c, condition, ...}` raw weather data
    - `done` — stream finished
    """
    if req.model.startswith("claude"):
        generator = _stream_claude(req)
    elif req.model.startswith(_GROQ_PREFIXES):
        generator = _stream_openai_compat(_get_groq(), req)
    elif req.model.startswith(_XAI_PREFIXES):
        generator = _stream_openai_compat(_get_xai(), req)
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{req.model}'. Supported prefixes: "
                "claude-* (Anthropic), "
                "llama-*/mixtral-*/gemma-* (Groq), "
                "grok-* (xAI)."
            ),
        )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
