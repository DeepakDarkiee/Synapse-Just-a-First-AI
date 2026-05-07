import os
from typing import Type

import anthropic
import groq as groq_sdk
import instructor
import openai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.models import (
    CalendarEvent,
    ContactInfo,
    ExtractionRequest,
    ExtractionResponse,
    ProductReview,
)

router = APIRouter()

SCHEMA_MAP: dict[str, Type[BaseModel]] = {
    "contact": ContactInfo,
    "event":   CalendarEvent,
    "review":  ProductReview,
}

# Cache one instructor client per provider (not per model)
_clients: dict[str, instructor.Instructor] = {}


def _get_instructor(model: str) -> tuple[instructor.Instructor, str]:
    """Return (instructor_client, provider) for the given model ID."""
    if model.startswith("claude"):
        if "anthropic" not in _clients:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
            _clients["anthropic"] = instructor.from_anthropic(
                anthropic.Anthropic(api_key=key)
            )
        return _clients["anthropic"], "anthropic"

    elif model.startswith(("llama", "mixtral", "gemma", "whisper")):
        if "groq" not in _clients:
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
            # Groq requires JSON mode for structured outputs
            _clients["groq"] = instructor.from_groq(
                groq_sdk.Groq(api_key=key),
                mode=instructor.Mode.JSON,
            )
        return _clients["groq"], "groq"

    elif model.startswith("grok"):
        if "xai" not in _clients:
            key = os.getenv("XAI_API_KEY")
            if not key:
                raise HTTPException(status_code=500, detail="XAI_API_KEY not configured")
            _clients["xai"] = instructor.from_openai(
                openai.OpenAI(api_key=key, base_url="https://api.x.ai/v1")
            )
        return _clients["xai"], "xai"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{model}'. Supported prefixes: "
                "claude-* (Anthropic), llama-*/mixtral-*/gemma-* (Groq), grok-* (xAI)."
            ),
        )


def _build_system_prompt(schema_type: str) -> str:
    return {
        "contact": (
            "You are an expert at extracting structured contact information from text. "
            "Extract all available fields. Leave fields as null if not found."
        ),
        "event": (
            "You are an expert at extracting calendar event details from text. "
            "Normalise dates to ISO-8601 (YYYY-MM-DD) and times to HH:MM 24h. "
            "Leave fields as null if not found."
        ),
        "review": (
            "You are an expert at analysing product reviews. "
            "Classify sentiment as positive, neutral, or negative. "
            "Extract concrete pros and cons as short bullet phrases."
        ),
    }.get(schema_type, "Extract structured information from the text.")


@router.post("/extract", response_model=ExtractionResponse)
async def extract(req: ExtractionRequest):
    """
    Structured extraction endpoint.

    Supports **Claude**, **Groq**, and **xAI Grok** via instructor + Pydantic.

    **schema_type options:**
    - `contact` — name, email, phone, company, location …
    - `event` — title, date, time, duration, attendees …
    - `review` — sentiment, rating, pros, cons, summary …

    **model options:** same as /chat (claude-*, llama-*/mixtral-*/gemma-*, grok-*)
    """
    if req.schema_type not in SCHEMA_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown schema_type '{req.schema_type}'. "
                   f"Choose from: {list(SCHEMA_MAP.keys())}",
        )

    schema_cls = SCHEMA_MAP[req.schema_type]
    client, provider = _get_instructor(req.model)
    system = _build_system_prompt(req.schema_type)

    # Anthropic uses a top-level `system` param; OpenAI-compat uses system message
    if provider == "anthropic":
        call_kwargs = dict(
            model=req.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": req.text}],
            response_model=schema_cls,
        )
    else:
        call_kwargs = dict(
            model=req.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": req.text},
            ],
            response_model=schema_cls,
        )

    try:
        result = client.chat.completions.create(**call_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {exc}") from exc

    data = result.model_dump()
    filled = sum(1 for v in _flatten(data) if v is not None and v != [] and v != "")
    total  = max(len(list(_flatten(data))), 1)
    ratio  = filled / total
    confidence = "high" if ratio > 0.6 else ("medium" if ratio > 0.3 else "low")

    return ExtractionResponse(schema_type=req.schema_type, data=data, confidence=confidence)


def _flatten(d):
    if isinstance(d, dict):
        for v in d.values():   yield from _flatten(v)
    elif isinstance(d, list):
        for v in d:            yield from _flatten(v)
    else:
        yield d
