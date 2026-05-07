import os
from typing import Type

import anthropic
import instructor
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

_instructor_client: instructor.Instructor | None = None

SCHEMA_MAP: dict[str, Type[BaseModel]] = {
    "contact": ContactInfo,
    "event": CalendarEvent,
    "review": ProductReview,
}


def _get_instructor() -> instructor.Instructor:
    global _instructor_client
    if _instructor_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        raw = anthropic.Anthropic(api_key=api_key)
        _instructor_client = instructor.from_anthropic(raw)
    return _instructor_client


def _build_system_prompt(schema_type: str) -> str:
    prompts = {
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
    }
    return prompts.get(schema_type, "Extract structured information from the text.")


@router.post("/extract", response_model=ExtractionResponse)
async def extract(req: ExtractionRequest):
    """
    Structured extraction endpoint.

    Sends unstructured text to Claude via **instructor** and returns a
    validated Pydantic object as JSON.

    **schema_type options:**
    - `contact` — name, email, phone, company, location …
    - `event` — title, date, time, duration, attendees …
    - `review` — sentiment, rating, pros, cons, summary …
    """
    if req.schema_type not in SCHEMA_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown schema_type '{req.schema_type}'. "
                   f"Choose from: {list(SCHEMA_MAP.keys())}",
        )

    schema_cls = SCHEMA_MAP[req.schema_type]
    client = _get_instructor()

    try:
        result = client.chat.completions.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_build_system_prompt(req.schema_type),
            messages=[{"role": "user", "content": req.text}],
            response_model=schema_cls,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {exc}") from exc

    # Rough confidence heuristic: count non-null fields
    data = result.model_dump()
    total = sum(1 for v in _flatten(data) if v is not None and v != [] and v != "")
    possible = max(len(list(_flatten(data))), 1)
    ratio = total / possible
    confidence = "high" if ratio > 0.6 else ("medium" if ratio > 0.3 else "low")

    return ExtractionResponse(schema_type=req.schema_type, data=data, confidence=confidence)


def _flatten(d, parent_key=""):
    """Yield leaf values from a nested dict/list."""
    if isinstance(d, dict):
        for v in d.values():
            yield from _flatten(v)
    elif isinstance(d, list):
        for v in d:
            yield from _flatten(v)
    else:
        yield d
