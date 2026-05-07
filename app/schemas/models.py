from pydantic import BaseModel, Field
from typing import Optional


# ── /chat ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send")
    system: str = Field(
        default="You are a helpful assistant.",
        description="System prompt",
    )
    model: str = Field(
        default="claude-sonnet-4-6",
        description=(
            "Model ID. "
            "Claude: claude-sonnet-4-6, claude-haiku-4-5-20251001. "
            "Groq (free): llama-3.3-70b-versatile, llama-3.1-8b-instant, "
            "mixtral-8x7b-32768, gemma2-9b-it. "
            "xAI Grok: grok-2-latest, grok-2-mini-latest, grok-beta."
        ),
    )
    max_tokens: int = Field(default=1024, ge=1, le=4096)


# ── /extract ───────────────────────────────────────────────────────────────

class ContactInfo(BaseModel):
    """Extracted contact information from unstructured text."""
    full_name: Optional[str] = Field(None, description="Person's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    company: Optional[str] = Field(None, description="Company or organisation")
    job_title: Optional[str] = Field(None, description="Job title or role")
    location: Optional[str] = Field(None, description="City, state, or country")
    linkedin: Optional[str] = Field(None, description="LinkedIn URL or handle")
    website: Optional[str] = Field(None, description="Personal or company website")


class CalendarEvent(BaseModel):
    """Extracted calendar event from unstructured text."""
    title: str = Field(..., description="Event title or name")
    date: Optional[str] = Field(None, description="Date in ISO-8601 format (YYYY-MM-DD)")
    time: Optional[str] = Field(None, description="Time in HH:MM 24h format")
    duration_minutes: Optional[int] = Field(None, description="Duration in minutes")
    location: Optional[str] = Field(None, description="Venue or URL")
    attendees: list[str] = Field(default_factory=list, description="List of attendee names")
    description: Optional[str] = Field(None, description="Event notes or agenda")
    is_online: bool = Field(default=False, description="True if virtual/online event")


class ProductReview(BaseModel):
    """Extracted product review sentiment from unstructured text."""
    product_name: Optional[str] = Field(None, description="Name of the product reviewed")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Rating out of 5")
    sentiment: str = Field(..., description="positive | neutral | negative")
    pros: list[str] = Field(default_factory=list, description="Positive aspects mentioned")
    cons: list[str] = Field(default_factory=list, description="Negative aspects mentioned")
    summary: str = Field(..., description="One-sentence summary of the review")


class ExtractionRequest(BaseModel):
    text: str = Field(..., description="Unstructured text to extract data from")
    schema_type: str = Field(
        default="contact",
        description="What to extract: 'contact' | 'event' | 'review'",
    )
    model: str = Field(
        default="claude-sonnet-4-6",
        description=(
            "Model to use for extraction. "
            "Claude: claude-sonnet-4-6, claude-haiku-4-5-20251001. "
            "Groq (free): llama-3.3-70b-versatile, mixtral-8x7b-32768. "
            "xAI Grok: grok-2-latest, grok-2-mini-latest."
        ),
    )


class ExtractionResponse(BaseModel):
    schema_type: str
    data: dict
    confidence: str = Field(
        description="high | medium | low — model's confidence in extraction"
    )
