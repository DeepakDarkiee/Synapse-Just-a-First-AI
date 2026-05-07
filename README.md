# AI Engineering Journey

FastAPI service demonstrating:
- **Streaming chat** via SSE with Claude + automatic weather tool calling
- **Structured extraction** via [instructor](https://python.useinstructor.com/) + Pydantic

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | SSE streaming chat; ask about weather to see tool calling |
| POST | `/extract` | Extract structured JSON from unstructured text |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/ai-engineering-journey
cd ai-engineering-journey

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Run
uvicorn app.main:app --reload
# Open http://localhost:8000/docs
```

## Example Requests

### /chat — streaming with weather tool

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather like in Mumbai right now?"}'
```

SSE events emitted:
```
event: tool_call
data: {"tool": "get_weather"}

event: tool_result_start
data: {"city": "Mumbai"}

event: tool_result
data: {"city": "Mumbai, India", "temperature_c": 32, "condition": "Partly cloudy", ...}

event: text
data: {"chunk": "The current weather in Mumbai is 32°C and partly cloudy..."}

event: done
data: {}
```

### /extract — contact info

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hi, I am Priya Sharma, Senior ML Engineer at Razorpay. Reach me at priya@razorpay.com or +91-9876543210. Based in Bengaluru.",
    "schema_type": "contact"
  }'
```

Response:
```json
{
  "schema_type": "contact",
  "data": {
    "full_name": "Priya Sharma",
    "email": "priya@razorpay.com",
    "phone": "+91-9876543210",
    "company": "Razorpay",
    "job_title": "Senior ML Engineer",
    "location": "Bengaluru",
    "linkedin": null,
    "website": null
  },
  "confidence": "high"
}
```

### /extract — calendar event

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Team standup every Monday at 10am IST for 30 mins on Google Meet. Attendees: Rahul, Sneha, Deepak.",
    "schema_type": "event"
  }'
```

### /extract — product review

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This laptop is blazing fast and the battery lasts all day. Keyboard feels great. Only gripe is the webcam is mediocre at best. 4/5 stars.",
    "schema_type": "review"
  }'
```

## Deploy

### Fly.io (recommended)

```bash
brew install flyctl
fly auth login
fly launch          # creates app, reads fly.toml
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

### Render

1. Push to GitHub
2. Go to https://dashboard.render.com → New Web Service → connect repo
3. Render auto-detects `render.yaml`
4. Set `ANTHROPIC_API_KEY` in Environment → Deploy

## Architecture

```
POST /chat
  └─ ChatRequest → Claude claude-sonnet-4-6 (streaming)
       ├─ stop_reason=tool_use → get_weather(city) → wttr.in API
       │     └─ tool_result → Claude second pass → SSE text stream
       └─ stop_reason=end_turn → SSE text stream

POST /extract
  └─ ExtractionRequest → instructor.from_anthropic → Pydantic model
       └─ ContactInfo | CalendarEvent | ProductReview → ExtractionResponse
```

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude streaming + tool use
- [instructor](https://python.useinstructor.com/) — structured outputs via Pydantic
- [wttr.in](https://wttr.in/) — free weather API (no key required)
- [Fly.io](https://fly.io/) / [Render](https://render.com/) — free tier deployment
