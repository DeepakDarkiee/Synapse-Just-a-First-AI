# AI Engineering Journey

FastAPI service with:
- **Streaming chat** (SSE) using Claude or Groq — with automatic weather tool calling
- **Structured extraction** — unstructured text → validated JSON via instructor + Pydantic

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Get API Keys](#2-get-api-keys)
3. [Clone & Install](#3-clone--install)
4. [Configure Environment](#4-configure-environment)
5. [Run Locally](#5-run-locally)
6. [Test the Endpoints](#6-test-the-endpoints)
7. [Deploy](#7-deploy)
8. [API Reference](#8-api-reference)
9. [Architecture](#9-architecture)

---

## 1. Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | **3.12 or 3.13** | `brew install python@3.13` |
| pip | latest | bundled with Python |
| git | any | `brew install git` |

> **Python 3.14 is not supported yet.** `pydantic-core` and `jiter` need pre-built wheels
> that only exist for Python ≤ 3.13. If you only have 3.14:
> ```bash
> brew install python@3.13
> ```

---

## 2. Get API Keys

You need at least one LLM key. Both have free tiers.

### Anthropic (Claude) — $5 free credit
1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Go to **API Keys** → **Create Key**
3. Copy the key — it starts with `sk-ant-`

### Groq — completely free
1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → **Create API Key**
3. Copy the key — it starts with `gsk_`

> The weather tool uses [wttr.in](https://wttr.in) — **no API key needed**.

---

## 3. Clone & Install

```bash
# Clone the repo
git clone https://github.com/<your-username>/ai-engineering-journey.git
cd ai-engineering-journey

# Create a virtual environment with Python 3.13
python3.13 -m venv .venv

# Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

You should see packages installing from pre-built wheels (fast, no Rust compilation).

---

## 4. Configure Environment

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Required for /chat (Claude) and /extract
ANTHROPIC_API_KEY=sk-ant-...

# Required for /chat with Groq models (free)
GROQ_API_KEY=gsk_...
```

> You only need the key for the provider you plan to use.
> If you skip `GROQ_API_KEY`, Groq models will return a 500 error.

---

## 5. Run Locally

```bash
# Make sure the venv is active
source .venv/bin/activate

# Start the server with auto-reload
uvicorn app.main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

Open these in your browser:
- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc
- **Health check** → http://localhost:8000/health

---

## 6. Test the Endpoints

### /chat — streaming chat with weather tool

**With Claude (Anthropic):**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Mumbai right now?"}'
```

**With Groq (free, supports tool calling):**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather in Mumbai right now?",
    "model": "llama-3.3-70b-versatile"
  }'
```

**With Groq (fast, plain chat — no tool calling):**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain what a transformer model is in 3 sentences.",
    "model": "llama-3.1-8b-instant"
  }'
```

SSE events you'll see when the weather tool fires:
```
event: tool_call
data: {"tool": "get_weather"}

event: tool_result_start
data: {"city": "Mumbai"}

event: tool_result
data: {"city": "Mumbai, India", "temperature_c": 32, "condition": "Partly cloudy", "humidity_pct": 78, ...}

event: text
data: {"chunk": "The current weather in Mumbai is 32°C and partly cloudy..."}

event: done
data: {}
```

---

### /extract — structured JSON from unstructured text

**Contact info:**
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

**Calendar event:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Team standup every Monday at 10am IST for 30 mins on Google Meet. Attendees: Rahul, Sneha, Deepak.",
    "schema_type": "event"
  }'
```

**Product review:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This laptop is blazing fast and the battery lasts all day. Keyboard feels great. Only gripe is the webcam is mediocre at best. 4/5 stars.",
    "schema_type": "review"
  }'
```

---

## 7. Deploy

### Option A — Fly.io (recommended, free tier)

```bash
# Install CLI
brew install flyctl

# Log in
fly auth login

# Create the app (reads fly.toml)
fly launch

# Set your secrets
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly secrets set GROQ_API_KEY=gsk_...

# Deploy
fly deploy
```

Your service will be live at `https://ai-engineering-journey.fly.dev`.

### Option B — Render (no CLI needed)

1. Push this repo to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New Web Service**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Under **Environment** → add:
   - `ANTHROPIC_API_KEY` = `sk-ant-...`
   - `GROQ_API_KEY` = `gsk_...`
5. Click **Deploy**

---

## 8. API Reference

### POST /chat

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | User message |
| `system` | string | `"You are a helpful assistant."` | System prompt |
| `model` | string | `claude-sonnet-4-6` | See model table below |
| `max_tokens` | int | `1024` | 1–4096 |

**Supported models:**

| Model ID | Provider | Tool calling | Speed |
|----------|----------|-------------|-------|
| `claude-sonnet-4-6` | Anthropic | Yes | Fast |
| `claude-haiku-4-5-20251001` | Anthropic | Yes | Fastest |
| `llama-3.3-70b-versatile` | Groq (free) | Yes | Fast |
| `llama-3.1-70b-versatile` | Groq (free) | Yes | Fast |
| `llama-3.1-8b-instant` | Groq (free) | No | Very fast |
| `mixtral-8x7b-32768` | Groq (free) | No | Fast |
| `gemma2-9b-it` | Groq (free) | No | Fast |

---

### POST /extract

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Unstructured input text |
| `schema_type` | string | `contact` | `contact` \| `event` \| `review` |

**Schema types:**

| `schema_type` | Extracted fields |
|---------------|-----------------|
| `contact` | name, email, phone, company, job title, location, linkedin, website |
| `event` | title, date (ISO-8601), time (HH:MM), duration, location, attendees, is\_online |
| `review` | product name, rating (1–5), sentiment, pros, cons, summary |

---

### GET /health

Returns `{"status": "healthy"}`. Use this for uptime checks.

---

## 9. Architecture

```
POST /chat
  └─ model starts with "claude-*"
       └─ _stream_claude() → Anthropic streaming API
            ├─ stop_reason = tool_use  → get_weather(city) → wttr.in
            │    └─ second pass stream with tool result
            └─ stop_reason = end_turn → SSE text stream

  └─ model starts with "llama-*" / "mixtral-*" / "gemma-*"
       └─ _stream_groq() → Groq streaming API (OpenAI-compatible)
            ├─ finish_reason = tool_calls (only on supported models)
            │    └─ get_weather(city) → second pass stream
            └─ finish_reason = stop → SSE text stream

POST /extract
  └─ ExtractionRequest
       └─ instructor.from_anthropic → claude-sonnet-4-6
            └─ ContactInfo | CalendarEvent | ProductReview
                 └─ ExtractionResponse { data, confidence }
```

**Tech stack:**

| Library | Purpose |
|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | Web framework + auto docs |
| [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) | Claude streaming + tool use |
| [Groq SDK](https://github.com/groq/groq-python) | Groq streaming + tool use (free) |
| [instructor](https://python.useinstructor.com/) | Structured Pydantic outputs from LLMs |
| [wttr.in](https://wttr.in/) | Weather data — no API key needed |
| [Fly.io](https://fly.io/) / [Render](https://render.com/) | Free tier deployment |
