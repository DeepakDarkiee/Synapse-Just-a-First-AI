# Learning Guide — AI Engineering Journey

This file explains **what** was built, **why** each decision was made, and **how** the key patterns work. Read it alongside the code. Every section links to the relevant file.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [FastAPI — Why Not Flask or Django?](#2-fastapi--why-not-flask-or-django)
3. [Streaming with SSE — What, Why, and How](#3-streaming-with-sse--what-why-and-how)
4. [LLM Tool Calling — The Two-Pass Pattern](#4-llm-tool-calling--the-two-pass-pattern)
5. [Structured Extraction with instructor + Pydantic](#5-structured-extraction-with-instructor--pydantic)
6. [Multi-Provider Routing](#6-multi-provider-routing)
7. [Environment Variables and python-dotenv](#7-environment-variables-and-python-dotenv)
8. [Frontend — Why No React?](#8-frontend--why-no-react)
9. [Deployment — Docker, Fly.io, Render](#9-deployment--dockerflyio-render)
10. [Concepts Glossary](#10-concepts-glossary)
11. [What to Learn Next](#11-what-to-learn-next)

---

## 1. The Big Picture

```
User
 │
 ▼
Browser (frontend/index.html)
 │  POST /chat     → streams SSE back
 │  POST /extract  → returns JSON
 ▼
FastAPI (app/main.py)
 │
 ├── /chat  ──► app/routers/chat.py
 │               ├── Claude (Anthropic SDK)   ← stops_reason=tool_use?
 │               ├── Groq   (Groq SDK)        ←   yes → get_weather() → wttr.in
 │               └── xAI    (OpenAI SDK)      ←   then second LLM pass
 │
 └── /extract ──► app/routers/extract.py
                  └── instructor wraps any provider
                      └── returns validated Pydantic model as JSON
```

**The core idea:** every LLM is just an HTTP API. Once you understand how to stream text and handle tool calls from one provider, the others are almost identical.

---

## 2. FastAPI — Why Not Flask or Django?

**File:** `app/main.py`

### What
FastAPI is a modern Python web framework built on top of Starlette and Pydantic.

### Why FastAPI over Flask
| | Flask | FastAPI |
|---|---|---|
| Async support | Bolted on (via async def) | Native (built on asyncio) |
| Request validation | Manual | Automatic via Pydantic |
| Auto docs | Plugin needed | Built-in at `/docs` |
| Performance | Sync by default | Async by default |
| Type hints | Optional | First-class |

For LLM apps, async is critical — you're waiting on network I/O (API calls) constantly. Flask's sync model would block a thread for every open SSE connection.

### Why Not Django
Django is a batteries-included framework (ORM, admin, auth) for database-heavy apps. This project has no database — it's a pure API gateway between the browser and LLM providers. Django's overhead would add complexity for no benefit.

### Key pattern — dependency injection via `include_router`
```python
# app/main.py
app.include_router(chat.router,    tags=["Chat"])
app.include_router(extract.router, tags=["Extract"])
```
Each router is a self-contained module. This is how large FastAPI apps are organised — one file per feature area, all wired together in `main.py`.

### Auto-generated docs
Visit `/docs` after running the server. FastAPI reads your Pydantic models and generates interactive Swagger UI automatically. No extra work.

---

## 3. Streaming with SSE — What, Why, and How

**File:** `app/routers/chat.py`

### What is SSE?
Server-Sent Events (SSE) is a protocol where the server pushes data to the browser over a single long-lived HTTP connection. The browser receives events as they arrive — no polling needed.

Each SSE message looks like:
```
event: text
data: {"chunk": "Hello"}

event: text
data: {"chunk": " world"}

event: done
data: {}
```
Two newlines (`\n\n`) separate messages.

### Why SSE over WebSockets?
| | SSE | WebSocket |
|---|---|---|
| Direction | Server → Client only | Bidirectional |
| Protocol | Plain HTTP | Upgraded HTTP (ws://) |
| Browser support | Native `EventSource` | Native `WebSocket` |
| Reconnection | Automatic | Manual |
| Complexity | Low | Higher |

For LLM chat, the client sends one request and the server streams the response. That's one direction — SSE is the right tool. WebSockets are for real-time two-way communication (multiplayer games, chat apps with typing indicators, etc.).

### Why not just wait for the full response?
A Claude response can take 5–15 seconds. Without streaming, the user stares at a blank screen. With streaming, they see tokens appear within ~200ms of sending. The perceived speed difference is enormous.

### How it works in FastAPI
```python
# app/routers/chat.py

async def _stream_claude(req) -> AsyncGenerator[str, None]:
    async with client.messages.stream(...) as stream:
        async for event in stream:
            if event.delta.type == "text_delta":
                yield _sse("text", {"chunk": event.delta.text})
    yield _sse("done", {})

@router.post("/chat")
async def chat(req):
    return StreamingResponse(
        _stream_claude(req),
        media_type="text/event-stream",   # ← tells browser this is SSE
    )
```

`StreamingResponse` + an async generator = SSE. The generator `yield`s strings; FastAPI sends each one to the browser immediately without buffering.

### How the browser reads it
```javascript
// frontend/index.html
const resp   = await fetch('/chat', { method: 'POST', body: ... });
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are split by \n\n
    const parts = buffer.split('\n\n');
    buffer = parts.pop();  // keep the incomplete last part

    for (const part of parts) {
        // parse event type and data, update the DOM
    }
}
```
> **Why not use the native `EventSource` API?**
> `EventSource` only supports GET requests. Our `/chat` endpoint is POST (to send the user's message in the body). So we use `fetch` + `ReadableStream` to get the same streaming behaviour with POST.

---

## 4. LLM Tool Calling — The Two-Pass Pattern

**Files:** `app/routers/chat.py`, `app/tools/weather.py`

### What is tool calling?
Tool calling (also called function calling) lets an LLM say: *"I need external data to answer this. Call this function with these arguments and give me the result."*

The LLM does not call the function itself — it just outputs a structured request. Your code calls the function and feeds the result back to the LLM.

### The two-pass pattern

```
Pass 1: User → LLM
         ↓
         LLM decides: "I need weather data"
         LLM outputs: tool_use { name: "get_weather", input: { city: "Mumbai" } }
         (LLM stops generating text here)

Your code: calls get_weather("Mumbai") → gets real data from wttr.in

Pass 2: User + tool result → LLM again
         ↓
         LLM now has real data and generates the final answer
         LLM streams: "The weather in Mumbai is 32°C and partly cloudy..."
```

### Why does this require two LLM calls?
The LLM cannot reach out to the internet itself. It's a text-in, text-out function. The "tool call" is just a structured JSON output that your application interprets and acts on. You are the bridge between the LLM and the real world.

### Code walkthrough (Anthropic)

```python
# Pass 1 — may return stop_reason="tool_use"
async with client.messages.stream(
    model=req.model,
    messages=[{"role": "user", "content": req.message}],
    tools=[WEATHER_TOOL_DEFINITION],   # ← tell LLM what tools exist
) as stream:
    ...
    final = await stream.get_final_message()

# Did the LLM want to call a tool?
if final.stop_reason == "tool_use":
    tool_block = next(b for b in final.content if b.type == "tool_use")
    city = tool_block.input["city"]

    # YOUR code fetches the weather
    weather_data = await get_weather(city)   # hits wttr.in

    # Pass 2 — give LLM the tool result
    messages = [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": final.content},  # LLM's tool_use request
        {"role": "user",      "content": [{"type": "tool_result", ...}]},
    ]
    async with client.messages.stream(model=..., messages=messages) as stream2:
        # now streams the real answer
```

### Tool definition schema
```python
# app/tools/weather.py
WEATHER_TOOL_DEFINITION = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Mumbai'"}
        },
        "required": ["city"],
    },
}
```
The description matters — the LLM reads it to decide when and how to call the tool. Vague descriptions lead to wrong tool usage.

### Why wttr.in?
`https://wttr.in/Mumbai?format=j1` returns structured JSON weather data for any city — no API key, no sign-up, free forever. Perfect for demos and learning.

### Groq/xAI tool calling
Groq and xAI use the OpenAI tool format (slightly different from Anthropic):
```python
# OpenAI-compatible format
_OAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "...",
        "parameters": { ... }   # same JSON Schema as Anthropic's input_schema
    }
}
```
The two-pass logic is identical — only the message format changes.

---

## 5. Structured Extraction with instructor + Pydantic

**Files:** `app/routers/extract.py`, `app/schemas/models.py`

### The problem
LLMs output text. You often want structured data (a Python dict, a database row, a typed object). Getting an LLM to reliably output valid JSON with the exact fields you need is surprisingly hard.

### What Pydantic does
Pydantic is a data validation library. You define a schema as a Python class:
```python
class ContactInfo(BaseModel):
    full_name: Optional[str] = None
    email:     Optional[str] = None
    phone:     Optional[str] = None
    company:   Optional[str] = None
```
Pydantic enforces types, raises clear errors on invalid data, and serialises to/from JSON automatically.

### What instructor does
instructor wraps an LLM client and:
1. Converts your Pydantic model into a JSON Schema
2. Tells the LLM to output data that matches the schema
3. Parses the LLM's response into your Pydantic object
4. **Retries automatically** if the output is invalid

```python
import instructor

client = instructor.from_anthropic(anthropic.Anthropic(...))

result = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Priya Sharma, ML Engineer at Razorpay..."}],
    response_model=ContactInfo,   # ← your Pydantic class
)

print(result.email)   # "priya@razorpay.com" — fully typed
```

### Why not just ask the LLM to "output JSON"?
LLMs hallucinate keys, forget fields, add extra text, or output malformed JSON. instructor uses the provider's native structured output mechanism (tool use under the hood for Anthropic, JSON mode for Groq) which is far more reliable than prompting alone.

### Provider differences in instructor
```python
# Anthropic — uses tool_use internally
instructor.from_anthropic(anthropic.Anthropic(...))

# Groq — must use JSON mode (Groq doesn't support tool_use for structured output)
instructor.from_groq(groq.Groq(...), mode=instructor.Mode.JSON)

# xAI — OpenAI-compatible
instructor.from_openai(openai.OpenAI(..., base_url="https://api.x.ai/v1"))
```

### System prompt vs messages (a provider quirk)
```python
# Anthropic — system prompt is a top-level parameter
client.chat.completions.create(
    system="You are an extraction expert...",
    messages=[{"role": "user", "content": text}],
)

# Groq / xAI — system prompt goes in messages[]
client.chat.completions.create(
    messages=[
        {"role": "system",  "content": "You are an extraction expert..."},
        {"role": "user",    "content": text},
    ],
)
```
This is a real friction point when supporting multiple providers. The `_get_instructor()` factory in `extract.py` returns both the client and the provider name so the right call shape can be built.

---

## 6. Multi-Provider Routing

**File:** `app/routers/chat.py`

### The pattern
Rather than one client per request, we use lazy singletons cached per provider:
```python
_clients: dict[str, instructor.Instructor] = {}

def _get_instructor(model: str):
    if model.startswith("claude"):
        if "anthropic" not in _clients:
            _clients["anthropic"] = instructor.from_anthropic(...)
        return _clients["anthropic"], "anthropic"
```

**Why singletons?** Creating an API client object is cheap, but if you're under load (100 req/s), creating a new Anthropic() object on every request wastes time and memory. One client per provider, reused forever, is the right pattern.

### Routing by model prefix
```python
if   req.model.startswith("claude"):  → Anthropic
elif req.model.startswith(("llama", "mixtral", "gemma")):  → Groq
elif req.model.startswith("grok"):    → xAI
else: raise 400
```
This is intentionally simple. The model name encodes the provider. In production you might use a registry dict instead.

### Why is Groq free?
Groq doesn't train models — they built custom silicon (LPUs — Language Processing Units) that run open-source models (LLaMA, Mixtral, Gemma) extremely fast. They offer a free tier to attract developers. The models are the same as running LLaMA locally, but hosted and fast.

### Why does xAI use the OpenAI SDK?
xAI's API is OpenAI-compatible — it accepts the same request format, same endpoints, same streaming protocol. This is a common pattern: many providers clone the OpenAI API so existing OpenAI code works with just a `base_url` change.
```python
openai.OpenAI(api_key="xai-...", base_url="https://api.x.ai/v1")
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                only this line changes
```

---

## 7. Environment Variables and python-dotenv

**Files:** `app/main.py`, `.env.example`

### Why not hardcode API keys?
Never put API keys in source code. Reason: git history is permanent. Even if you delete the key in a later commit, it's still visible in the git log. Anyone who clones your repo (including GitHub's search crawlers) can find it.

### How dotenv works
```
.env          ← your actual keys, never committed (in .gitignore)
.env.example  ← template with placeholder values, committed to git
```

```python
# app/main.py — first two lines
from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ before anything else runs
```

Then anywhere in your app:
```python
key = os.getenv("ANTHROPIC_API_KEY")   # None if not set
```

### Why must load_dotenv() come first?
If any module import triggers `os.getenv()` before `load_dotenv()` runs, the key is `None`. Python imports run top-to-bottom, so `load_dotenv()` must execute before any import that reads env vars. That's why it's the very first code in `main.py`, even before the FastAPI import.

---

## 8. Frontend — Why No React?

**File:** `frontend/index.html`

### The choice
The entire frontend is ~400 lines in one HTML file — no build step, no `node_modules`, no bundler.

### Why not React/Vue/Svelte?
For this project:
- The UI has two tabs, one chat area, one extract panel — not a complex enough component tree to justify a framework
- A build step (npm install, webpack/vite, etc.) would add friction for learners just trying to run the project
- The Docker image stays small (no 200MB `node_modules` to copy in)
- The whole frontend deploys as a single static file served by FastAPI

React makes sense when you have: complex state across many components, client-side routing, large teams, or need a component library. None of those apply here.

### SSE from fetch (not EventSource)
The browser's built-in `EventSource` API only supports GET requests. Since `/chat` is a POST (to send the message body), we use `fetch` with a `ReadableStream` reader instead:
```javascript
const resp   = await fetch('/chat', { method: 'POST', body: JSON.stringify({...}) });
const reader = resp.body.getReader();
// read chunks, split on \n\n, parse event/data lines
```
This is more code than `EventSource` but works with any HTTP method.

### Why is the frontend served by FastAPI?
Convenience for deployment. One service, one `fly deploy`, one URL. In production with a separate frontend (e.g. Vercel), you'd serve the HTML from a CDN and point it at the FastAPI URL — the backend code doesn't change at all.

---

## 9. Deployment — Docker, Fly.io, Render

**Files:** `Dockerfile`, `fly.toml`, `render.yaml`

### Why Docker?
Docker packages your app + its Python version + all dependencies into one image that runs identically everywhere. No more "works on my machine."

```dockerfile
FROM python:3.12-slim      # exact Python version pinned
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # install first (layer cache)
COPY . .                   # copy code second — changes don't bust the dep cache
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer caching trick:** copy `requirements.txt` and install *before* copying the rest of the code. Docker caches each layer. If only your code changes (not requirements), the pip install layer is reused — builds go from 60s to 3s.

### Fly.io vs Render
| | Fly.io | Render |
|---|---|---|
| Config | `fly.toml` | `render.yaml` or UI |
| Free tier | Shared VMs, auto-sleep | 750 hrs/month web service |
| Cold start | ~2s | ~30s (free tier) |
| Region | Global, choose region | US/EU/Singapore |
| CLI | Yes (`flyctl`) | Optional |
| Best for | Always-on, low latency | Simplest setup |

Fly.io's `auto_stop_machines = true` in `fly.toml` means the VM sleeps when idle (saves money) and wakes on the next request (~2s cold start). Fine for demos.

### Why `--host 0.0.0.0`?
By default uvicorn listens on `127.0.0.1` (localhost only). Inside a Docker container, `0.0.0.0` means "accept connections from any network interface" — required for the container's port to be reachable from outside.

---

## 10. Concepts Glossary

| Term | What it means |
|------|--------------|
| **Token** | The unit LLMs process. Roughly 0.75 words. "Hello world" ≈ 2 tokens. You pay per token. |
| **Streaming** | Sending tokens to the client as they're generated instead of waiting for the full response |
| **SSE** | Server-Sent Events — HTTP protocol for server→client streaming |
| **Tool calling** | LLM outputs a structured "call this function" request; your code executes it and feeds the result back |
| **Two-pass** | First LLM call decides to use a tool; second LLM call generates the final answer with the tool's data |
| **instructor** | Python library that wraps LLM clients to return validated Pydantic objects |
| **Pydantic** | Python library for data validation using type hints |
| **AsyncGenerator** | A Python function with `yield` inside an `async def` — produces values asynchronously |
| **Singleton** | One instance of an object reused everywhere (e.g., one API client per provider) |
| **OpenAI-compatible** | An API that accepts the same request format as OpenAI's API (Groq, xAI, Ollama all do this) |
| **LPU** | Language Processing Unit — Groq's custom chip optimised for LLM inference |
| **JSON Schema** | A standard format for describing the structure of JSON data — used in tool definitions |
| **CORS** | Cross-Origin Resource Sharing — browser security policy; FastAPI's `CORSMiddleware` lets the frontend call the API |
| **Lazy singleton** | An object created only on first use, then cached (`if client is None: client = ...`) |

---

## 11. What to Learn Next

### Immediate next steps (build on this project)
- **Add conversation history** — currently each `/chat` call is stateless. Store messages in a list and pass the full history to the LLM on each call.
- **Add more tools** — calculator, currency converter, Wikipedia search. Each tool follows the same pattern as `get_weather`.
- **Stream the extract endpoint** — right now extract waits for the full response. You could stream partial JSON using instructor's streaming mode.
- **Add authentication** — protect the API with an API key header using FastAPI's `Header` dependency.

### Deeper concepts to study
| Topic | Why it matters |
|-------|---------------|
| **Prompt engineering** | The system prompt in `/extract` directly controls output quality. Learn few-shot prompting. |
| **Token limits & chunking** | Long documents won't fit in one LLM call. Learn to split text into chunks. |
| **Embeddings + vector search** | Store documents as vectors, find similar ones by meaning — the foundation of RAG. |
| **RAG (Retrieval-Augmented Generation)** | Feed retrieved documents to the LLM as context. Fixes the "I don't know recent events" problem. |
| **LangChain / LlamaIndex** | Higher-level frameworks for building LLM pipelines. Understand the primitives first (like this project), then frameworks make sense. |
| **Async Python** | `async def`, `await`, `asyncio.gather()` — essential for concurrent LLM calls. |
| **Pydantic v2 deep dive** | Validators, serializers, computed fields — Pydantic is everywhere in modern Python APIs. |

### Projects to build next
1. **PDF Q&A** — upload a PDF, chunk it, embed it, answer questions about it (RAG)
2. **Multi-tool agent** — give the LLM 5+ tools and let it decide which to call and in what order
3. **Structured data pipeline** — scrape unstructured web pages, extract structured data, store in a database
4. **Streaming dashboard** — poll multiple LLMs in parallel with `asyncio.gather()`, show which answers first

### References
- [Anthropic API docs](https://docs.anthropic.com) — tool use, streaming, prompt caching
- [Groq docs](https://console.groq.com/docs) — supported models, rate limits
- [instructor docs](https://python.useinstructor.com) — structured outputs, retrying, streaming
- [FastAPI docs](https://fastapi.tiangolo.com) — routing, dependencies, middleware
- [Pydantic docs](https://docs.pydantic.dev) — models, validators, JSON Schema
- [SSE spec (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — the protocol this project uses
