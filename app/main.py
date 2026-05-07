from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import chat, extract

ROOT = Path(__file__).parent.parent   # repo root, works regardless of cwd
FRONTEND = ROOT / "frontend"

app = FastAPI(
    title="Synapse — Just a First AI",
    description=(
        "FastAPI service with streaming chat (SSE + tool calling) "
        "and structured extraction via instructor + Pydantic."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, tags=["Chat"])
app.include_router(extract.router, tags=["Extract"])

# Serve frontend — absolute path so it works from any working directory
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
