from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import chat, extract

app = FastAPI(
    title="AI Engineering Journey",
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


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"status": "ok", "docs": "/docs"})


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
