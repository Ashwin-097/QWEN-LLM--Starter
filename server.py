import configparser
import time
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

import model_loader
import json

# -------------------------------
# Load configuration
# -------------------------------
config = configparser.ConfigParser()
config.read("config.ini")

HOST = config.get("SERVER", "host", fallback="0.0.0.0")
PORT = config.getint("SERVER", "port", fallback=11434)

# Parse workspaces (strip inline comments)
WORKSPACES: List[str] = []
if "WORKSPACES" in config:
    for k, v in config.items("WORKSPACES"):
        clean_value = v.split('#')[0].split(';')[0].strip()
        if clean_value:
            WORKSPACES.append(clean_value)

# Parse memory switch
MEMORY_ENABLED = set()
if "WORKSPACES MEM SWITCH" in config:
    for k, v in config.items("WORKSPACES MEM SWITCH"):
        if k.lower() == "enabled":
            for name in v.split(","):
                MEMORY_ENABLED.add(name.strip())

# Load user memories JSON if needed
try:
    with open("user_memories.json", "r") as f:
        USER_MEMORIES = json.load(f)
except FileNotFoundError:
    USER_MEMORIES = {}

# -------------------------------
# Helpers
# -------------------------------
def validate_workspace(workspace: str):
    """Ensure workspace is allowed."""
    if workspace not in WORKSPACES:
        return {
            "error": f"Workspace '{workspace}' not allowed",
            "object": "error"
        }
    return None

def persist_memory(workspace: str, user_message: str, assistant_reply: str):
    """Persist memory for a workspace if enabled."""
    if workspace in MEMORY_ENABLED:
        if workspace not in USER_MEMORIES:
            USER_MEMORIES[workspace] = {"history": []}
        USER_MEMORIES[workspace]["history"].append(
            {"role": "user", "content": user_message}
        )
        USER_MEMORIES[workspace]["history"].append(
            {"role": "assistant", "content": assistant_reply}
        )
        with open("user_memories.json", "w") as f:
            json.dump(USER_MEMORIES, f, indent=2)

# -------------------------------
# FastAPI setup
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()
    yield
    model_loader.cleanup_vram()

app = FastAPI(title="Workspace-enabled Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request models
# -------------------------------
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str = model_loader.MODEL_ALIAS
    messages: List[Message]
    temperature: Optional[float] = model_loader.DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = model_loader.DEFAULT_MAX_TOKENS
    stream: Optional[bool] = False

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/")
async def root():
    return PlainTextResponse("Workspace-enabled Qwen server is running")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Default chat endpoint (no workspace isolation)."""
    reply = model_loader.generate_cli(
        [{"role": m.role, "content": m.content} for m in req.messages],
        user_id="user_default"
    )
    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": reply},
        "done": True
    }

@app.post("/workspace/{workspace}/chat")
async def chat_with_workspace(workspace: str, req: ChatRequest):
    """Workspace-scoped chat endpoint with optional memory persistence."""
    error = validate_workspace(workspace)
    if error:
        return error

    reply = model_loader.generate_cli(
        [{"role": m.role, "content": m.content} for m in req.messages],
        user_id=f"user_{workspace}"
    )

    persist_memory(workspace, req.messages[-1].content, reply)

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": workspace,
        "message": {"role": "assistant", "content": reply},
        "done": True
    }

@app.get("/models")
async def list_models():
    """Global model listing: all workspaces appear as models."""
    return {
        "data": [
            {
                "id": ws,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-server",
                "permission": []
            }
            for ws in WORKSPACES
        ]
    }

@app.get("/{workspace}/models")
async def list_workspace_models(workspace: str):
    """Workspace-scoped model listing."""
    error = validate_workspace(workspace)
    if error:
        return error
    return {
        "data": [
            {
                "id": workspace,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-server",
                "permission": []
            }
        ]
    }

@app.post("/{workspace}/chat/completions")
async def workspace_chat_completions(workspace: str, req: ChatRequest):
    """
    Workspace-scoped OpenAI-style completions.
    The workspace is taken from the path parameter.
    """
    error = validate_workspace(workspace)
    if error:
        return error

    reply = model_loader.generate_cli(
        [{"role": m.role, "content": m.content} for m in req.messages],
        user_id=f"user_{workspace}"
    )

    persist_memory(workspace, req.messages[-1].content, reply)

    if req.stream:
        def event_stream():
            yield f'data: {json.dumps({"id":"chatcmpl-123","object":"chat.completion.chunk","created":int(time.time()),"model":workspace,"choices":[{"delta":{"role":"assistant","content":reply}}]})}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": workspace,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }
        ]
    }

# -------------------------------
# Run server
# -------------------------------
def run_http_server():
    uvicorn.run(app, host=HOST, port=PORT, reload=False)

if __name__ == "__main__":
    run_http_server()
