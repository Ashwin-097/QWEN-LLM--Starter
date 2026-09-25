# QWEN-LLM--Starter

# Workspace-enabled FastAPI Server

A lightweight FastAPI server that provides chat endpoints with **workspace isolation** and optional **memory persistence**. Designed to integrate with local LLMs and mimic OpenAI-style `/chat/completions` APIs.

---

## Features
- Multiple workspaces with isolated contexts
- Optional memory persistence per workspace (`user_memories.json`)
- OpenAI-compatible `/chat/completions` endpoint
- Streaming responses via Server-Sent Events
- Configurable host/port via `config.ini`

---

## Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/my-workspace-server.git
cd my-workspace-server

# Install dependencies
pip install -r requirements.txt
ini
# Configure config.ini
[SERVER]
host = 0.0.0.0
port = 11434

[WORKSPACES]
test1 = enabled
test2 = enabled

[WORKSPACES MEM SWITCH]
enabled = test1
bash
# Run the server
python server.py
Usage
Default Chat: POST /api/chat

Workspace Chat: POST /workspace/{workspace}/chat

OpenAI-style Completions: POST /chat/completions

List Models: GET /models or GET /{workspace}/models

Memory Persistence
Enabled per workspace in [WORKSPACES MEM SWITCH]

Stores conversation history in user_memories.json

Useful for stateless clients that don’t replay history
