---
summary: How the agent runs and how to use it
tags: [agent, mcp]
updated: 2025-08-21
audience: [dev, operator, agent]
---

# Agent

tatbot is controlled by an agent model.

```{admonition} Quick Reference
:class: tip
- Serve model (on `ojo`): `ollama serve`
- Configure opencode: set `OPENCODE_CONFIG=~/tatbot/opencode.json`
- Launch: `opencode`
```

## 🏃 Quick Start
- Install and serve a model with Ollama on `ojo`.
- Configure and run the opencode client locally.

## ⚙️ Setup (Ollama on `ojo`)

```bash
# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# In ~/.bashrc
export OLLAMA_HOST=0.0.0.0       # allow remote access
export OLLAMA_MODELS=/mnt/ollama # use NVMe (faster, more space)

# Start server and model
ollama serve
ollama pull gpt-oss:20b
ollama run gpt-oss:20b

# Verify tools capability
ollama show gpt-oss:20b

# Inspect running models
ollama list
ollama ps

# If issues arise
sudo systemctl restart ollama
```

## 🖥️ Usage (opencode)

The main agent interface is via [opencode](https://github.com/sst/opencode).

```bash
curl -fsSL https://opencode.ai/install | bash

# In ~/.bashrc
export OPENCODE_CONFIG=~/tatbot/opencode.json

opencode
```

## 🔗 Reference
- https://opencode.ai/docs/mcp-servers/
- https://opencode.ai/docs/agents/
- https://modelcontextprotocol.io/legacy/tools/inspector
