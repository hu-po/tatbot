# Agent

tatbot is controlled by an agent model.

currently use [ollama](https://github.com/ollama/ollama) to serve agent on `ojo` node:

```bash
# install ollama
curl -fsSL https://ollama.com/install.sh | sh

# put into bashrc
export OLLAMA_HOST=0.0.0.0 # allows remote access
export OLLAMA_MODELS=/mnt/ollama # use nvme (faster, more space)

# start ollama server
ollama serve
ollama pull gpt-oss:20b
ollama run gpt-oss:20b

# verify that the model has the "tools" capability
ollama show gpt-oss:20b

# examine running models
ollama list
ollama ps

# if issues arise, restart ollama
sudo systemctl restart ollama
```

the main agent interface is via [opencode](https://github.com/sst/opencode)

```bash
curl -fsSL https://opencode.ai/install | bash

# add to bashrc
export OPENCODE_CONFIG=~/tatbot/config/opencode.json

opencode
```

helpful links:
- https://opencode.ai/docs/mcp-servers/
- https://opencode.ai/docs/agents/