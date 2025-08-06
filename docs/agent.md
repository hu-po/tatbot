# Agent

tatbot is controlled by an agent model.

currently use [ollama](https://github.com/ollama/ollama) to serve agent on `ojo` node:

```bash
# install ollama
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_MODELS=/mnt/ollama # put into ~/.bashrc

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
opencode
```

set the opencode config file

```bash
export OPENCODE_CONFIG=~/tatbot/config/opencode.json
```

helpful links:
- https://opencode.ai/docs/mcp-servers/
- https://opencode.ai/docs/agents/