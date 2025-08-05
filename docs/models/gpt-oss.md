# GPT-OSS

open source openai model used for chatbot.

using transformers

```
source scripts/setup_env.sh
uv pip install transformers kernels torch aiohttp transformers[serving]
uv run transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-20b
```

using vllm

```
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b
````

ollama

```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull gpt-oss:20b
ollama run gpt-oss:20b

# use custom model cache
OLLAMA_MODELS=/mnt/ollama ollama serve
```

ggml

