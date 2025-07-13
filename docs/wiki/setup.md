# Setup

```bash
# Basic install
git clone --depth=1 https://github.com/hu-po/tatbot.git && \
cd ~/tatbot && \
git pull && \
# Optional: Clean old uv environment
deactivate && \
rm -rf .venv && \
rm uv.lock && \
# Setup new uv environment
uv venv --prompt="tatbot" && \
source .venv/bin/activate && \
uv pip install .
# source env variables (i.e. keys, tokens, camera passwords)
source .env
```

Turn on the robot

1. flip power strip in the back to on.
2. press power button on `trossen-ai`, it will glow blue.
3. flip rocker switches to "on" on `arm-r` and `arm-l` control boxes underneath workspace.
4. flip rocker switch on the back of the light to turn it on. 