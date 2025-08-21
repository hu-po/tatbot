---
summary: SmolVLA model references, training, and evaluation
tags: [models, smolvla]
updated: 2025-08-21
audience: [dev]
---

# SmolVLA

- [blog](https://huggingface.co/blog/smolvla)
- [model](https://huggingface.co/lerobot/smolvla_base)
- [tatbot fork](https://github.com/hu-po/lerobot)
- smolvla is part of the lerobot repo

## ⚡ Train

instructions for `oop`

```bash
# basic install
git clone --depth=1 https://github.com/hu-po/lerobot.git && \
cd lerobot/
# setup uv venv
uv venv && \
source .venv/bin/activate && \
uv pip install -e ".[smolvla]"
# run training
wandb login
uv run python ~/lerobot/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=tatbot/tatbot-calib-test \
  --output_dir=~/tatbot/output/train/calib-test/smolvla \
  --batch_size=64 \
  --wandb.enable=true \
  --wandb.project=tatbot-calib \
  --steps=1000
```

## 🖥️ Eval

instructions for `eek` performing model inference and running robot

```bash
# TODO
```
