uv run python ~/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=hu-po/tatbot-stencil-1749648845 \
  --output_dir=~/tatbot/outputs/train \
  --batch_size=64 \
  --wandb.enable=true \
  --steps=20000