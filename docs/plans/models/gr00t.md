---
summary: Gr00t model links, training, and evaluation steps
tags: [models, gr00t]
updated: 2025-08-21
audience: [dev]
---

# Gr00t

- [blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [model](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- [repo](https://github.com/NVIDIA/Isaac-GR00T)
- [tatbot fork](https://github.com/hu-po/Isaac-GR00T)

## ⚡ Train

instructions for `oop`

```bash
# basic install
git clone --depth 1 https://github.com/hu-po/Isaac-GR00T.git && \
cd Isaac-GR00T/
# setup uv venv
uv venv --python=3.11 && \
source .venv/bin/activate && \
uv pip install .[base]
# download dataset locally
export DATASET_DIR="/home/oop/tatbot/output/train/tatbot-calib-test/dataset" && \
huggingface-cli download \
  --repo-type dataset tatbot/tatbot-calib-test \
  --local-dir $DATASET_DIR
# copy modality config file
cp /home/oop/tatbot/config/gr00t_modality.json $DATASET_DIR/meta/modality.json
# load dataset
python scripts/load_dataset.py \
  --dataset-path $DATASET_DIR \
  --embodiment-tag new_embodiment \
  --plot-state-action \
  --steps 64 \
  --video-backend torchvision_av
# train with docker
docker build -f Dockerfile -t gr00t-train .
docker run -it --gpus all --shm-size=8g --rm \
  -e WANDB_RUN_ID="gr00t-test" \
  -e WANDB_PROJECT="tatbot-calib" \
  -v $DATASET_DIR:/dataset \
  -v $HF_HOME:/root/.cache/huggingface \
  -v /home/oop/tatbot/output/train/tatbot-calib-test/gr00t:/output \
  -v /home/oop/Isaac-GR00T:/workspace \
  gr00t-train \
  bash -c "pip install -e . --no-deps && \
  python scripts/gr00t_finetune.py \
    --dataset-path /dataset \
    --embodiment-tag new_embodiment \
    --num-gpus 1 \
    --output-dir /output \
    --max-steps 10000 \
    --data-config tatbot \
    --batch_size 1 \
    --video-backend torchvision_av"
```

## 🖥️ Eval

instructions for `ojo`, acting as the policy server

```bash
# basic install
git clone https://github.com/hu-po/Isaac-GR00T.git && \
cd Isaac-GR00T/
# copy policy checkpoint into ojo
scp oop@192.168.1.53:/home/oop/tatbot/output/train/tatbot-calib-test/gr00t /tmp/gr00t
# policy with dockerfile
docker build -f orin.Dockerfile -t gr00t-eval .
docker run -it --gpus all --rm \
  -v /tmp/gr00t:/checkpoint \
  -v /home/ojo/Isaac-GR00T:/workspace \
  gr00t-eval \
  bash -c "pip3 install .[orin] && \
  python scripts/inference_service.py --server \
    --model_path /checkpoint \
    --embodiment-tag new_embodiment \
    --data-config tatbot \
    --denoising-steps 4"
```

instructions for `eek` acting as the robot client

```bash
git clone https://github.com/hu-po/Isaac-GR00T.git && \
cd Isaac-GR00T/
# setup uv venv
uv venv --python=3.11 && \
source .venv/bin/activate && \
uv pip install .[base]
# run robot client
python getting_started/examples/eval_lerobot.py \
    --robot.type=tatbot \
    --policy_host=192.168.1.96 \
    --lang_instruction="move slightly upwards in z"
```
