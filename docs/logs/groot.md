# Groot N1

https://github.com/NVIDIA/Isaac-GR00T/tree/main

python scripts/inference_service.py --model_path nvidia/GR00T-N1-2B --server

https://github.com/dmlc/decord?tab=readme-ov-file#linux

https://catalog.ngc.nvidia.com/containers?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=

jetson-containers groot image, inference via huggingface

https://github.com/dusty-nv/jetson-containers/tree/master/packages/robots/Isaac-GR00T

nvidia has multiple heuristic sim datasets

https://huggingface.co/collections/nvidia/physical-ai-67c643edbb024053dcbcd6d8

pulling groot image into ojo

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
jetson-containers run $(autotag isaac-gr00t)
```

gr00t has finetuning scripts, with LoRA support via peft

https://github.com/NVIDIA/Isaac-GR00T/blob/153ee6fee23d4c3bf52cb4b520ddd258f98d017b/scripts/gr00t_finetune.py

they have their own server and client implementation in python only

https://github.com/NVIDIA/Isaac-GR00T/blob/153ee6fee23d4c3bf52cb4b520ddd258f98d017b/scripts/inference_service.py

the dataset is lerobot format