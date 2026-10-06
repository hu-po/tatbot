"""Turn a FLUX 3 Action policy package into a trunk that export_base can start a finetune from.

A package (black-forest-labs/flux-3-action-so101 or -droid, or a run of our
own) saves the whole policy, its DiT under ``dit.``; export_base wants the DiT
alone. The package's embodiment heads come along -- they load where their
shapes match this arm's (SO-101's six joint channels do) -- unless
``--fresh-heads`` drops them for the reference initialization.

    python package_trunk.py ~/travel/weights/flux-3-action-so101/model.safetensors \\
        ~/travel/weights/so101-trunk.safetensors
"""

from __future__ import annotations

import argparse

from safetensors import safe_open
from safetensors.torch import save_file

HEADS = ("emb_in.action", "final_layer.action")  # the action modality's heads, *_cond included


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", help="a policy package's model.safetensors")
    parser.add_argument("out", help="trunk .safetensors to write")
    parser.add_argument("--fresh-heads", action="store_true", help="drop the package's action heads")
    args = parser.parse_args()
    tensors = {}
    with safe_open(args.package, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 -- safe_open handles are not mappings
            name = key.removeprefix("dit.")
            if name == key or (args.fresh_heads and name.startswith(HEADS)):
                continue
            tensors[name] = f.get_tensor(key)
    save_file(tensors, args.out, metadata={"source": args.package})
    print(f"{len(tensors)} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
