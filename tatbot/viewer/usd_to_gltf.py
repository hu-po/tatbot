# tatbot/viewer/usd_to_gltf.py
import sys
from pathlib import Path
from usd2gltf import converter as conv_mod

# ---------------------------------------------------------
# 🚑  Work‑around for missing globals in usd2gltf 0.3.x
conv_mod.point_instancers = []
conv_mod.point_instancer_prototypes = []
# ---------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: usd_to_gltf.py <scene.usd/.usdz> <output.glb>")
    src, dst = map(Path, sys.argv[1:3])

    conv = conv_mod.Converter()
    # turn off instancer handling if you don’t need it
    conv.convert_instancers = False

    stage = conv.load_usd(str(src))
    conv.process(stage, str(dst))
    print(f"✅  Wrote {dst.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()