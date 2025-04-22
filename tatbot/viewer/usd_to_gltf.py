import sys
from pathlib import Path

try:
    from usd2gltf.converter import Converter   # type: ignore
except ImportError:
    sys.exit("❌  'usd2gltf' is not installed inside this container.  Run: pip install usd2gltf")

def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: usd_to_gltf.py <scene.usd/.usdz> <output.gltf|glb>")

    src, dst = map(Path, sys.argv[1:3])

    conv = Converter()            # new OO‑style API
    stage = conv.load_usd(str(src))
    conv.process(stage, str(dst))

    print(f"✅  Wrote {dst.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()