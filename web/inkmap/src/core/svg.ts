// Designs are vectors. Each mounted decal owns one raster texture at a
// resolution that keeps ≥ 8 px/mm for any design up to 128 mm on its long
// side. Per-decal ownership is deliberate: mirror transforms mutate a texture,
// and a global cache both leaked inactive GPU textures and let one placement
// change another placement's appearance.
import * as THREE from "three";

const RASTER_PX = 1024;

export function svgTexture(url: string): Promise<THREE.CanvasTexture> {
  return rasterise(url);
}

async function rasterise(url: string): Promise<THREE.CanvasTexture> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`design ${url}: HTTP ${res.status}`);
  const svg = await res.text();
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const objUrl = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error(`design ${url}: SVG failed to decode`));
      im.src = objUrl;
    });
    const aspect = img.naturalWidth / img.naturalHeight || 1;
    const w = aspect >= 1 ? RASTER_PX : Math.round(RASTER_PX * aspect);
    const h = aspect >= 1 ? Math.round(RASTER_PX / aspect) : RASTER_PX;
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    tex.needsUpdate = true;
    return tex;
  } finally {
    URL.revokeObjectURL(objUrl);
  }
}
