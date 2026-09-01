import { BODIES } from "../core/body.ts";
import { FANTASY_TONES, SKIN_TONES, useStore } from "../store.ts";

/** The bar at the bottom of the viewport: which body, and what skin it wears. */
export function BodyBar() {
  const bodyId = useStore((s) => s.bodyId);
  const setBodyId = useStore((s) => s.setBodyId);
  const skinTone = useStore((s) => s.skinTone);
  const setSkinTone = useStore((s) => s.setSkinTone);
  const atlas = useStore((s) => s.atlas);
  const showAtlas = useStore((s) => s.showAtlas);
  const toggleAtlas = useStore((s) => s.toggleAtlas);
  const isActive = (hex: string) => hex.toLowerCase() === skinTone.toLowerCase();
  const swatch = (hex: string) => (
    <button
      key={hex}
      type="button"
      role="radio"
      aria-checked={isActive(hex)}
      className={isActive(hex) ? "tone active" : "tone"}
      style={{ background: hex }}
      title={hex}
      onClick={() => setSkinTone(hex)}
    />
  );
  return (
    <div className="bodybar">
      <div className="bodies" role="radiogroup" aria-label="body">
        {BODIES.map((b) => (
          <button
            key={b.id}
            type="button"
            role="radio"
            aria-checked={b.id === bodyId}
            className={b.id === bodyId ? "body active" : "body"}
            title={b.label}
            onClick={() => setBodyId(b.id)}
          >
            {b.glyph}
          </button>
        ))}
        <button
          type="button"
          className={showAtlas ? "body active" : "body"}
          title={atlas ? "toggle the body-site atlas" : "no region atlas for this body"}
          aria-pressed={showAtlas}
          disabled={!atlas}
          onClick={toggleAtlas}
        >
          ▦
        </button>
      </div>
      <div className="tones" role="radiogroup" aria-label="skin tone">
        {SKIN_TONES.map(swatch)}
        <span className="sep" aria-hidden />
        {FANTASY_TONES.map(swatch)}
        <label className="tone custom" title="custom colour" style={{ background: skinTone }}>
          <input type="color" value={skinTone} onChange={(e) => setSkinTone(e.target.value)} aria-label="custom skin tone" />
          <span>+</span>
        </label>
      </div>
    </div>
  );
}
