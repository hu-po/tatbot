import { importArtwork } from "../exports.ts";

/** Native acquisition runs on the licensed installation, before importing into this editor. */
export function Generate({ onUse }: { onUse: (designId: string) => void; useLabel?: string }) {
  return <section className="generate" aria-label="Generate artwork">
    <p>Generate artwork with DrawingBot V3 at the physical size and pen settings you need.</p>
    <p className="small">Use the native acquisition command with a version 3 job. Import the resulting artwork.json to preview and place its frozen paths here.</p>
    <code>tatbot drawingbot generate job.json --out acquisition --app /path/to/drawingbotv3</code>
    <p className="small">Source images and SVGs belong in DBV3. Resizing an acquired preview requires regeneration before drawing.</p>
    <label className="file">Import artwork<input type="file" accept=".json,application/json" onChange={(e) => {
      void importArtwork(e.target.files?.[0]).then(id => { if (id) onUse(id); }); e.target.value = "";
    }} /></label>
  </section>;
}
