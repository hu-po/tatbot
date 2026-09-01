import { MAX_WIDTH_MM, MIN_WIDTH_MM, useStore } from "../store.ts";

export function Inspector() {
  const placements = useStore((s) => s.placements);
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  const update = useStore((s) => s.update);
  const remove = useStore((s) => s.remove);
  const designs = useStore((s) => s.designs);
  const toFile = useStore((s) => s.toFile);
  const loadFile = useStore((s) => s.loadFile);
  const setError = useStore((s) => s.setError);
  const p = placements.find((x) => x.id === selected);
  const design = p && designs.find((d) => d.id === p.design_id);
  const aspect = design ? design.default_size_mm[0] / design.default_size_mm[1] : 1;

  const download = () => {
    const f = toFile();
    if (!f) return;
    const blob = new Blob([JSON.stringify(f, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `inkmap-${f.body.id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const upload = async (file: File | undefined) => {
    if (!file) return;
    try {
      loadFile(JSON.parse(await file.text()));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <section>
      <h2>Placements <span className="muted">({placements.length})</span></h2>
      <ul className="layers">
        {placements.map((x) => (
          <li key={x.id}>
            <button type="button" className={x.id === selected ? "active" : ""} onClick={() => select(x.id)}>
              {designs.find((d) => d.id === x.design_id)?.name ?? x.design_id} · {x.size_mm[0].toFixed(0)}×{x.size_mm[1].toFixed(0)} mm
            </button>
          </li>
        ))}
      </ul>
      {p && (
        <div className="controls">
          <label>
            width <output>{p.size_mm[0].toFixed(0)} mm</output>
            <input type="range" min={MIN_WIDTH_MM} max={MAX_WIDTH_MM} step={1} value={p.size_mm[0]}
              onChange={(e) => { const w = Number(e.target.value); update(p.id, { size_mm: [w, w / aspect] }); }} />
          </label>
          <label>
            rotation <output>{((p.rotation_rad * 180) / Math.PI).toFixed(0)}°</output>
            <input type="range" min={-180} max={180} step={1} value={(p.rotation_rad * 180) / Math.PI}
              onChange={(e) => update(p.id, { rotation_rad: (Number(e.target.value) * Math.PI) / 180 })} />
          </label>
          <label className="row">
            <input type="checkbox" checked={p.mirror} onChange={(e) => update(p.id, { mirror: e.target.checked })} /> mirror
          </label>
          {p.language && <p className="muted caption">“{p.language.sentence}”</p>}
          {p.site && !p.language && <p className="muted caption">{[p.site.laterality, p.site.level, p.site.aspect, p.site.id.replace(/_/g, " ")].filter(Boolean).join(" ")}</p>}
          <p className="muted mono">face {p.anchor.face} · bary {p.anchor.barycentric.map((w) => w.toFixed(2)).join(" ")}</p>
          <button type="button" className="danger" onClick={() => remove(p.id)}>delete</button>
        </div>
      )}
      <div className="io">
        <button type="button" disabled={!placements.length} onClick={download}>download JSON</button>
        <label className="file">
          load JSON
          <input type="file" accept="application/json" onChange={(e) => upload(e.target.files?.[0])} />
        </label>
      </div>
    </section>
  );
}
