import { useEffect, useState } from "react";
import { useStore, type ShowcaseSlide } from "../store.ts";

interface ShowcaseManifest {
  schema_version: 3;
  title: string;
  source_commit: string;
  generated_at: string;
  validation: {
    placements: number;
    body_asset_sha256: string;
    pose_asset_sha256: string;
    visual_review: "pending" | "accepted";
  };
  coverage: { bodies: number; poses: number; sites: number; designs: number };
  slides: ShowcaseSlide[];
}

export function ShowcasePanel() {
  const [manifest, setManifest] = useState<ShowcaseManifest | null>(null);
  const [active, setActive] = useState(0);
  const loadSlide = useStore((s) => s.loadShowcaseSlide);
  const shown = useStore((s) => s.showcaseSlide);
  const focus = useStore((s) => s.showcaseFocus);
  const toggleFocus = useStore((s) => s.toggleShowcaseFocus);
  const setError = useStore((s) => s.setError);
  const slide = manifest?.slides[active];
  const design = useStore((s) => slide ? s.designs.find((d) => d.id === slide.artwork_id && d.embedded) : undefined);

  useEffect(() => {
    const controller = new AbortController();
    fetch("showcase/manifest.json", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error(`showcase manifest: HTTP ${response.status}`)))
      .then((value: ShowcaseManifest) => setManifest(value))
      .catch((error: Error) => { if (error.name !== "AbortError") setError(error.message); });
    return () => controller.abort();
  }, [setError]);

  useEffect(() => {
    // A pose change reloads the body; once this slide is shown, that reload
    // must not place it again and reset the user's camera focus.
    if (!slide || !design || shown?.id === slide.id) return;
    try {
      loadSlide(slide);
    } catch (error) {
      setError((error as Error).message);
    }
  }, [slide, design, loadSlide, setError, shown?.id]);

  const site = shown?.placement.site as { laterality?: string; id?: string } | undefined;
  const studioHref = `${window.location.pathname}${window.location.hash}`;

  return (
    <div className="showcase-panel">
      <header className="showcase-header">
        <div className="showcase-kicker"><span className="live-dot" /> shared tattoo artwork</div>
        <h1>{manifest?.title ?? "Rigged-body tattoo placements"}</h1>
        <p>Inspect simple tattoo designs placed across five body poses.</p>
      </header>

      <section className="showcase-metrics" aria-label="delivered coverage">
        {([
          [manifest?.coverage.bodies ?? "—", "bodies"],
          [manifest?.coverage.poses ?? "—", "poses"],
          [manifest?.coverage.sites ?? "—", "sites"],
          [manifest?.coverage.designs ?? "—", "designs"],
        ] as const).map(([value, label]) => <div key={label}><b>{value}</b><span>{label}</span></div>)}
      </section>

      <section className="showcase-section">
        <div className="section-label"><span>01</span> pose gallery</div>
        <div className="scenario-list" role="list">
          {manifest?.slides.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={index === active ? "scenario-card active" : "scenario-card"}
              aria-pressed={index === active}
              onClick={() => setActive(index)}
            >
              <span className="scenario-index">{String(index + 1).padStart(2, "0")}</span>
              <span><small>{item.eyebrow}</small><strong>{item.title}</strong></span>
              <span className="scenario-arrow">↗</span>
            </button>
          ))}
        </div>
      </section>

      {slide && shown?.id === slide.id && (
        <section className="showcase-section scenario-detail">
          <div className="section-label"><span>02</span> placement</div>
          <p>{slide.description}</p>
          <dl>
            <div><dt>body</dt><dd>MHR through SOMA</dd></div>
            <div><dt>pose</dt><dd>{shown.pose_id.replaceAll("-", " ")}</dd></div>
            <div><dt>tattoo</dt><dd>{design?.name ?? shown.artwork_id} · {shown.placement.size_mm.map(Math.round).join(" × ")} mm</dd></div>
            <div><dt>surface</dt><dd>{site?.laterality} {site?.id?.replaceAll("_", " ")}</dd></div>
            <div><dt>qualification</dt><dd>offline geometry only</dd></div>
          </dl>
          <div className="showcase-view-controls">
            <button className={focus ? "trace-toggle active" : "trace-toggle"} type="button" onClick={toggleFocus}>
              {focus ? "full body overview" : "focus tattoo site"}
            </button>
          </div>
          <p className="hash">artwork {shown.artwork_sha256.slice(0, 16)}…</p>
        </section>
      )}

      <section className="showcase-section">
        <div className="section-label"><span>03</span> contract chain</div>
        <ol className="contract-chain">
          <li className="done"><span>Inkmap</span><small>artwork + body site</small></li>
          <li className="done"><span>SOMA</span><small>named pose</small></li>
          <li className="done"><span>Surface</span><small>face + barycentric</small></li>
          <li className="next"><span>Execution</span><small>separate measured surface</small></li>
        </ol>
      </section>

      {manifest && (
        <section className="evidence-card">
          <div><span className="status-pill">CPU validated</span></div>
          <strong>{manifest.validation.placements} SOMA-native placements</strong>
          <p>body {manifest.validation.body_asset_sha256.slice(0, 12)}… · poses {manifest.validation.pose_asset_sha256.slice(0, 12)}…</p>
          <p>Human visual review: {manifest.validation.visual_review}.</p>
          <p className="qualification">Offline nominal-body preview only. No measured-surface, reach, powered-arm, contact-mechanics, or human-use acceptance is represented here.</p>
        </section>
      )}

      <footer className="showcase-footer">
        <a href={studioHref}>Open the Inkmap editor <span>→</span></a>
        {manifest && <small>artifact {manifest.source_commit} · {manifest.generated_at.slice(0, 10)}</small>}
      </footer>
    </div>
  );
}

export function ShowcaseHud() {
  const slide = useStore((s) => s.showcaseSlide);
  const focus = useStore((s) => s.showcaseFocus);
  if (!slide) return <div className="showcase-hud">loading placement…</div>;
  return (
    <div className="showcase-hud">
      <span className="eyebrow">live 3D evidence</span>
      <strong>{slide.pose_id.replaceAll("-", " ")}</strong>
      <span>{focus ? "site detail" : "full body overview"} · drag to orbit · scroll to zoom</span>
    </div>
  );
}
