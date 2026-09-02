import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store.ts";

interface ShowcaseSlide {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  scenario: string;
  probe_max_residual_mm: number;
}

interface ShowcaseManifest {
  schema_version: 1;
  title: string;
  seed: number;
  source_commit: string;
  generated_at: string;
  validation: {
    accepted: number;
    attempts: number;
    rejection_rate: number;
    reach_audited: boolean;
    max_skinning_error_mm: number;
    pose_quality: {
      max_joint_rotation_deg: number;
      edge_length_ratio: [number, number];
      triangle_area_ratio: [number, number];
    };
    anatomy: {
      knee_angle_deg: [number, number];
      knee_bend_offset_m: [number, number];
      max_knee_off_axis_m: number;
      supported_elbow_angle_deg: [number, number];
      supported_wrist_angle_deg: [number, number];
    };
  };
  coverage: { bodies: number; poses: number; sites: number; designs: number };
  slides: ShowcaseSlide[];
}

const countTracePoints = (strokes: { length: number }[]) => strokes.reduce((sum, stroke) => sum + stroke.length, 0);

export function ShowcasePanel() {
  const [manifest, setManifest] = useState<ShowcaseManifest | null>(null);
  const [active, setActive] = useState(0);
  const loadScenario = useStore((s) => s.loadShowcaseScenario);
  const scenario = useStore((s) => s.showcaseScenario);
  const traceVisible = useStore((s) => s.showcaseTraceVisible);
  const toggleTrace = useStore((s) => s.toggleShowcaseTrace);
  const focus = useStore((s) => s.showcaseFocus);
  const toggleFocus = useStore((s) => s.toggleShowcaseFocus);
  const setError = useStore((s) => s.setError);

  useEffect(() => {
    const controller = new AbortController();
    fetch("showcase/manifest.json", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error(`showcase manifest: HTTP ${response.status}`)))
      .then((value: ShowcaseManifest) => setManifest(value))
      .catch((error: Error) => { if (error.name !== "AbortError") setError(error.message); });
    return () => controller.abort();
  }, [setError]);

  const slide = manifest?.slides[active];
  useEffect(() => {
    if (!slide) return;
    const controller = new AbortController();
    fetch(`showcase/${slide.scenario}`, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error(`${slide.scenario}: HTTP ${response.status}`)))
      .then(loadScenario)
      .catch((error: Error) => { if (error.name !== "AbortError") setError(error.message); });
    return () => controller.abort();
  }, [slide, loadScenario, setError]);

  const tracePoints = useMemo(
    () => scenario ? countTracePoints(scenario.trace.strokes) : 0,
    [scenario],
  );
  const site = scenario?.placement.site as { laterality?: string; id?: string } | undefined;
  const studioHref = `${window.location.pathname}${window.location.hash}`;

  return (
    <div className="showcase-panel">
      <header className="showcase-header">
        <div className="showcase-kicker"><span className="live-dot" /> procedural sim · milestone 01</div>
        <h1>{manifest?.title ?? "Rigged-body tattoo scenarios"}</h1>
        <p>Inspect the real Inkmap meshes as one surface-anchor contract moves from tattoo intent to a posed, simulation-ready toolpath.</p>
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

      {slide && scenario && (
        <section className="showcase-section scenario-detail">
          <div className="section-label"><span>02</span> resolved scenario</div>
          <p>{slide.description}</p>
          <dl>
            <div><dt>body</dt><dd>{scenario.body.id.replace("hbm-", "").replace("-stylized", "")}</dd></div>
            <div><dt>pose</dt><dd>{scenario.pose.id.replaceAll("-", " ")}</dd></div>
            <div><dt>tattoo</dt><dd>{scenario.design.name} · {scenario.placement.size_mm.map(Math.round).join(" × ")} mm</dd></div>
            <div><dt>surface</dt><dd>{site?.laterality} {site?.id?.replaceAll("_", " ")}</dd></div>
            <div><dt>toolpath</dt><dd>{scenario.trace.strokes.length} strokes · {tracePoints.toLocaleString()} anchors</dd></div>
            <div><dt>reach probe</dt><dd>{slide.probe_max_residual_mm.toFixed(4)} mm max</dd></div>
          </dl>
          <div className="showcase-view-controls">
            <button className={focus ? "trace-toggle active" : "trace-toggle"} type="button" onClick={toggleFocus}>
              {focus ? "full body overview" : "focus tattoo site"}
            </button>
            <button className={traceVisible ? "trace-toggle active" : "trace-toggle"} type="button" onClick={toggleTrace}>
              <span className="trace-swatch" /> {traceVisible ? "toolpath on" : "toolpath off"}
            </button>
          </div>
          <p className="hash">trace {scenario.trace.sha256.slice(0, 16)}…</p>
        </section>
      )}

      <section className="showcase-section">
        <div className="section-label"><span>03</span> contract chain</div>
        <ol className="contract-chain">
          <li className="done"><span>Inkmap</span><small>SVG + body site</small></li>
          <li className="done"><span>Rig</span><small>named pose</small></li>
          <li className="done"><span>Surface</span><small>face + barycentric</small></li>
          <li className="done"><span>Trace</span><small>metric toolpath</small></li>
          <li className="next"><span>Episode</span><small>GPU shard pending</small></li>
        </ol>
      </section>

      {manifest && (
        <section className="evidence-card">
          <div><span className="status-pill">CPU validated</span><span>seed {manifest.seed}</span></div>
          <strong>{manifest.validation.accepted} accepted / {manifest.validation.attempts} attempts</strong>
          <p>{(manifest.validation.rejection_rate * 100).toFixed(2)}% explicit reach rejection · browser/Python skinning ≤ {manifest.validation.max_skinning_error_mm.toFixed(1)} mm</p>
          <p>
            pose gate: joints ≤ {manifest.validation.pose_quality.max_joint_rotation_deg.toFixed(1)}° · edges {manifest.validation.pose_quality.edge_length_ratio.map((value) => value.toFixed(3)).join("–")}× · areas {manifest.validation.pose_quality.triangle_area_ratio.map((value) => value.toFixed(3)).join("–")}×
          </p>
          <p>
            anatomy gate: knees {manifest.validation.anatomy.knee_angle_deg.map((value) => value.toFixed(1)).join("–")}° with {manifest.validation.anatomy.knee_bend_offset_m.map((value) => Math.round(value * 1000)).join("–")} mm sagittal bend · elbows {manifest.validation.anatomy.supported_elbow_angle_deg.map((value) => value.toFixed(1)).join("–")}° · wrists {manifest.validation.anatomy.supported_wrist_angle_deg.map((value) => value.toFixed(1)).join("–")}°
          </p>
          <p className="qualification">Kinematic-contact validation only. No deformable skin, MediaPipe input, powered arm, or GPU episode is represented here.</p>
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
  const scenario = useStore((s) => s.showcaseScenario);
  const visible = useStore((s) => s.showcaseTraceVisible);
  const focus = useStore((s) => s.showcaseFocus);
  if (!scenario) return <div className="showcase-hud">loading scenario…</div>;
  return (
    <div className="showcase-hud">
      <span className="eyebrow">live 3D evidence</span>
      <strong>{scenario.pose.id.replaceAll("-", " ")}</strong>
      <span>{focus ? "site detail" : "full body overview"} · drag to orbit · scroll to zoom</span>
      {visible && <span className="legend"><i /> cyan = compiled surface trace</span>}
    </div>
  );
}
