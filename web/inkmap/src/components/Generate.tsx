import { useEffect, useRef, useState } from "react";
import { loadConfig, type AppConfig } from "../core/config.ts";
import { generateDesign, health, releaseGeneratedPreview, svgDataUrl, type Generated, type Health } from "../core/gen.ts";
import { stylePrompt, type TattooProgram } from "../core/lang.ts";
import { findDesignForMotif, useStore } from "../store.ts";

type Phase = { kind: "idle" } | { kind: "busy" } | { kind: "done"; g: Generated } | { kind: "error"; message: string };

/** Sidebar panel: a subject in, a traced black-ink design out, added to the picker. The drawing happens on the inkgen service. */
export function Generate() {
  const addDesign = useStore((s) => s.addDesign);
  const startPlacing = useStore((s) => s.startPlacing);
  const pending = useStore((s) => s.pending);
  const atlas = useStore((s) => s.atlas);
  const placeAt = useStore((s) => s.placeAt);
  const designs = useStore((s) => s.designs);
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [svc, setSvc] = useState<Health | "down" | null>(null);
  const [subject, setSubject] = useState("");
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [count, setCount] = useState(0);
  const abort = useRef<AbortController | null>(null);
  const mounted = useRef(true);

  // Check the generator at load and keep checking while it is down (it may be
  // starting, or the user may bring one up after opening the page).
  useEffect(() => {
    loadConfig().then(setCfg);
    let stop = false;
    const probe = () => health().then((h) => { if (!stop) setSvc(h); }).catch(() => { if (!stop) setSvc("down"); });
    probe();
    const id = window.setInterval(() => { if (svc === "down" || svc === null) probe(); }, 10_000);
    return () => { stop = true; window.clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => () => {
    mounted.current = false;
    abort.current?.abort();
  }, []);
  useEffect(() => {
    if (phase.kind !== "done") return;
    const generated = phase.g;
    return () => releaseGeneratedPreview(generated);
  }, [phase]);

  const run = (seed?: number, subj?: string, autoPlace = false) => {
    const s = (subj ?? subject).trim();
    if (!s || phase.kind === "busy") return;
    const controller = new AbortController();
    abort.current = controller;
    setPhase({ kind: "busy" });
    generateDesign(s, seed, undefined, controller.signal, pending ? stylePrompt(pending) : undefined)
      .then((g) => {
        if (!mounted.current || controller.signal.aborted) {
          releaseGeneratedPreview(g);
          return;
        }
        health().then(setSvc).catch(() => {});
        // A sentence-initiated generation lands on its site without a click.
        if (autoPlace) { keep(g, s); return; }
        setPhase({ kind: "done", g });
      })
      .catch((e: Error) => {
        if (!mounted.current) return;
        if (e.name === "AbortError") { setPhase({ kind: "idle" }); return; }
        const where = cfg ? ` (generator: ${cfg.api})` : "";
        setPhase({ kind: "error", message: e.message + (/Failed to fetch|HTTP 5|not answering/.test(e.message) ? where : "") });
        health().then(setSvc).catch(() => setSvc("down"));
      });
  };

  // A parsed sentence waiting for a design drives the generator itself: fill
  // the subject with its motif and draw, once per sentence, when a generator
  // is up. "A circle on the chest" should never wait for a button.
  const autoFor = useRef<TattooProgram | null>(null);
  useEffect(() => {
    if (!pending || autoFor.current === pending) return;
    if (phase.kind === "busy" || !svc || svc === "down") return;
    if (findDesignForMotif(designs, pending.motif)) return; // SentenceBar places these itself
    autoFor.current = pending;
    setSubject(pending.motif);
    run(undefined, pending.motif, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending, svc, designs, phase.kind]);

  const keep = (g: Generated, subj?: string) => {
    const id = `gen-${Date.now().toString(36)}-${count + 1}`;
    const long = 70;
    const size: [number, number] = g.width >= g.height ? [long, (long * g.height) / g.width] : [(long * g.width) / g.height, long];
    const name = (subj ?? subject).trim().slice(0, 24) || "generated";
    addDesign({
      id, name, path: svgDataUrl(g.svg), default_size_mm: size,
      embedded: { name, svg: g.svg, default_size_mm: size, source: { kind: "generated", model: svc && svc !== "down" ? svc.model : "inkgen", prompt: g.prompt, seed: g.seed } },
    });
    // Auto-placement never enters the "done" phase whose effect owns this
    // preview URL. Revoke here as well; duplicate revocation is harmless for
    // the interactive path.
    releaseGeneratedPreview(g);
    setCount(count + 1);
    setPhase({ kind: "idle" });
    // A waiting sentence lands the new design on its named site directly;
    // otherwise the user places it by hand.
    if (pending && atlas) {
      try {
        placeAt(id, atlas.anchorForPhrase(pending.site), pending);
        return;
      } catch (e) {
        console.warn("[inkmap] sentence placement failed:", (e as Error).message);
      }
    }
    startPlacing(id);
  };

  const down = svc === "down";
  return (
    <section className="generate">
      <h2>Generate a design</h2>
      <div className="genrow">
        <input
          type="text" value={subject} placeholder="a swallow with a rose" maxLength={120}
          disabled={phase.kind === "busy"}
          onChange={(e) => setSubject(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") run(); e.stopPropagation(); }}
        />
        <button type="button" className="primary" disabled={phase.kind === "busy" || !subject.trim()} onClick={() => run()}>Generate</button>
      </div>
      {phase.kind === "busy" && (
        <div className="progress" role="status">
          <span>drawing{svc && svc !== "down" && svc.zero_gpu ? " on a shared GPU — a few seconds, longer if there is a queue" : "…"}</span>
          <div className="bar"><div className="indeterminate" /></div>
          <button type="button" className="link" onClick={() => abort.current?.abort()}>cancel</button>
        </div>
      )}
      {phase.kind === "error" && <p className="error">{phase.message}</p>}
      {phase.kind === "done" && (
        <div className="result">
          <div className="pair">
            <figure><img src={phase.g.pngUrl} alt="" /><figcaption>model output · {phase.g.seconds.toFixed(1)} s</figcaption></figure>
            <figure><img src={svgDataUrl(phase.g.svg)} alt="" /><figcaption>traced ink · {Math.round(phase.g.coverage * 100)}% cover</figcaption></figure>
          </div>
          <div className="genactions">
            <button type="button" className="primary" onClick={() => keep(phase.g)}>Use this design</button>
            <button type="button" onClick={() => run()}>Try another</button>
            <span className="muted small">seed {phase.g.seed}</span>
          </div>
        </div>
      )}
      {(svc === null || down) && (
        <p className="muted small">
          {svc === null && "checking the generator…"}
          {down && cfg && <>no generator at <code>{cfg.api}</code> yet — start one with <code>tatbot inkgen serve</code>, or run <code>tatbot inkmap dev --api &lt;url&gt;</code> (or add <code>?api=&lt;url&gt;</code> to this page) to use one elsewhere; checking again every 10 s</>}
        </p>
      )}
    </section>
  );
}
