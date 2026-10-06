import { useEffect, useRef, useState } from "react";
import { intentFromSite, parsePlacement, resolveIntent, type InkLangIntent } from "../core/inklang/index.ts";
import { parseSentence, type TattooProgram } from "../core/lang.ts";
import { findDesignForMotif, useStore } from "../store.ts";

/** Resolve placement-only InkLang. A legacy full tattoo sentence is accepted
 * through the compatibility adapter, but its design fields never participate
 * in grounding. Interactive ambiguity must be accepted before placement.
 *
 * Either order works: a location first waits for artwork, artwork first
 * (`placing`) lands on the location the moment it resolves. */
export function SentenceBar({ compact = false }: { compact?: boolean }) {
  const atlas = useStore((s) => s.atlas);
  const designs = useStore((s) => s.designs);
  const pending = useStore((s) => s.pending);
  const placing = useStore((s) => s.placing);
  const setPending = useStore((s) => s.setPending);
  const choosePending = useStore((s) => s.choosePending);
  const placeAt = useStore((s) => s.placeAt);
  const setToast = useStore((s) => s.setToast);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const autoPlaced = useRef<object | null>(null);

  // Existing designs can land immediately, but only after the canonical
  // resolver has returned a concrete face+barycentric anchor (or the operator
  // has accepted one of its explicit choices). The artwork already in hand
  // (`placing`) counts as the design when the sentence names none.
  useEffect(() => {
    if (!pending || pending.resolution.status !== "resolved" || !pending.resolution.anchor) return;
    if (autoPlaced.current === pending.resolution) return;
    const design = pending.program ? findDesignForMotif(designs, pending.program.motif) : placing ? designs.find((d) => d.id === placing) : undefined;
    if (!design) return;
    autoPlaced.current = pending.resolution;
    placeAt(design.id, pending.resolution.anchor, pending);
  }, [designs, pending, placing, placeAt]);

  const go = () => {
    if (!text.trim()) return;
    if (!atlas) {
      setErr("this body has no region atlas — sites cannot be grounded");
      return;
    }
    let intent: InkLangIntent = parsePlacement(text);
    let program: TattooProgram | null = null;
    if (!intent.site) {
      try {
        program = parseSentence(text);
        intent = intentFromSite(text, program.site);
      } catch (error) {
        setErr((error as Error).message);
        setPending(null);
        return;
      }
    }
    const resolution = resolveIntent(intent, atlas, { id: "interactive", seed: null });
    const next = { program, intent, resolution };
    setPending(next);
    setText("");
    if (resolution.status === "rejected") {
      const issue = resolution.issues[0];
      setErr(issue ? `${issue.code}: ${issue.message}` : "InkLang could not resolve that placement");
      return;
    }
    setErr(null);
    if (resolution.status === "needs_choice") return;
    if (!program) {
      if (!placing) setToast("location resolved — choose artwork");
    } else if (!findDesignForMotif(designs, program.motif)) {
      setToast(`Generate “${program.motif}” with DrawingBot V3, then import artwork.json.`);
    }
  };

  return (
    <section className={compact ? "sentence compact" : "sentence"} aria-label="Describe a location">
      <div className="genrow">
        <input
          type="text" value={text} maxLength={160} aria-label="Describe a location"
          placeholder={placing ? "or describe where: on the left forearm" : "on the left knee ditch"}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") go(); if (event.key !== "Escape") event.stopPropagation(); }}
        />
        <button type="button" className={placing ? "" : "primary"} disabled={!text.trim() || !atlas} onClick={go}>Resolve</button>
      </div>
      {err && <p className="error small" role="alert">{err}</p>}
      {pending && (
        <div className={`resolution ${pending.resolution.status}`}>
          <p className="small"><span className="status">{pending.resolution.status === "needs_choice" ? "choose one" : pending.resolution.status}</span>{" "}<em>{pending.resolution.choice?.label ?? pending.intent.canonical_phrase ?? pending.intent.description}</em></p>
          {pending.resolution.status === "needs_choice" && (
            <div className="candidates" aria-label="Choose a concrete InkLang location">
              {pending.resolution.candidates.map((candidate, index) => (
                <button type="button" key={`${candidate.site_id}:${candidate.laterality}:${candidate.anchor.face}`} onClick={() => choosePending(index)}>
                  {candidate.label}
                </button>
              ))}
            </div>
          )}
          {pending.resolution.status === "resolved" && !pending.program && !placing && <p className="muted small">Choose artwork to place it at this exact location.</p>}
          <button type="button" className="link" onClick={() => setPending(null)}>clear location</button>
        </div>
      )}
    </section>
  );
}
