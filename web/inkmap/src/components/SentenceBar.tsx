import { useState } from "react";
import { findDesignForMotif, useStore } from "../store.ts";
import { parseSentence, realize, type TattooProgram } from "../core/lang.ts";

/** Say the tattoo: "a fine line octopus on the left knee ditch". The sentence
 *  parses to a program; the named region glows; a design whose name matches
 *  the motif is placed there at once, otherwise the program waits (glowing)
 *  for a picked or generated design, and the next placement inherits it. */
export function SentenceBar() {
  const atlas = useStore((s) => s.atlas);
  const designs = useStore((s) => s.designs);
  const pending = useStore((s) => s.pending);
  const setPending = useStore((s) => s.setPending);
  const placeAt = useStore((s) => s.placeAt);
  const setToast = useStore((s) => s.setToast);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const go = () => {
    if (!text.trim()) return;
    let program: TattooProgram;
    try {
      program = parseSentence(text);
      setErr(null);
    } catch (e) {
      setErr((e as Error).message);
      return;
    }
    setPending(program);
    if (!atlas) {
      setErr("this body has no region atlas — sites cannot be grounded");
      return;
    }
    const design = findDesignForMotif(designs, program.motif);
    if (design) {
      try {
        placeAt(design.id, atlas.anchorForPhrase(program.site), program);
        setText("");
      } catch (e) {
        setErr((e as Error).message);
      }
    } else {
      // The Generate panel picks the waiting sentence up on its own when a
      // generator is reachable and lands the result on the glowing region.
      setToast(`drawing “${program.motif}”…`);
      setText("");
    }
  };

  return (
    <section className="sentence">
      <h2>Say it</h2>
      <div className="genrow">
        <input
          type="text" value={text} maxLength={160}
          placeholder="a fine line octopus on the left knee ditch"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") go(); e.stopPropagation(); }}
        />
        <button type="button" className="primary" disabled={!text.trim()} onClick={go}>Place</button>
      </div>
      {err && <p className="error">{err}</p>}
      {pending && (
        <p className="muted small">
          waiting: <em>{realize(pending)}</em>{" "}
          <button type="button" className="link" onClick={() => setPending(null)}>clear</button>
        </p>
      )}
    </section>
  );
}
