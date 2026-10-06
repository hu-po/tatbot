import { useEffect, useRef, useState } from "react";
import { artworkSizeM, artworkWidthM } from "../core/artwork-record.ts";
import { downloadJson } from "../core/download.ts";
import { feedbackDocument, loadReview, mergeObservations, observe, readObservations, REVIEW_LIMIT,
  reviewStoragePrefix, saveObservation, type LoadedReview, type Observation, type Preference, type ReviewEntry } from "../core/study-review.ts";
import { importArtwork } from "../exports.ts";
import { useStore } from "../store.ts";

interface Session { review: LoadedReview; observations: Observation[]; pending: Observation[] }
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const mm = (m: number) => Number((m * 1000).toFixed(3)).toString();
const duration = (seconds: number) => seconds < 60 ? `${Math.round(seconds)} sec` : `${(seconds / 60).toFixed(1)} min`;

export function StudyReview({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const current = useRef<Session | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState("");
  const ready = useStore(s => s.projectReady);
  function update(next: Session) { current.current = next; setSession(next); }

  useEffect(() => {
    const node = dialog.current!;
    if (open) node.showModal(); else node.close();
    return () => node.close();
  }, [open]);
  useEffect(() => {
    const sync = (event: StorageEvent) => {
      const s = current.current;
      if (!s || (event.key && !event.key.startsWith(reviewStoragePrefix(s.review)))) return;
      try { update({ ...s, observations: mergeObservations(s.observations, readObservations(localStorage, s.review)) }); }
      catch (e) { setError(`Feedback could not be restored: ${message(e)}`); }
    };
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (current.current?.pending.length) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("storage", sync); window.addEventListener("beforeunload", beforeUnload);
    return () => { window.removeEventListener("storage", sync); window.removeEventListener("beforeunload", beforeUnload); };
  }, []);

  async function openFile(file?: File) {
    if (!file || busy) return;
    if (current.current?.pending.length) { setError("Retry saving your pending feedback before opening another review. You can also download a recovery copy."); return; }
    setBusy(true); setError(null);
    try {
      if (file.size > REVIEW_LIMIT) throw new Error("Review exceeds 20 MiB");
      const review = await loadReview(await file.text());
      const observations = readObservations(localStorage, review);
      update({ review, observations, pending: [] }); setSource("");
    } catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  }

  function vote(entry: ReviewEntry, preference: Preference) {
    const s = current.current!;
    try {
      const observation = observe(s.review, entry, preference);
      const pending = [...s.pending, observation];
      update({ ...s, observations: mergeObservations(s.observations, [observation]), pending });
      retry();
    } catch (e) { setError(message(e)); }
  }

  function retry() {
    const s = current.current;
    if (!s) return;
    try {
      s.pending.forEach(o => saveObservation(localStorage, s.review, o));
      update({ ...s, observations: mergeObservations(s.observations, readObservations(localStorage, s.review)), pending: [] });
      setError(null);
    } catch (e) { setError(`Feedback save failed: ${message(e)}. Keep this tab open, retry or download feedback to keep your votes.`); }
  }

  async function download() {
    const s = current.current;
    if (!s) return;
    try { downloadJson(await feedbackDocument(s.review, s.observations), `inkmap-feedback-${s.review.bundle.content_sha256.slice(0, 12)}.json`); }
    catch (e) { setError(message(e)); }
  }

  async function add(entry: ReviewEntry) {
    const art = current.current!.review.bundle.artworks[entry.artwork_sha256];
    const id = await importArtwork(new File([JSON.stringify(art)], `${entry.source_case}.artwork.json`, { type: "application/json" }));
    if (!id) setError("Artwork could not be added to the library.");
  }

  const latest = new Map(session?.observations.map(o => [o.entry_id, o]) ?? []);
  const rated = [...latest.values()].filter(o => o.preference !== "unrated").length;
  const entries = session?.review.bundle.entries ?? [];
  return <dialog className="study-review" ref={dialog} aria-labelledby="review-title" onCancel={onClose}>
    <header className="review-header">
      <div><p className="review-eyebrow">Artwork study</p><h2 id="review-title">{session?.review.bundle.name ?? "Review a study"}</h2></div>
      <button type="button" onClick={onClose} aria-label="Close study review">Close</button>
    </header>
    <div className="review-toolbar">
      <label className="review-file">{busy ? "Opening…" : "Open review JSON"}<input aria-label="Open review JSON" type="file" accept="application/json,.json" disabled={busy || Boolean(session?.pending.length)} onChange={e => { void openFile(e.target.files?.[0]); e.target.value = ""; }} /></label>
      {session && <>
        <label>Source <select aria-label="Review source" value={source} onChange={e => setSource(e.target.value)}><option value="">All sources</option>{[...new Set(entries.map(e => e.source_case))].map(id => <option key={id}>{id}</option>)}</select></label>
        <span className="review-progress" role="status">{rated} / {entries.length} rated · {entries[0].source_split} · {session.pending.length ? "Unsaved feedback" : "Feedback saved locally"}</span>
        <button type="button" disabled={!session.observations.length} onClick={() => void download()}>Download feedback</button>
      </>}
    </div>
    {error && <div className="review-error" role="alert">{error}{session && <button type="button" onClick={retry}>Retry feedback save</button>}</div>}
    {!session ? <p className="review-empty">Open a study bundle to compare acquired artwork at its intended size. Your current placement stays in the editor.</p> : <>
      <p className="review-explanation">Enlarged previews use generation line width. Estimates come from ROS preparation; they are not measured drawing times. Add artwork to the library to place it in the editor.</p>
      <div className="review-grid">{entries.filter(e => !source || e.source_case === source).map(entry => {
        const art = session.review.bundle.artworks[entry.artwork_sha256], prep = entry.preparation;
        const preference = latest.get(entry.id)?.preference ?? "unrated";
        const size = artworkSizeM(art), preview = session.review.previews[entry.artwork_sha256];
        return <article className={`review-card ${preference}`} key={entry.id} data-entry-id={entry.id}>
          <div className="review-paper"><img alt={`${entry.source_case} — ${String(entry.recipe.pfm ?? art.name)}`} src={`data:image/svg+xml,${encodeURIComponent(preview.svg)}`} /></div>
          <div className="review-card-body">
            <h3>{entry.source_case}</h3><p className="review-algorithm">{String(entry.recipe.pfm ?? art.name)}</p>
            <div className="review-votes" role="group" aria-label={`Preference for ${entry.id}`}>
              {([['like', 'Like'], ['dislike', 'Dislike'], ['unrated', 'Clear']] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={preference === value} onClick={() => vote(entry, value)}>{label}</button>)}
            </div>
            <dl><dt>Canvas</dt><dd>{size.map(mm).join(" × ")} mm</dd>
              <dt>Generation width</dt><dd>{mm(artworkWidthM(art))} mm</dd>
              <dt>Tool width</dt><dd>{prep.tool.line_width_m === null ? "Unknown" : `${mm(prep.tool.line_width_m)} mm · ${prep.tool.line_width_status}`}</dd>
              <dt>Paths / chunks</dt><dd>{prep.stats.paths} / {prep.stats.strokes}</dd>
              <dt>Contact length</dt><dd>{mm(prep.stats.contact_m)} mm</dd>
              <dt>Speed</dt><dd>{mm(prep.speed_m_s)} mm/s</dd>
              <dt>Modeled time</dt><dd>{duration(prep.stats.time_estimate.modeled_s)}</dd>
            </dl>
            <details><summary>Recipe and estimate details</summary>
              <p>{prep.stats.time_estimate.scope}</p>
              {Object.keys(prep.stats.time_estimate.unknown_operations).length > 0 && <p>Unknown costs: {Object.keys(prep.stats.time_estimate.unknown_operations).join(", ")}</p>}
              {prep.stats.notes.map((note, i) => <p key={i}>{note}</p>)}
              <pre>{JSON.stringify(entry.recipe, null, 2)}</pre><p>Tool: {prep.tool.id}</p><p className="mono">Artwork {entry.artwork_sha256.slice(0, 12)} · Program {prep.program_sha256.slice(0, 12)}</p>
            </details>
            <button type="button" className="review-add" disabled={!ready} onClick={() => void add(entry)}>Add to library</button>
          </div>
        </article>;
      })}</div>
    </>}
  </dialog>;
}
