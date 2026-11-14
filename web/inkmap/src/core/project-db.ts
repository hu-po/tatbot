import { PROJECT_SCHEMA, validateProject, type InkmapProject } from "./project.ts";
import { validateRender, type InkmapRender } from "./render-record.ts";

/** `project` is null when the store holds another version: there is no migration, and the next save replaces it. */
export interface SavedProject { revision: number; project: InkmapProject | null }
function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("inkmap-projects", 1);
    let refused = false;
    const fail = (error: unknown) => { refused = true; clearTimeout(timer); reject(error); };
    const timer = setTimeout(() => fail(new Error("project_storage_timeout: database did not open")), 10_000);
    request.onupgradeneeded = () => request.result.createObjectStore("projects");
    request.onerror = () => fail(request.error);
    request.onblocked = () => fail(new Error("project_storage_blocked: close older Inkmap tabs"));
    request.onsuccess = () => { clearTimeout(timer); if (refused) { request.result.close(); return; } request.result.onversionchange = () => request.result.close(); resolve(request.result); };
  });
}

export async function loadProject(): Promise<SavedProject | null> {
  const db = await open();
  try {
    const saved = await new Promise<SavedProject | null>((resolve, reject) => {
      const transaction = db.transaction("projects", "readonly");
      const request = transaction.objectStore("projects").get("current");
      transaction.oncomplete = () => resolve(request.result ?? null);
      transaction.onabort = () => reject(transaction.error ?? new Error("Project read aborted"));
    });
    if (!saved) return null;
    if (!Number.isSafeInteger(saved.revision) || saved.revision < 1) throw new Error("project_invalid: storage revision");
    if (saved.project?.schema !== PROJECT_SCHEMA) return { revision: saved.revision, project: null };
    return { revision: saved.revision, project: await validateProject(saved.project) };
  } finally { db.close(); }
}

/** Compare-and-swap inside one read/write transaction prevents lost tab edits.
 * Retain the immediately previous version for recovery.
 */
export async function saveProject(project: InkmapProject, expectedRevision: number): Promise<number> {
  if (!Number.isSafeInteger(expectedRevision) || expectedRevision < 0 || expectedRevision >= Number.MAX_SAFE_INTEGER) throw new Error("project_invalid: storage revision outside range");
  const checked = await validateProject(project);
  const db = await open();
  try {
    return await new Promise<number>((resolve, reject) => {
      let conflict = false;
      let failure: string | null = null;
      const transaction = db.transaction("projects", "readwrite");
      const store = transaction.objectStore("projects");
      const request = store.get("current");
      request.onsuccess = () => {
        const prior = request.result as SavedProject | undefined;
        if ((prior && (!Number.isSafeInteger(prior.revision) || prior.revision < 1)) || (prior?.revision ?? 0) !== expectedRevision) { conflict = true; transaction.abort(); return; }
        try {
          if (prior) store.put(prior, "previous");
          store.put({ revision: expectedRevision + 1, project: checked }, "current");
        } catch (error) { failure = error instanceof Error ? error.message : String(error); transaction.abort(); }
      };
      transaction.oncomplete = () => resolve(expectedRevision + 1);
      // Request errors bubble before transaction.error is finalized. Report
      // only the terminal abort, retaining our located synchronous refusal.
      transaction.onabort = () => reject(new Error(conflict ? "project_conflict: another tab saved newer work; download this copy before reloading" : `project_save_failed: ${failure ?? transaction.error?.message ?? "transaction aborted"}`));
    });
  } finally { db.close(); }
}

/** Renders beside the project, not in it: they are made from it, so a backup leaves them to be loaded again. */
export async function loadRenders(): Promise<InkmapRender[]> {
  const db = await open();
  try {
    const saved = await new Promise<unknown>((resolve, reject) => {
      const transaction = db.transaction("projects", "readonly");
      const request = transaction.objectStore("projects").get("renders");
      transaction.oncomplete = () => resolve(request.result ?? []);
      transaction.onabort = () => reject(transaction.error ?? new Error("Render read aborted"));
    });
    return await Promise.all((saved as unknown[]).map(validateRender));
  } finally { db.close(); }
}

export async function saveRenders(renders: InkmapRender[]): Promise<void> {
  const db = await open();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("projects", "readwrite");
      transaction.objectStore("projects").put(renders, "renders");
      transaction.oncomplete = () => resolve();
      transaction.onabort = () => reject(transaction.error ?? new Error("Render save aborted"));
    });
  } finally { db.close(); }
}
