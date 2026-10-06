/** Browser session orchestration; no hidden writes from store constructors. */
import { loadProject, saveProject } from "./core/project-db.ts";
import { useStore, type State } from "./store.ts";

const fields = (s: State) => [s.placements, s.past, s.future, s.editBefore, s.selected, s.poseId, s.skinTone, s.showAtlas, s.cameraSnapshot, s.projectName, s.designs, s.chart, s.chartParked];

export function startProjectSession(): () => void {
  let stopped = false, initializing = false, initialized = false, saving = false, dirty = false, failed = false;
  let revision = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let previous: unknown[] = [];
  const fail = (error: unknown) => {
    if (stopped) return;
    failed = true;
    // Once recovery fails, the user may continue authoring. A retry must not
    // reload over those edits. Revision zero can only create an absent record;
    // an existing/unknown stored record will cause a compare-and-swap refusal.
    if (!initialized) { initialized = true; previous = fields(useStore.getState()); }
    useStore.setState({ saveStatus: "failed", saveError: error instanceof Error ? error.message : String(error), projectReady: true });
  };
  const persist = async () => {
    if (stopped || failed || !initialized || saving || !dirty || !useStore.getState().body) return;
    saving = true; dirty = false;
    useStore.setState({ saveStatus: "saving" });
    try {
      const project = await useStore.getState().toProject();
      if (stopped) return;
      revision = await saveProject(project, revision);
      if (!stopped) useStore.setState({ saveStatus: dirty ? "saving" : "saved", saveError: null });
    } catch (error) { fail(error); }
    finally { saving = false; if (dirty && !failed && !stopped) timer = setTimeout(persist, 500); }
  };
  const changed = (state: State) => {
    if (stopped) return;
    if (!initialized) {
      if (!initializing && state.body && state.atlas) {
        initializing = true;
        void loadProject().then(saved => {
          if (stopped) return;
          if (saved) { revision = saved.revision; useStore.getState().restoreProject(saved.project); }
          initialized = true; previous = fields(useStore.getState()); dirty = !saved;
          useStore.setState({ projectReady: true, saveStatus: saved ? "saved" : "saving" });
          if (dirty) timer = setTimeout(persist, 500);
        }).catch(fail);
      }
      return;
    }
    const current = fields(state);
    if (current.some((value, i) => value !== previous[i])) {
      previous = current; dirty = true;
      if (!failed) useStore.setState({ saveStatus: "saving" });
      clearTimeout(timer); timer = setTimeout(persist, 500);
    } else if (dirty && state.body && !saving && !failed) {
      clearTimeout(timer); timer = setTimeout(persist, 500);
    }
  };
  const unsubscribe = useStore.subscribe(changed);
  changed(useStore.getState());
  const warn = (event: BeforeUnloadEvent) => {
    if (dirty || saving || failed) { event.preventDefault(); event.returnValue = ""; }
  };
  const flush = () => { if (document.visibilityState === "hidden") void persist(); };
  const retry = () => {
    failed = false;
    dirty = true; void persist();
  };
  window.addEventListener("beforeunload", warn);
  window.addEventListener("inkmap-save-retry", retry);
  document.addEventListener("visibilitychange", flush);
  return () => { stopped = true; clearTimeout(timer); unsubscribe(); window.removeEventListener("beforeunload", warn); window.removeEventListener("inkmap-save-retry", retry); document.removeEventListener("visibilitychange", flush); };
}
