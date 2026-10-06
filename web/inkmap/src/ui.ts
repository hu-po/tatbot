/** Presentation state only: which workspace is showing, whether the tray is
 *  open, which popover is up. Nothing here is placement truth — that stays in
 *  `store.ts`, and the editor's contextual mode is *derived* from it. */
import { create } from "zustand";
import { useStore } from "./store.ts";

/** The body, or the one analytic chart (paper pad or cylinder — the chart's
 *  own `kind` says which, so the tab and the draft can never disagree). */
export type Workspace = "body" | "chart";
export type WorkspaceTab = "body" | "paper" | "cylinder";
export type Tray = "collapsed" | "working" | "expanded";
/** What the person is doing on the body, read off the placement store. */
export type BodyMode = "choose" | "place" | "adjust" | "ready";

export interface UiState {
  workspace: Workspace;
  /** "Add artwork" from the ready state opens the chooser without touching the store. */
  choosing: boolean;
  /** Which chooser tab is up; native acquisition instructions live in Generate. */
  chooserTab: "library" | "generate";
  browsingAll: boolean;
  tray: Tray;
  /** The popover currently open, by name; at most one. */
  popover: string | null;
  /** A notice beside an action: exports, opens, saves. Cleared by the next action. */
  notice: { action: string; message: string } | null;
  /** Simulation export settings; the panel that edits them lives in a menu. */
  simTool: string;
  simSeed: number;
  setWorkspace: (tab: WorkspaceTab) => void;
  setChoosing: (choosing: boolean) => void;
  setChooserTab: (tab: "library" | "generate") => void;
  setBrowsingAll: (browsing: boolean) => void;
  setTray: (tray: Tray) => void;
  setPopover: (popover: string | null) => void;
  setNotice: (notice: { action: string; message: string } | null) => void;
}

export function workspaceTab(workspace: Workspace, chartKind: "plane" | "cylinder"): WorkspaceTab {
  return workspace === "body" ? "body" : chartKind === "cylinder" ? "cylinder" : "paper";
}

export const useUi = create<UiState>((set) => ({
  // Always the body first: the project only finishes loading once the body
  // and atlas are in, and the chart tabs stay disabled until then.
  workspace: "body",
  choosing: false,
  chooserTab: "library",
  browsingAll: false,
  tray: "working",
  popover: null,
  notice: null,
  simTool: "lutin-3rl-bugpin",
  simSeed: 0,
  setWorkspace: (tab) => {
    const workspace: Workspace = tab === "body" ? "body" : "chart";
    // A paper or cylinder tab names the chart's kind and nothing else: the
    // chart draft keeps its items and dimensions, the body draft is never
    // read as chart coordinates, and neither is erased by the switch.
    // Paper and cylinder are separate drafts in the project: the tab shows
    // one and parks the other, and nothing crosses between them.
    if (tab !== "body") useStore.getState().switchChartKind(tab === "paper" ? "plane" : "cylinder");
    set({ workspace, popover: null, notice: null, choosing: false, browsingAll: false });
  },
  setChoosing: (choosing) => set({ choosing, browsingAll: false }),
  setChooserTab: (chooserTab) => set({ chooserTab }),
  setBrowsingAll: (browsingAll) => set({ browsingAll }),
  setTray: (tray) => set({ tray }),
  setPopover: (popover) => set({ popover }),
  setNotice: (notice) => set({ notice }),
}));

/** The body editor's mode is a reading of the store, never a second record of it. */
export function bodyMode(placing: string | null, selected: string | null, placements: number, choosing: boolean): BodyMode {
  if (placing) return "place";
  if (selected) return "adjust";
  if (choosing || placements === 0) return "choose";
  return "ready";
}
