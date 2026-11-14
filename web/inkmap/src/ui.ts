/** Presentation state only: whether the tray is open, which popover is up.
 *  Nothing here is placement truth — that stays in `store.ts`, and the
 *  editor's contextual mode is *derived* from it. */
import { create } from "zustand";
import { DEFAULT_TILE_ID, STENCIL_TILES } from "./core/tiles.ts";

export type Tray = "collapsed" | "working" | "expanded";
/** What the person is doing on the body, read off the placement store. */
export type BodyMode = "choose" | "place" | "adjust" | "ready";

function loadTile(): string {
  try {
    const id = localStorage.getItem("inkmap.tile");
    if (id && STENCIL_TILES.some((tile) => tile.id === id)) return id;
  } catch { /* no storage */ }
  return DEFAULT_TILE_ID;
}

export interface UiState {
  /** "Add artwork" from the ready state opens the chooser without touching the store. */
  choosing: boolean;
  browsingAll: boolean;
  tray: Tray;
  /** The popover currently open, by name; at most one. */
  popover: string | null;
  /** A notice beside an action: exports, opens, saves. Cleared by the next action. Advice refused nothing. */
  notice: { action: string; message: string; advice?: boolean } | null;
  /** The stencil tile a design is judged against and outlined with. */
  tile: string;
  setChoosing: (choosing: boolean) => void;
  setBrowsingAll: (browsing: boolean) => void;
  setTray: (tray: Tray) => void;
  setPopover: (popover: string | null) => void;
  setNotice: (notice: UiState["notice"]) => void;
  setTile: (tile: string) => void;
}

export const useUi = create<UiState>((set) => ({
  choosing: false,
  browsingAll: false,
  tray: "working",
  popover: null,
  notice: null,
  tile: loadTile(),
  setChoosing: (choosing) => set({ choosing, browsingAll: false }),
  setBrowsingAll: (browsingAll) => set({ browsingAll }),
  setTray: (tray) => set({ tray }),
  setPopover: (popover) => set({ popover }),
  setNotice: (notice) => set({ notice }),
  setTile: (tile) => {
    try { localStorage.setItem("inkmap.tile", tile); } catch { /* private mode etc. */ }
    set({ tile });
  },
}));

/** The body editor's mode is a reading of the store, never a second record of it. */
export function bodyMode(placing: string | null, selected: string | null, placements: number, choosing: boolean): BodyMode {
  if (placing) return "place";
  if (selected) return "adjust";
  if (choosing || placements === 0) return "choose";
  return "ready";
}
