import { useEffect, useId, useRef, type ReactNode } from "react";
import { useUi } from "../ui.ts";

/** One popover at a time, named. Escape and an outside click close it and
 *  hand focus back to the button that opened it, so the keyboard never lands
 *  in the dark. Escape here wins over the editor's Escape (cancel edit): the
 *  App key handler sees `popover` and leaves the edit alone. */
export function Menu({ name, label, children, className = "", align = "start" }:
  { name: string; label: ReactNode; children: ReactNode; className?: string; align?: "start" | "end" }) {
  const open = useUi((s) => s.popover === name);
  const setPopover = useUi((s) => s.setPopover);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const id = useId();
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open) {
      wasOpen.current = true;
      const onDown = (event: PointerEvent) => {
        const target = event.target as Node;
        if (!panel.current?.contains(target) && !trigger.current?.contains(target)) setPopover(null);
      };
      document.addEventListener("pointerdown", onDown);
      return () => document.removeEventListener("pointerdown", onDown);
    }
    if (wasOpen.current) { wasOpen.current = false; trigger.current?.focus(); }
  }, [open, setPopover]);
  return (
    <div className={`menu ${className}`}>
      <button ref={trigger} type="button" className={open ? "menu-trigger open" : "menu-trigger"} aria-haspopup="true" aria-expanded={open} aria-controls={id}
        onClick={() => setPopover(open ? null : name)}>{label}</button>
      {open && <div ref={panel} id={id} role="group" aria-label={typeof label === "string" ? label : name} className={`menu-panel ${align}`}
        onClick={(event) => { const t = event.target as HTMLElement; if (t.closest("[data-closes]")) setPopover(null); }}>{children}</div>}
    </div>
  );
}
