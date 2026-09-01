import { useStore } from "../store.ts";

/** `pulse` plays the attention animation once on mount (App remounts the picker per accepted tattoo). */
export function Picker({ pulse = false }: { pulse?: boolean } = {}) {
  const designs = useStore((s) => s.designs);
  const placing = useStore((s) => s.placing);
  const startPlacing = useStore((s) => s.startPlacing);
  const cancelPlacing = useStore((s) => s.cancelPlacing);
  return (
    <section>
      <h2>Designs</h2>
      <div className={pulse ? "picker pulse" : "picker"}>
        {designs.map((d) => (
          <button
            key={d.id}
            type="button"
            className={placing === d.id ? "design active" : "design"}
            onClick={() => (placing === d.id ? cancelPlacing() : startPlacing(d.id))}
            title={`${d.name} — ${d.default_size_mm[0]}×${d.default_size_mm[1]} mm`}
          >
            <img src={d.path} alt="" />
            <span>{d.name}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
