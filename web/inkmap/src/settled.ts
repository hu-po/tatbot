import { useEffect, useState } from "react";

/** `value` once it has held still for `delayMs`. A reading that rebuilds a chart waits for the input to settle, so a
 *  drag or a held key pays for the preview alone. */
export function useSettled<T>(value: T, delayMs = 150): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setSettled(value), delayMs);
    return () => window.clearTimeout(id);
  }, [value, delayMs]);
  return settled;
}
