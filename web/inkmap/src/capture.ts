/** The rendered view as an image. The canvas draws on demand, so its drawing buffer holds nothing between frames: a
 *  capture renders and reads in one step, which only the Scene can do. It registers itself here. */
import type { Placement } from "./core/schema.ts";

type Capture = (placement?: Placement) => string;
let capture: Capture | null = null;

export const registerCapture = (fn: Capture | null): void => { capture = fn; };

/** The current view, or a close view of `placement`, as a JPEG data URL. */
export function captureView(placement?: Placement): string {
  if (!capture) throw new Error("Wait for the body before capturing a view.");
  return capture(placement);
}
