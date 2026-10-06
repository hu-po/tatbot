import { DOMParser } from "@xmldom/xmldom";

/** SVGLoader's only selector for admitted input is its gradient discovery.
 * Actual gradients are refused by the materializer before this parser runs.
 */
export class SvgDOMParser extends DOMParser {
  override parseFromString(...args: Parameters<DOMParser["parseFromString"]>) {
    const document = super.parseFromString(...args);
    Object.assign(document, { querySelectorAll: (selector: string) => {
      if (selector !== "linearGradient, radialGradient") throw new Error(`unsupported SVG selector ${selector}`);
      return [...Array.from(document.getElementsByTagName("linearGradient")), ...Array.from(document.getElementsByTagName("radialGradient"))];
    } });
    return document;
  }
}
