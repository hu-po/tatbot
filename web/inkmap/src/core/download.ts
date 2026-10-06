/** Keep the blob alive while the browser starts its asynchronous download. */
export function downloadJson(value: unknown, filename: string): void {
  downloadText(JSON.stringify(value), filename, "application/json");
}

/** The same deferred release for any text artifact: SVG, JSON, a recovery copy. */
export function downloadText(text: string, filename: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  try { link.click(); }
  finally {
    link.remove();
    // click() returning does not mean the browser has opened the blob. This
    // matches the project recovery download's deferred release and avoids
    // canceling larger design/bundle downloads at their initiation boundary.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
