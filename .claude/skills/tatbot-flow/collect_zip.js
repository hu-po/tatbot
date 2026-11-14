// Run in the Flow project tab (javascript_tool). Scrolls the virtualized grid, reads every batch's
// prompt text and image URLs, fetches the originals with the page's session, and downloads ONE zip:
// batches.json plus bNN-K.jpg files. Returns a summary. Selectors were current on 2026-11-06.
const T = new Uint32Array(256);
for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; T[n] = c >>> 0; }
const crc = u => { let c = 0xFFFFFFFF; for (let i = 0; i < u.length; i++) c = T[(c ^ u[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
const seen = {};  // image path -> its batch's prompt; each image counted once however often it is seen
let order = 0;
const grab = () => {
  for (const b of document.querySelectorAll('div.batch-container')) {
    const lines = (b.innerText || '').split('\n');
    const prompt = lines.filter(s => s.length > 60).join(' ');
    const created = lines.find(s => /^Created/.test(s)) || '';
    for (const i of b.querySelectorAll('img')) {
      if (i.naturalWidth <= 200) continue;
      const id = new URL(i.src).pathname;  // the signature in the query changes between reads
      if (!(id in seen)) seen[id] = { src: i.src, prompt, created, order: order++ };
    }
  }
};
const sc = [...document.querySelectorAll('*')].find(e => e.scrollHeight > e.clientHeight + 500 && /auto|scroll/.test(getComputedStyle(e).overflowY));
sc.scrollTop = 0; await new Promise(r => setTimeout(r, 800)); grab();
for (let k = 0; k < 400 && sc.scrollTop + sc.clientHeight < sc.scrollHeight - 2; k++) {
  sc.scrollTop += 350; await new Promise(r => setTimeout(r, 450)); grab();
}
grab();
const enc = new TextEncoder(); const files = []; const meta = [];
const groups = new Map();  // one entry per prompt, in the grid's order (newest first)
for (const v of Object.values(seen).sort((a, b) => a.order - b.order)) {
  const key = v.prompt + '\u0000' + v.created;
  if (!groups.has(key)) groups.set(key, { prompt: v.prompt, created: v.created, imgs: [] });
  groups.get(key).imgs.push(v.src);
}
const list = [...groups.values()];
for (let i = 0; i < list.length; i++) {
  const names = [];
  for (let k = 0; k < list[i].imgs.length; k++) {
    const r = await fetch(list[i].imgs[k], { credentials: 'omit' });  // signed: credentials fail its CORS check
    if (!r.ok) throw new Error(`image ${i}-${k}: ${r.status}`);
    const name = `b${String(i).padStart(2, '0')}-${k}.jpg`;
    files.push([name, new Uint8Array(await r.arrayBuffer())]); names.push({ file: name });
  }
  meta.push({ batch: i, prompt: list[i].prompt, created: list[i].created, images: names });
}
files.push(['batches.json', enc.encode(JSON.stringify({ project: location.href, fetched_utc: new Date().toISOString(), batches: meta }, null, 1))]);
const parts = [], cen = []; let off = 0;
for (const [name, data] of files) {  // a stored (uncompressed) zip: JPEGs gain nothing from deflate
  const nb = enc.encode(name), c = crc(data);
  const h = new DataView(new ArrayBuffer(30)); h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true);
  h.setUint32(14, c, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, nb.length, true);
  parts.push(new Uint8Array(h.buffer), nb, data);
  const ch = new DataView(new ArrayBuffer(46)); ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true);
  ch.setUint32(16, c, true); ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true); ch.setUint16(28, nb.length, true); ch.setUint32(42, off, true);
  cen.push(new Uint8Array(ch.buffer), nb); off += 30 + nb.length + data.length;
}
const cenSize = cen.reduce((a, p) => a + p.length, 0);
const e = new DataView(new ArrayBuffer(22)); e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true);
e.setUint16(10, files.length, true); e.setUint32(12, cenSize, true); e.setUint32(16, off, true);
const blob = new Blob([...parts, ...cen, new Uint8Array(e.buffer)], { type: 'application/zip' });
const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
a.download = `flow-${location.pathname.split('/').pop()}-${Date.now()}.zip`; document.body.appendChild(a); a.click(); a.remove();
JSON.stringify({ batches: list.length, images: files.length - 1, bytes: blob.size, file: a.download });
