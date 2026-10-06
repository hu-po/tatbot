import { spawn } from "node:child_process";
import { join } from "node:path";
import { stripVTControlCharacters } from "node:util";

const host = "127.0.0.1";
const production = process.argv.includes("--production");
const port = production ? 4183 : 4181;
const url = `http://${host}:${port}/`;
const server = spawn(process.execPath, ["node_modules/vite/bin/vite.js", ...(production ? ["preview"] : []), "--host", host, "--port", String(port), "--strictPort"], {
  stdio: ["ignore", "pipe", "pipe"],
});
let serverLog = "";
server.stdout.on("data", (chunk) => { serverLog += chunk; });
server.stderr.on("data", (chunk) => { serverLog += chunk; });

async function waitForServer() {
  for (let attempt = 0; attempt < 120; attempt++) {
    if (server.exitCode !== null) throw new Error(`Vite exited ${server.exitCode}\n${serverLog}`);
    try {
      const response = await fetch(url);
      // HTTP alone may belong to somebody else's process on an occupied port.
      if (response.ok && stripVTControlCharacters(serverLog).includes(url)) return;
    } catch { /* starting */ }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Vite did not start\n${serverLog}`);
}

try {
  await waitForServer();
  for (const script of ["sentence", "project", "editor", "bundle", "chart", "generation", "layout", "placement-fixes", "study-review"]) {
    const env = { ...process.env };
    if (env.INKMAP_E2E_EVIDENCE) env.INKMAP_E2E_EVIDENCE = join(env.INKMAP_E2E_EVIDENCE, production ? "production" : "development");
    const test = spawn(process.execPath, ["--experimental-strip-types", `tests/e2e/${script}.mjs`, url], { stdio: "inherit", env });
    const code = await new Promise((resolve) => test.on("exit", resolve));
    if (code !== 0) throw new Error(`${script} browser test exited ${code}`);
  }
} finally {
  server.kill("SIGTERM");
  await new Promise((resolve) => {
    if (server.exitCode !== null) { resolve(); return; }
    server.once("exit", resolve);
    setTimeout(() => resolve(), 2_000);
  });
}
