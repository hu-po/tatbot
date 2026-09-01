import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the bundle serves from "/" on the Space and from any subpath elsewhere.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    // The tracer wasm (160 KB) is fetched from inside a Web Worker; on a private Space
    // inside the Hub's iframe that fetch does not carry the viewer's credentials, so
    // embed it in the bundle as a data: URL instead of shipping it as a file.
    assetsInlineLimit: (file) => (file.endsWith(".wasm") ? true : undefined),
  },
  server: { host: "127.0.0.1", port: 4180 },
});
