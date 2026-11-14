import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the bundle serves from "/" on the Space and from any subpath elsewhere.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
  },
  server: { host: "127.0.0.1", port: 4180 },
});
