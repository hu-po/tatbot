import React from "react";
import ReactDOM from "react-dom/client";
import Viewer from "./components/Viewer";

const params = new URLSearchParams(window.location.search);
const src = params.get("file") ?? "scene.glb";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Viewer src={src} />
  </React.StrictMode>
);