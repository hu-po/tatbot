import { Canvas } from "@react-three/fiber";
import { OrbitControls, TransformControls } from "@react-three/drei";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { useLoader, useThree } from "@react-three/fiber";
import { XR, XRButton } from "@react-three/xr";
import { useMemo, useRef, useState } from "react";
import * as THREE from "three";

interface Props { src: string; }
export default function Viewer({ src }: Props) {
  const gltf = useLoader(GLTFLoader, src);
  const [mode, setMode] = useState<"translate" | "rotate" | "scale">("translate");
  const sceneRef = useRef<THREE.Object3D>(gltf.scene);
  const { camera } = useThree();

  // collect xforms to send back
  const collect = () => {
    const out: Record<string, number[]> = {};
    sceneRef.current.traverse(obj => {
      if (obj instanceof THREE.Mesh) {
        out[obj.name || obj.uuid] = obj.matrix.elements;
      }
    });
    return out;
  };

  const handleSave = async () => {
    const xforms = collect();
    await fetch("/save-xforms", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(xforms)
    });
    alert("Transforms saved back to USD ✔");
  };

  return (
    <>
      <Canvas camera={{ position: [1.5, 1.5, 1.5], fov: 50 }}>
        <XR>
          <ambientLight />
          <primitive ref={sceneRef as any} object={gltf.scene} />
          <TransformControls mode={mode} />
          <OrbitControls makeDefault />
        </XR>
      </Canvas>

      <div className="absolute left-4 top-4 flex flex-col gap-2 text-white">
        {(["translate", "rotate", "scale"] as const).map(m => (
          <button key={m} onClick={() => setMode(m)} className="rounded-xl bg-slate-800 px-3 py-1 text-sm">
            {m}
          </button>
        ))}
        <button onClick={handleSave} className="rounded-xl bg-green-700 px-3 py-1 text-sm">Save</button>
        <XRButton className="rounded-xl bg-sky-700 px-3 py-1 text-sm" />
      </div>
    </>
  );
}