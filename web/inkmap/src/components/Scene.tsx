import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as cloneSkeleton } from "three/examples/jsm/utils/SkeletonUtils.js";
import { useStore, type LoadedBody } from "../store.ts";
import { bodySpec, buildPosedSkin, buildSkin, canonicalSurfaceBytes, sha256Hex, type BodySpec } from "../core/body.ts";
import { POSE_CATALOG, poseRecord } from "../core/pose.ts";
import { anchorToPoint, pointToAnchor } from "../core/anchor.ts";
import { buildDecal } from "../core/decal.ts";
import { svgTexture } from "../core/svg.ts";
import type { Placement } from "../core/schema.ts";
import { AtlasIndex, parseAtlas, type RegionRef } from "../core/atlas.ts";

export function Scene() {
  const spec = bodySpec(useStore((s) => s.bodyId));
  const poseId = useStore((s) => s.poseId);
  const eye = spec.eyeHeight;
  return (
    <Canvas
      camera={{ position: [1.6, -2.4, eye * 0.85], up: [0, 0, 1], fov: 38, near: 0.01, far: 50 }}
      onCreated={({ camera }) => camera.up.set(0, 0, 1)}
      dpr={[1, 2]}
    >
      <color attach="background" args={["#1b1d22"]} />
      <hemisphereLight args={["#ffffff", "#3a3f4a", 0.9]} />
      <directionalLight position={[3, -4, 6]} intensity={1.4} />
      <directionalLight position={[-4, 3, 2]} intensity={0.4} />
      <gridHelper args={[4, 16, "#3c414b", "#2a2e36"]} rotation={[Math.PI / 2, 0, 0]} />
      <CanvasErrorBoundary>
        <Suspense fallback={null}>
          <Body key={`${spec.id}:${poseId}`} spec={spec} poseId={poseId} />
        </Suspense>
        <PoseSupport />
        <AtlasOverlay />
        <RegionHighlight />
        <Placements />
        <ScenarioTrace />
        <ScenarioSiteMarker />
        <ShowcaseCamera />
      </CanvasErrorBoundary>
      <OrbitControls makeDefault target={[0, 0, eye * 0.6]} minDistance={0.3} maxDistance={8} />
    </Canvas>
  );
}

interface SupportBox {
  center: [number, number, number];
  size: [number, number, number];
  rotation?: [number, number, number];
}

/** Kinematic support proxy shared by the editor and showcase pose contract. */
function PoseSupport() {
  const body = useStore((s) => s.body);
  const boxes = useMemo<SupportBox[]>(() => {
    if (!body) return [];
    const supportId = poseRecord(body.spec.id, body.poseId).support_id;
    const low = body.skin.bbox.min;
    const center = body.skin.bbox.getCenter(new THREE.Vector3());
    const span = body.skin.bbox.getSize(new THREE.Vector3());
    if (supportId === "standing-reference-v1") return [];
    if (supportId === "tattoo-bed-v1") {
      return [{
        center: [center.x, center.y, low.z - 0.04],
        size: [Math.max(0.9, span.x + 0.16), Math.max(1.8, span.y + 0.16), 0.08],
      }];
    }
    if (supportId.startsWith("tattoo-chair-")) {
      const seatZ = low.z + span.z * 0.18;
      const armrestZ = low.z + span.z * (body.spec.id === "hbm-female-stylized" ? 0.365 : 0.34);
      const result: SupportBox[] = [
        { center: [0, center.y, seatZ - 0.035], size: [0.56, 0.50, 0.07] },
        {
          center: [0, center.y + 0.40, seatZ + 0.32],
          size: [0.52, 0.06, 0.72],
          rotation: [-Math.PI / 6, 0, 0],
        },
        { center: [0, -0.04, low.z + 0.045], size: [0.50, 0.54, 0.07] },
      ];
      if (supportId.endsWith("left-armrest-v1")) {
        result.push({
          center: [0.45, 0.40, armrestZ],
          size: [0.32, 0.38, 0.05],
        });
      }
      if (supportId.endsWith("right-armrest-v1")) {
        result.push({
          center: [-0.45, 0.40, armrestZ],
          size: [0.32, 0.38, 0.05],
        });
      }
      return result;
    }
    return [];
  }, [body]);
  return (
    <group>
      {boxes.map((box, index) => (
        <mesh key={index} position={box.center} rotation={box.rotation} raycast={() => null}>
          <boxGeometry args={box.size} />
          <meshStandardMaterial color="#343a46" roughness={0.82} metalness={0.05} />
        </mesh>
      ))}
    </group>
  );
}

function ShowcaseCamera() {
  const body = useStore((s) => s.body);
  const scenario = useStore((s) => s.showcaseScenario);
  const focus = useStore((s) => s.showcaseFocus);
  const { camera, controls } = useThree();
  useEffect(() => {
    if (!body) return;
    const orbit = controls as unknown as { target: THREE.Vector3; update: () => void } | undefined;
    camera.up.set(0, 0, 1);
    const matchingScenario = scenario && body.spec.id === scenario.body.id && body.poseId === scenario.pose.id
      ? scenario
      : null;
    if (matchingScenario && focus) {
      const { p, n } = anchorToPoint(body.skin.geometry, matchingScenario.placement.anchor);
      const largest = Math.max(...matchingScenario.placement.size_mm) / 1000;
      const distance = THREE.MathUtils.clamp(largest * 10, 0.45, 0.70);
      const studioView = new THREE.Vector3(0.3, -0.6, 1.0).normalize();
      const view = n.clone().multiplyScalar(0.25).addScaledVector(studioView, 0.75).normalize();
      camera.position.copy(p).addScaledVector(view, distance).add(new THREE.Vector3(0, 0, largest * 0.35));
      orbit?.target.copy(p);
    } else {
      const center = body.skin.bbox.getCenter(new THREE.Vector3());
      const size = body.skin.bbox.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z);
      camera.position.copy(center).add(new THREE.Vector3(radius * 0.95, -radius * 1.55, radius * 0.55));
      orbit?.target.copy(center);
    }
    camera.lookAt(orbit?.target ?? body.skin.bbox.getCenter(new THREE.Vector3()));
    camera.updateProjectionMatrix();
    orbit?.update();
  }, [body, scenario, focus, camera, controls]);
  return null;
}

function ScenarioSiteMarker() {
  const body = useStore((s) => s.body);
  const scenario = useStore((s) => s.showcaseScenario);
  const focus = useStore((s) => s.showcaseFocus);
  const marker = useMemo(() => {
    if (!body || !scenario || body.spec.id !== scenario.body.id || body.poseId !== scenario.pose.id) return null;
    const { p, n } = anchorToPoint(body.skin.geometry, scenario.placement.anchor);
    const scale = THREE.MathUtils.clamp(Math.max(...scenario.placement.size_mm) / 1000 * 1.35, 0.042, 0.075);
    return {
      position: p.addScaledVector(n, 0.0025),
      quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n),
      scale,
    };
  }, [body, scenario]);
  if (!marker || focus) return null;
  return (
    <mesh position={marker.position} quaternion={marker.quaternion} raycast={() => null}>
      <ringGeometry args={[marker.scale * 0.58, marker.scale * 0.7, 40]} />
      <meshBasicMaterial color="#61e8ff" transparent opacity={0.9} depthTest={false} />
    </mesh>
  );
}

/** Exact compiled stroke anchors replayed on the currently posed skin. */
function ScenarioTrace() {
  const body = useStore((s) => s.body);
  const scenario = useStore((s) => s.showcaseScenario);
  const visible = useStore((s) => s.showcaseTraceVisible);
  const lines = useMemo(() => {
    if (!body || !scenario || body.spec.id !== scenario.body.id || body.poseId !== scenario.pose.id) return [];
    return scenario.trace.strokes.map((stroke) => {
      const points = stroke.map((anchor) => {
        const { p, n } = anchorToPoint(body.skin.geometry, anchor);
        return p.addScaledVector(n, 0.0015);
      });
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: "#61e8ff", transparent: true, opacity: 0.96 });
      const line = new THREE.Line(geometry, material);
      line.raycast = () => undefined;
      return line;
    });
  }, [body, scenario]);
  useEffect(() => () => lines.forEach((line) => {
    line.geometry.dispose();
    (line.material as THREE.Material).dispose();
  }), [lines]);
  if (!visible) return null;
  return <group>{lines.map((line, index) => <primitive key={index} object={line} />)}</group>;
}

/** The Canvas is its own React root: an uncaught error there blanks the scene silently. Surface it in the sidebar instead. */
class CanvasErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(err: Error) {
    console.error("[inkmap] scene error", err);
    useStore.getState().setError(`scene: ${err.message}`);
  }
  render() { return this.state.failed ? null : this.props.children; }
}

function Body({ spec, poseId }: { spec: BodySpec; poseId: string }) {
  const gltf = useLoader(GLTFLoader, spec.path);
  const riggedGltf = useLoader(GLTFLoader, spec.rigPath);
  const setBody = useStore((s) => s.setBody);
  const setError = useStore((s) => s.setError);
  const body = useStore((s) => s.body);
  const placing = useStore((s) => s.placing);
  const setHover = useStore((s) => s.setHover);
  const commit = useStore((s) => s.commit);
  const select = useStore((s) => s.select);
  const skinTone = useStore((s) => s.skinTone);
  // Own click detection: R3F's onClick is dropped when the pointer moves more
  // than 2 px between press and release, which a real mouse does all the time
  // while the ghost decal is rebuilding underneath it.
  const press = useRef<{ x: number; y: number; t: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const t0 = performance.now();
        const restSkin = buildSkin(gltf.scene, spec);
        const surfaceSha256 = await sha256Hex(canonicalSurfaceBytes(restSkin.geometry));
        const catalogBody = POSE_CATALOG.bodies[spec.id];
        if (!catalogBody) throw new Error(`body ${spec.id}: missing from pose catalog`);
        if (catalogBody.rigged_path !== spec.rigPath) throw new Error(`body ${spec.id}: rig path differs from pose catalog`);
        if (catalogBody.surface_sha256 !== surfaceSha256) throw new Error(`body ${spec.id}: rig was built for surface ${catalogBody.surface_sha256.slice(0, 12)}…, loaded ${surfaceSha256.slice(0, 12)}…`);
        const rigScene = cloneSkeleton(riggedGltf.scene);
        const pose = poseRecord(spec.id, poseId);
        const skin = buildPosedSkin(rigScene, spec, pose.joint_rotations, pose.body_rotation_xyzw);
        skin.map = restSkin.map;
        skin.vertexColors = restSkin.vertexColors;
        const t1 = performance.now();
        const bytes = await (await fetch(spec.path)).arrayBuffer();
        const t2 = performance.now();
        const assetSha256 = await sha256Hex(bytes);
        const t3 = performance.now();
        console.info(`[inkmap] body ${spec.id} timings: skin ${(t1 - t0).toFixed(0)} ms, fetch ${(t2 - t1).toFixed(0)} ms, sha256 ${(t3 - t2).toFixed(0)} ms (since page start ${t0.toFixed(0)} ms)`);
        if (cancelled) return;
        const loaded: LoadedBody = { spec, skin, assetSha256, surfaceSha256, scene: rigScene, poseId };
        setBody(loaded);
        setError(null);
        // The region atlas is optional per body: without it the app works,
        // sentences and captions just have nothing to ground in.
        try {
          const res = await fetch(`bodies/${spec.id}.regions.json`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const raw = parseAtlas(await res.json(), skin.centroids.length / 3);
          if (raw.body.sha256 !== assetSha256) throw new Error(`atlas is for another export of ${spec.id} — rerun tools/build_atlas.ts`);
          if (!cancelled) useStore.getState().setAtlas(new AtlasIndex(raw, skin.geometry, skin.centroids));
        } catch (e) {
          console.warn(`[inkmap] no region atlas for ${spec.id}:`, (e as Error).message);
          if (!cancelled) useStore.getState().setAtlas(null);
        }
        const size = new THREE.Vector3();
        skin.bbox.getSize(size);
        console.info(`[inkmap] body ${spec.id} asset=${assetSha256.slice(0, 12)}… surface=${surfaceSha256.slice(0, 12)}… faces=${skin.centroids.length / 3} height=${size.z.toFixed(3)} m`);
      } catch (e) {
        setError((e as Error).message);
      }
    })();
    return () => { cancelled = true; };
  }, [gltf, riggedGltf, spec, poseId, setBody, setError]);

  if (!body || body.spec.id !== spec.id) return null;

  const onMove = (e: ThreeEvent<PointerEvent>) => {
    if (!placing || e.faceIndex == null) return;
    e.stopPropagation();
    setHover(pointToAnchor(body.skin.geometry, e.faceIndex, e.point));
  };
  const onDown = (e: ThreeEvent<PointerEvent>) => {
    press.current = { x: e.clientX, y: e.clientY, t: performance.now() };
  };
  const onUp = (e: ThreeEvent<PointerEvent>) => {
    const p = press.current;
    press.current = null;
    if (!p || e.faceIndex == null) return;
    const moved = Math.hypot(e.clientX - p.x, e.clientY - p.y);
    if (moved > 8 || performance.now() - p.t > 800) return; // that was an orbit drag, not a click
    e.stopPropagation();
    if (placing) commit(pointToAnchor(body.skin.geometry, e.faceIndex, e.point));
    else select(null);
  };

  return (
    <mesh
      geometry={body.skin.geometry}
      onPointerMove={onMove}
      onPointerOut={() => { setHover(null); press.current = null; }}
      onPointerDown={onDown}
      onPointerUp={onUp}
    >
      <meshStandardMaterial
        map={body.skin.map ?? undefined}
        vertexColors={body.skin.vertexColors}
        color={body.skin.map ? "#ffffff" : skinTone}
        roughness={0.85}
        metalness={0}
      />
    </mesh>
  );
}

/** Stable, distinct colour per region (hashed hue; right side darker than left). */
function regionColor(ref: RegionRef): THREE.Color {
  let h = 0;
  for (const ch of ref.id) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const hue = ((h * 137.508) % 360) / 360;
  const light = ref.laterality === "right" ? 0.38 : 0.55;
  return new THREE.Color().setHSL(hue, 0.62, light);
}

/** The toggle-able atlas: every region tinted its own colour over the skin. */
function AtlasOverlay() {
  const body = useStore((s) => s.body);
  const atlas = useStore((s) => s.atlas);
  const show = useStore((s) => s.showAtlas);
  const geometry = useMemo(() => {
    if (!body || !atlas) return null;
    const src = body.skin.geometry.getAttribute("position") as THREE.BufferAttribute;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", src);
    g.setAttribute("normal", body.skin.geometry.getAttribute("normal"));
    const colors = new Float32Array(src.count * 3);
    const nFaces = src.count / 3;
    for (let f = 0; f < nFaces; f++) {
      const ref = atlas.regionOf(f);
      const c = ref ? regionColor(ref) : new THREE.Color("#15161a");
      for (let v = 0; v < 3; v++) { colors[9 * f + 3 * v] = c.r; colors[9 * f + 3 * v + 1] = c.g; colors[9 * f + 3 * v + 2] = c.b; }
    }
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return g;
  }, [body, atlas]);
  useEffect(() => () => geometry?.dispose(), [geometry]);
  if (!show || !geometry) return null;
  return (
    <mesh geometry={geometry} raycast={() => null}>
      <meshBasicMaterial vertexColors transparent opacity={0.45} depthWrite={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
    </mesh>
  );
}

/** The site a parsed sentence names glows until the tattoo lands or is cleared. */
function RegionHighlight() {
  const body = useStore((s) => s.body);
  const atlas = useStore((s) => s.atlas);
  const pending = useStore((s) => s.pending);
  const geometry = useMemo(() => {
    if (!body || !atlas || !pending) return null;
    const lat = pending.site.laterality === "left" || pending.site.laterality === "right" ? pending.site.laterality : null;
    const faces = atlas.facesOf(pending.site.id, lat);
    if (faces.length === 0) return null;
    const src = body.skin.geometry.getAttribute("position") as THREE.BufferAttribute;
    const arr = src.array as Float32Array;
    const out = new Float32Array(faces.length * 9);
    faces.forEach((f, i) => out.set(arr.subarray(9 * f, 9 * f + 9), 9 * i));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(out, 3));
    g.computeVertexNormals();
    return g;
  }, [body, atlas, pending]);
  useEffect(() => () => geometry?.dispose(), [geometry]);
  if (!geometry) return null;
  return (
    <mesh geometry={geometry} raycast={() => null}>
      <meshBasicMaterial color="#5b8cff" transparent opacity={0.5} depthWrite={false} polygonOffset polygonOffsetFactor={-3} polygonOffsetUnits={-3} />
    </mesh>
  );
}

function Placements() {
  const body = useStore((s) => s.body);
  const placements = useStore((s) => s.placements);
  const placing = useStore((s) => s.placing);
  const hover = useStore((s) => s.hover);
  const designs = useStore((s) => s.designs);
  const selected = useStore((s) => s.selected);
  const draft = useStore((s) => s.draft);
  if (!body) return null;
  const ghostDesign = placing ? designs.find((d) => d.id === placing) : undefined;
  const ghost: Placement | null = placing && hover && ghostDesign
    ? { id: "__ghost", design_id: ghostDesign.id, anchor: hover, rotation_rad: draft.rotation_rad, size_mm: draft.size_mm, mirror: false }
    : null;
  return (
    <>
      {placements.map((p) => <Decal key={p.id} placement={p} selected={p.id === selected} />)}
      {ghost && <Decal placement={ghost} ghost />}
    </>
  );
}

function Decal({ placement, ghost = false, selected = false }: { placement: Placement; ghost?: boolean; selected?: boolean }) {
  const body = useStore((s) => s.body)!;
  const designs = useStore((s) => s.designs);
  const select = useStore((s) => s.select);
  const design = designs.find((d) => d.id === placement.design_id);
  const [tex, setTex] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    let live = true;
    let owned: THREE.Texture | null = null;
    setTex(null);
    if (design) {
      svgTexture(design.path).then((texture) => {
        if (!live) {
          texture.dispose();
          return;
        }
        owned = texture;
        setTex(texture);
      }).catch(console.error);
    }
    return () => {
      live = false;
      owned?.dispose();
    };
  }, [design]);

  const { anchor, rotation_rad, size_mm, mirror } = placement;
  const key = `${anchor.face}:${anchor.barycentric.join(",")}:${rotation_rad}:${size_mm.join("x")}`;
  const geometry = useMemo(
    () => buildDecal(body.skin.geometry, body.skin.centroids, { anchor, rotationRad: rotation_rad, sizeMm: size_mm }).geometry,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [body, key],
  );
  useEffect(() => () => geometry.dispose(), [geometry]);

  // Mirror by flipping the texture, not the geometry, so the anchor and frame are untouched.
  useEffect(() => {
    if (!tex) return;
    tex.repeat.x = mirror ? -1 : 1;
    tex.offset.x = mirror ? 1 : 0;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
  }, [tex, mirror]);

  if (!tex) return null;
  return (
    <mesh
      geometry={geometry}
      raycast={() => null}
      onClick={ghost ? undefined : (e) => { e.stopPropagation(); select(placement.id); }}
    >
      <meshStandardMaterial
        map={tex}
        transparent
        opacity={ghost ? 0.65 : 1}
        depthWrite={false}
        polygonOffset
        polygonOffsetFactor={-4}
        polygonOffsetUnits={-4}
        roughness={0.9}
        emissive={selected ? "#3355ff" : "#000000"}
        emissiveIntensity={selected ? 0.25 : 0}
      />
    </mesh>
  );
}
