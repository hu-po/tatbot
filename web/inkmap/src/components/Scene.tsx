import { Component, Suspense, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { Canvas, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { useStore, type LoadedBody } from "../store.ts";
import {
  applyBodyRotation,
  BODY_SPEC,
  buildPosedSkin,
  buildSkin,
  canonicalSurfaceBytes,
  canonicalTopologyBytes,
  sha256Hex,
  type BodySpec,
} from "../core/body.ts";
import { POSE_CATALOG, poseRecord } from "../core/pose.ts";
import { anchorToPoint, frameAt, pointToAnchor } from "../core/anchor.ts";
import { buildDecal } from "../core/decal.ts";
import { selectionView } from "../core/selection-view.ts";
import { artworkTexture } from "../core/svg.ts";
import type { Placement } from "../core/schema.ts";
import { AtlasIndex, parseAtlas, validateExclusionMask, type RegionRef } from "../core/atlas.ts";
import type { OrbitControls as OrbitControlsImpl } from "three/examples/jsm/controls/OrbitControls.js";

export function Scene() {
  const spec = BODY_SPEC;
  const poseId = useStore((s) => s.poseId);
  const neutralLight = useStore((s) => s.neutralLight);
  const surfaceInteraction = useStore((s) => s.surfaceInteraction);
  const eye = spec.eyeHeight;
  return (
    <Canvas
      camera={{ position: [1.6, -2.4, eye * 0.85], up: [0, 0, 1], fov: 38, near: 0.01, far: 50 }}
      onCreated={({ camera }) => camera.up.set(0, 0, 1)}
      gl={{ antialias: false, powerPreference: "high-performance", precision: "mediump" }}
      dpr={1}
      frameloop="demand"
    >
      <color attach="background" args={["#1b1d22"]} />
      {neutralLight
        ? <><ambientLight intensity={1.25} /><directionalLight position={[0, -4, 5]} intensity={0.75} /></>
        : <><hemisphereLight args={["#ffffff", "#3a3f4a", 1.0]} /><directionalLight position={[3, -4, 6]} intensity={1.5} /></>}
      <CanvasErrorBoundary>
        <Suspense fallback={null}>
          <Body key={`${spec.id}:${poseId}`} spec={spec} poseId={poseId} />
        </Suspense>
        <AtlasOverlay />
        <RegionHighlight />
        <Placements />
        <ScenarioTrace />
        <ScenarioSiteMarker />
        <ShowcaseCamera />
        <ProjectCamera />
        <SelectionControls />
      </CanvasErrorBoundary>
      <OrbitControls enabled={!surfaceInteraction} makeDefault target={[0, 0, eye * 0.6]} minDistance={0.3} maxDistance={8} />
    </Canvas>
  );
}

function ProjectCamera() {
  const { camera, controls, invalidate } = useThree();
  const body = useStore(s => s.body);
  const revision = useStore(s => s.cameraRevision);
  const command = useStore(s => s.cameraCommand);
  const showcase = new URLSearchParams(window.location.search).get("showcase") === "1";
  useEffect(() => {
    if (showcase || !body || !controls) return;
    const orbit = controls as OrbitControlsImpl;
    const saved = useStore.getState().cameraSnapshot;
    if (!saved) return;
    camera.position.fromArray(saved.position); orbit.target.fromArray(saved.target);
    if (camera instanceof THREE.PerspectiveCamera) { camera.fov = saved.fov; camera.updateProjectionMatrix(); }
    camera.lookAt(orbit.target); orbit.update(); invalidate();
  }, [body, revision, camera, controls, invalidate, showcase]);
  useEffect(() => {
    if (showcase || !body || !controls || !command) return;
    const orbit = controls as OrbitControlsImpl;
    const center = body.skin.bbox.getCenter(new THREE.Vector3());
    const size = body.skin.bbox.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z);
    const distance = Math.max(radius * 1.35, 0.55);
    let target = center;
    let position: THREE.Vector3;
    if (command.preset === "selection") {
      // The command names the tattoo once; dragging it afterwards moves the
      // tattoo under a still camera, not the camera after the tattoo.
      const state = useStore.getState();
      const placement = state.placements.find(item => item.id === state.selected);
      if (!placement) return;
      const frame = anchorToPoint(body.skin.geometry, placement.anchor);
      target = frame.p;
      const close = THREE.MathUtils.clamp(Math.max(...placement.size_mm) / 1000 * 8, 0.32, 0.72);
      position = selectionView(body.skin.geometry, placement.anchor, close)
        ?? target.clone().addScaledVector(frame.n, close).addScaledVector(new THREE.Vector3(0, 0, 1), close * 0.12);
    } else {
      const direction = command.preset === "front" ? new THREE.Vector3(0, -1, 0)
        : command.preset === "back" ? new THREE.Vector3(0, 1, 0)
          : command.preset === "left" ? new THREE.Vector3(1, 0, 0)
            : command.preset === "right" ? new THREE.Vector3(-1, 0, 0)
              : new THREE.Vector3(0.95, -1.55, 0.55).normalize();
      position = center.clone().addScaledVector(direction, distance);
    }
    camera.position.copy(position);
    orbit.target.copy(target);
    camera.lookAt(target);
    orbit.update();
    invalidate();
    useStore.setState({ cameraSnapshot: { position: camera.position.toArray(), target: target.toArray(), fov: camera instanceof THREE.PerspectiveCamera ? camera.fov : 38 } });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [body, camera, command, controls, invalidate, showcase]);
  useEffect(() => {
    if (!controls || showcase) return;
    const orbit = controls as OrbitControlsImpl;
    const save = () => {
      if (!useStore.getState().projectReady) return;
      useStore.setState({ cameraSnapshot: { position: camera.position.toArray(), target: orbit.target.toArray(),
        fov: camera instanceof THREE.PerspectiveCamera ? camera.fov : 38 } });
    };
    orbit.addEventListener("end", save);
    return () => orbit.removeEventListener("end", save);
  }, [camera, controls, showcase]);
  return null;
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
    const matchingScenario = scenario && body.spec.id === scenario.body.model_spec_id && body.poseId === scenario.pose.id
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
    if (!body || !scenario || body.spec.id !== scenario.body.model_spec_id || body.poseId !== scenario.pose.id) return null;
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
    if (!body || !scenario || body.spec.id !== scenario.body.model_spec_id || body.poseId !== scenario.pose.id) return [];
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
        const restSkin = buildSkin(gltf.scene);
        const surfaceSha256 = await sha256Hex(canonicalSurfaceBytes(restSkin.geometry));
        const topologySha256 = await sha256Hex(canonicalTopologyBytes(restSkin.geometry));
        if (surfaceSha256 !== spec.restSurfaceSha256 || surfaceSha256 !== POSE_CATALOG.rest_surface_sha256) {
          throw new Error(`body_rest_surface_mismatch: loaded ${surfaceSha256.slice(0, 12)}…`);
        }
        if (topologySha256 !== spec.topologySha256 || topologySha256 !== POSE_CATALOG.topology_sha256) {
          throw new Error(`body_topology_mismatch: loaded ${topologySha256.slice(0, 12)}…`);
        }
        if (POSE_CATALOG.rest_asset.path !== spec.path || POSE_CATALOG.pose_asset.path !== spec.posePath) {
          throw new Error("body_model_unpinned: browser paths differ from the pose catalog");
        }
        const pose = poseRecord(poseId);
        const t1 = performance.now();
        const [bytes, poseBytes] = await Promise.all([
          fetch(spec.path).then((response) => response.arrayBuffer()),
          fetch(spec.posePath).then((response) => response.arrayBuffer()),
        ]);
        const t2 = performance.now();
        const [assetSha256, poseAssetSha256] = await Promise.all([
          sha256Hex(bytes),
          sha256Hex(poseBytes),
        ]);
        if (
          assetSha256 !== POSE_CATALOG.rest_asset.sha256
          || bytes.byteLength !== POSE_CATALOG.rest_asset.size
          || poseAssetSha256 !== POSE_CATALOG.pose_asset.sha256
          || poseBytes.byteLength !== POSE_CATALOG.pose_asset.size
        ) throw new Error("body_asset_hash_mismatch: browser body or pose cache");
        const poseChunk = poseBytes.slice(pose.byte_offset, pose.byte_offset + pose.byte_length);
        if (await sha256Hex(poseChunk) !== pose.chunk_sha256) {
          throw new Error(`body_asset_hash_mismatch: pose ${poseId}`);
        }
        const skin = buildPosedSkin(restSkin, poseChunk);
        if (await sha256Hex(canonicalSurfaceBytes(skin.geometry)) !== pose.surface_sha256) {
          throw new Error(`body_rest_surface_mismatch: pose ${poseId}`);
        }
        const nominalBounds = { min: skin.bbox.min.toArray(), max: skin.bbox.max.toArray() } as LoadedBody["nominalBounds"];
        applyBodyRotation(skin, pose.body_rotation_xyzw);
        const t3 = performance.now();
        console.info(`[inkmap] SOMA timings: rest ${(t1 - t0).toFixed(0)} ms, fetch ${(t2 - t1).toFixed(0)} ms, verify/pose ${(t3 - t2).toFixed(0)} ms`);
        if (cancelled) return;
        const loaded: LoadedBody = {
          spec,
          restSkin,
          skin,
          assetSha256,
          surfaceSha256,
          topologySha256,
          poseAssetSha256,
          poseId,
          nominalBounds,
        };
        const [res, exclusionResponse] = await Promise.all([
          fetch(`bodies/${spec.id}.regions.json`),
          fetch(POSE_CATALOG.exclusion_asset.path),
        ]);
        if (!res.ok) throw new Error(`body_asset_missing: region atlas HTTP ${res.status}`);
        if (!exclusionResponse.ok) throw new Error(`body_asset_missing: exclusion mask HTTP ${exclusionResponse.status}`);
        const exclusionBytes = await exclusionResponse.arrayBuffer();
        if (
          exclusionBytes.byteLength !== POSE_CATALOG.exclusion_asset.size
          || await sha256Hex(exclusionBytes) !== POSE_CATALOG.exclusion_asset.sha256
        ) throw new Error("body_asset_hash_mismatch: upstream exclusion mask");
        const raw = parseAtlas(await res.json(), restSkin.centroids.length / 3);
        if (
          raw.body.model_spec_id !== spec.id
          || raw.body.model_spec_sha256 !== spec.modelSpecSha256
          || raw.body.identity_sha256 !== spec.identitySha256
          || raw.body.topology_sha256 !== topologySha256
          || raw.body.rest_surface_sha256 !== surfaceSha256
          || raw.body.asset_sha256 !== assetSha256
        ) {
          throw new Error("body_asset_hash_mismatch: region atlas does not bind the loaded SOMA surface");
        }
        if (raw.upstream_exclusions.asset_sha256 !== POSE_CATALOG.exclusion_asset.sha256) {
          throw new Error("body_asset_hash_mismatch: atlas exclusion digest differs from the reviewed mask");
        }
        validateExclusionMask(raw, exclusionBytes);
        const atlas = new AtlasIndex(raw, restSkin.geometry, restSkin.centroids);
        if (cancelled) return;
        setBody(loaded);
        useStore.getState().setAtlas(atlas);
        setError(null);
        const size = new THREE.Vector3();
        skin.bbox.getSize(size);
        console.info(`[inkmap] body ${spec.id} asset=${assetSha256.slice(0, 12)}… surface=${surfaceSha256.slice(0, 12)}… faces=${skin.centroids.length / 3} height=${size.z.toFixed(3)} m`);
      } catch (e) {
        setError((e as Error).message);
      }
    })();
    return () => { cancelled = true; };
  }, [gltf, spec, poseId, setBody, setError]);

  if (!body || body.spec.id !== spec.id) return null;

  const onMove = (e: ThreeEvent<PointerEvent>) => {
    const state = useStore.getState();
    if (e.faceIndex == null) return;
    if (state.surfaceInteraction === "move" && state.selected) {
      e.stopPropagation();
      state.update(state.selected, { anchor: pointToAnchor(body.skin.geometry, e.faceIndex, e.point) });
      return;
    }
    if (!placing) return;
    e.stopPropagation();
    setHover(pointToAnchor(body.skin.geometry, e.faceIndex, e.point));
  };
  const onDown = (e: ThreeEvent<PointerEvent>) => {
    press.current = { x: e.clientX, y: e.clientY, t: performance.now() };
  };
  const onUp = (e: ThreeEvent<PointerEvent>) => {
    if (useStore.getState().surfaceInteraction) {
      useStore.getState().setSurfaceInteraction(null);
      press.current = null;
      e.stopPropagation();
      return;
    }
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
    const site = pending?.intent.site;
    if (!body || !atlas || !site) return null;
    const lat = site.laterality === "left" || site.laterality === "right" ? site.laterality : null;
    const faces = atlas.facesOf(site.id, lat);
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

function SelectionControls() {
  const body = useStore(s => s.body);
  const selected = useStore(s => s.selected);
  const placement = useStore(s => s.placements.find(item => item.id === s.selected));
  const interaction = useStore(s => s.surfaceInteraction);
  const frame = useMemo(() => body && placement
    ? frameAt(body.skin.geometry, placement.anchor, placement.rotation_rad)
    : null, [body, placement]);
  const line = useMemo(() => {
    if (!frame || !placement) return null;
    const half = placement.size_mm[0] / 2000;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      frame.p.clone().addScaledVector(frame.u, -half).addScaledVector(frame.n, 0.004),
      frame.p.clone().addScaledVector(frame.u, half).addScaledVector(frame.n, 0.004),
    ]);
    const material = new THREE.LineBasicMaterial({ color: "#ffe071", depthTest: false });
    const object = new THREE.Line(geometry, material);
    object.raycast = () => undefined;
    return object;
  }, [frame, placement]);
  useEffect(() => () => {
    line?.geometry.dispose();
    (line?.material as THREE.Material | undefined)?.dispose();
  }, [line]);
  if (!body || !placement || !frame || selected !== placement.id) return null;
  const handlePosition = frame.p.clone().addScaledVector(frame.u, placement.size_mm[0] / 2000 + 0.025).addScaledVector(frame.n, 0.008);
  const drag = (kind: "rotate" | "size") => ({
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      event.currentTarget.setPointerCapture(event.pointerId);
      useStore.getState().setSurfaceInteraction(kind);
    },
    onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (useStore.getState().surfaceInteraction !== kind || event.movementX === 0) return;
      const current = useStore.getState().placements.find(item => item.id === placement.id);
      if (!current) return;
      if (kind === "rotate") useStore.getState().update(current.id, { rotation_rad: current.rotation_rad + event.movementX * 0.012 });
      else useStore.getState().nudgeSize(Math.exp(event.movementX * 0.012));
    },
    onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      event.currentTarget.releasePointerCapture(event.pointerId);
      useStore.getState().setSurfaceInteraction(null);
    },
    onPointerCancel: () => useStore.getState().setSurfaceInteraction(null),
  });
  return <>
    {line && <primitive object={line} />}
    <Html center position={handlePosition} zIndexRange={[12, 0]}>
      <div className="surface-handles" onPointerDown={event => event.stopPropagation()}>
        <span className="ruler" aria-label={`Tattoo width ${placement.size_mm[0].toFixed(0)} millimeters`}>{placement.size_mm[0].toFixed(0)} mm</span>
        <button type="button" className={interaction === "rotate" ? "active" : ""} aria-label="Drag to rotate tattoo; A and D keys also rotate" title="Drag horizontally to rotate" {...drag("rotate")}>↻</button>
        <button type="button" className={interaction === "size" ? "active" : ""} aria-label="Drag to resize tattoo; W and S keys also resize" title="Drag horizontally to resize" {...drag("size")}>↔</button>
      </div>
    </Html>
  </>;
}

function Placements() {
  const body = useStore((s) => s.body);
  const placements = useStore((s) => s.placements);
  const placing = useStore((s) => s.placing);
  const hover = useStore((s) => s.hover);
  const designs = useStore((s) => s.designs);
  const selected = useStore((s) => s.selected);
  const draft = useStore((s) => s.draft);
  const atlas = useStore((s) => s.atlas);
  if (!body) return null;
  const ghostDesign = placing ? designs.find((d) => d.id === placing) : undefined;
  const ghost: Placement | null = placing && hover && ghostDesign
    ? { id: "__ghost", design_id: ghostDesign.id, anchor: hover, rotation_rad: draft.rotation_rad, size_mm: draft.size_mm, mirror: false }
    : null;
  // The ghost is judged by the same atlas rule commit applies, so a spot that
  // looks placeable cannot then be refused for being outside the domain.
  const ghostInvalid = Boolean(ghost && atlas && !atlas.isValidAnchor(ghost.anchor));
  return (
    <>
      {placements.map((p) => <Decal key={p.id} placement={p} selected={p.id === selected} />)}
      {ghost && <Decal placement={ghost} ghost invalid={ghostInvalid} />}
    </>
  );
}

function Decal({ placement, ghost = false, selected = false, invalid = false }: { placement: Placement; ghost?: boolean; selected?: boolean; invalid?: boolean }) {
  const body = useStore((s) => s.body)!;
  const designs = useStore((s) => s.designs);
  const select = useStore((s) => s.select);
  const design = designs.find((d) => d.id === placement.design_id);
  const [tex, setTex] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    let live = true;
    if (!design) setTex(null);
    if (design) {
      artworkTexture(design, placement.size_mm).then((texture) => {
        if (!live) {
          texture.dispose();
          return;
        }
        setTex(texture);
      }).catch(console.error);
    }
    return () => {
      live = false;
    };
  }, [design, placement.size_mm[0], placement.size_mm[1]]);
  // Keep the mesh pickable during resizing; retire its previous texture only
  // after the replacement has reached the rendered mesh.
  useEffect(() => () => tex?.dispose(), [tex]);

  const { anchor, rotation_rad, size_mm, mirror } = placement;
  const key = `${anchor.face}:${anchor.barycentric.join(",")}:${rotation_rad}:${size_mm.join("x")}`;
  // The ghost follows the pointer anywhere on the skin, so a refusal here is
  // routine (girth, seam, open edge). Thrown from render it would reach the
  // canvas error boundary and blank the whole body; report it instead.
  const { geometry, refusal } = useMemo(() => {
    try {
      return {
        geometry: buildDecal(body.restSkin.geometry, body.skin.geometry, {
          anchor,
          rotationRad: rotation_rad,
          sizeMm: size_mm,
        }).geometry,
        refusal: null,
      };
    } catch (error) {
      return { geometry: null, refusal: error instanceof Error ? error.message : String(error) };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [body, key]);
  useEffect(() => () => geometry?.dispose(), [geometry]);
  useEffect(() => {
    if (!ghost) return;
    const { error, setError } = useStore.getState();
    if (refusal) setError(`preview: ${refusal}`);
    else if (error?.startsWith("preview: ")) setError(null);
  }, [ghost, refusal]);

  // Mirror by flipping the texture, not the geometry, so the anchor and frame are untouched.
  useEffect(() => {
    if (!tex) return;
    tex.repeat.x = mirror ? -1 : 1;
    tex.offset.x = mirror ? 1 : 0;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
  }, [tex, mirror]);

  if (!tex || !geometry) return null;
  return (
    <mesh
      geometry={geometry}
      onPointerDown={ghost ? undefined : () => {
        const state = useStore.getState();
        // While placing, the body under the pointer takes the click: a new
        // tattoo may land over an old one without selecting it instead.
        if (state.placing) return;
        if (state.selected !== placement.id) state.select(placement.id);
        if (useStore.getState().selected === placement.id) useStore.getState().setSurfaceInteraction("move");
      }}
      onPointerUp={ghost ? undefined : () => useStore.getState().setSurfaceInteraction(null)}
      onClick={ghost ? undefined : (e) => { if (useStore.getState().placing) return; e.stopPropagation(); select(placement.id); }}
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
        emissive={invalid ? "#ff2020" : selected ? "#3355ff" : "#000000"}
        emissiveIntensity={invalid ? 0.6 : selected ? 0.25 : 0}
      />
    </mesh>
  );
}
