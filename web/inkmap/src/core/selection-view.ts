import * as THREE from "three";
import { anchorToPoint, type Anchor } from "./anchor.ts";

/** Find the most face-on unobstructed close-up. A surface normal can point
 * through the torso when a forearm rests beside it, so normal-only focus hides
 * the very placement the person asked to inspect. This changes only the view.
 */
export function selectionView(geometry: THREE.BufferGeometry, anchor: Anchor, distance: number): THREE.Vector3 | null {
  const { p, n } = anchorToPoint(geometry, anchor);
  const tangent = new THREE.Vector3(0, 0, 1).addScaledVector(n, -n.z);
  if (tangent.lengthSq() < 1e-6) tangent.set(1, 0, 0).addScaledVector(n, -n.x);
  tangent.normalize();
  const other = new THREE.Vector3().crossVectors(n, tangent).normalize();
  const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  const ray = new THREE.Raycaster();
  try {
    for (const angle of [0, 20, 40, 60, 75, 85]) for (let azimuth = 0; azimuth < (angle ? 12 : 1); azimuth++) {
      const radians = angle * Math.PI / 180, around = azimuth * Math.PI / 6;
      const direction = n.clone().multiplyScalar(Math.cos(radians))
        .addScaledVector(tangent, Math.sin(radians) * Math.cos(around))
        .addScaledVector(other, Math.sin(radians) * Math.sin(around));
      const position = p.clone().addScaledVector(direction, distance);
      ray.set(position, direction.clone().negate());
      const hit = ray.intersectObject(mesh, false)[0];
      if (hit && Math.abs(hit.distance - distance) < .002) return position;
    }
    return null;
  } finally { material.dispose(); }
}
