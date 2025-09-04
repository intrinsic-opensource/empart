// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// geometry.js
// ---------------------------------------------------------------------------
// Adjusts boxes to tightly fit geometry vertices within a scene, preserving bounds.

import * as THREE from "three";

export function snapBoxesToGeometry(boxes, scene) {
  if (!scene) return boxes;

  // 2-a  gather *model* meshes
  const modelMeshes = [];
  scene.traverse((o) => {
    if (o.isMesh && o.userData.isModelMesh) modelMeshes.push(o);
  });

  // 2-b  dump their vertices (world space) into one flat list
  const verts = [];
  const v = new THREE.Vector3();
  for (const m of modelMeshes) {
    const pos = m.geometry.attributes.position;
    for (let i = 0; i < pos.count; ++i) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
      verts.push(v.clone()); // clone because v is reused
    }
  }

  // 2-c  process every user box
  return boxes.map((b) => {
    const boxMin = new THREE.Vector3(
      b.center.x - b.size.x / 2,
      b.center.y - b.size.y / 2,
      b.center.z - b.size.z / 2,
    );
    const boxMax = new THREE.Vector3(
      b.center.x + b.size.x / 2,
      b.center.y + b.size.y / 2,
      b.center.z + b.size.z / 2,
    );

    // find verts already inside the box
    let min = new THREE.Vector3(Infinity, Infinity, Infinity);
    let max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

    for (const p of verts) {
      if (
        p.x < boxMin.x ||
        p.x > boxMax.x ||
        p.y < boxMin.y ||
        p.y > boxMax.y ||
        p.z < boxMin.z ||
        p.z > boxMax.z
      )
        continue;

      min.min(p);
      max.max(p);
    }

    // nothing inside? keep the old box
    if (!Number.isFinite(min.x)) return b;
    // ── add a tiny outward “breathing room” ───────────────────────────
    const EPS = 5e-4; // 0.0005 scene units

    // proposed extents with clearance …
    const wantMin = min.clone().subScalar(EPS);
    const wantMax = max.clone().addScalar(EPS);

    // …but never let a face move outward
    const newMin = wantMin.clone().max(boxMin); // per-axis max()
    const newMax = wantMax.clone().min(boxMax); // per-axis min()
    const newCenter = newMin.clone().add(newMax).multiplyScalar(0.5);
    const newSize = newMax.clone().sub(newMin);
    return {
      ...b,
      center: newCenter,
      size: { x: newSize.x, y: newSize.y, z: newSize.z },
    };
  });
}
