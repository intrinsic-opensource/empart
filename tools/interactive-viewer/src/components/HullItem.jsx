// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// HullItem.jsx
// -----------------------------------------------------------------------------
// React component for rendering a single hull (GLB model) inside the 3D scene.
//
// Features:
// • Handles hull selection, highlighting, and visibility
// • Applies custom materials based on selection state and wireframe mode
// • Emits callbacks on click for selection and hide actions
// • Uses @react-three/drei's <Edges> to draw selection outlines
// -----------------------------------------------------------------------------

import React, { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Edges } from "@react-three/drei";

/** Return { vertices, faces, volume } for the given (hull) scene */
function analyseHull(scene) {
  let vertices = 0,
    faces = 0,
    volume = null;

  scene.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry;
    const v = g.attributes.position.count;
    vertices += v;
    faces += g.index ? g.index.count / 3 : v / 3;

    if (o.userData.volume != null && typeof o.userData.volume === "number")
      volume = o.userData.volume;
  });

  return { vertices, faces, volume };
}

/**
 * One GLB hull inside the approximation layer.
 *
 * Props
 * -----
 * obj             GLTF result from useLoader
 * group           'nonSelect' | 'select'
 * localIdx        number                    // index inside its group
 * meshKey         string                    // stable key (obj.scene.uuid)
 * wireframe       boolean
 * selectedHulls   [{ meshKey, group, idx, … }]
 * hullVolumes     { nonSelect:number[], select:number[] }
 * hiddenHulls     Set<string>
 * onSelectHull    (meshKey, group, idx, stats) => void
 * onPickForHide   (meshKey) => void
 */
export default function HullItem({
  obj,
  group,
  localIdx,
  meshKey,
  wireframe,
  selectedHulls,
  hullVolumes,
  hiddenHulls,
  onSelectHull,
  onPickForHide,
}) {
  /* grab first mesh once */
  const firstMesh = useMemo(() => {
    let m;
    obj.scene.traverse((c) => !m && c.isMesh && (m = c));
    return m;
  }, [obj]);

  /* keep untouched material for deselection */
  const originalMat = useMemo(() => firstMesh?.material, [firstMesh]);

  /* outline ref */
  const outlineRef = useRef(null);

  /* update materials every render */
  useEffect(() => {
    obj.scene.traverse((child) => {
      if (!child.isMesh) return;

      child.material.toneMapped = false;
      child.material.flatShading = true;
      child.material.transparent = true;
      child.material.opacity = 0.9;
      child.material.wireframe = wireframe;

      const picked = selectedHulls.some(
        (h) => h.group === group && h.idx === localIdx,
      );

      if (picked) {
        if (!child.userData.__highlightMat) {
          const m = child.material.clone();
          m.color.set("#ff8800");
          m.vertexColors = false;
          child.userData.__highlightMat = m;
        }
        child.material = child.userData.__highlightMat;
      } else {
        child.material = originalMat;
      }
      child.material.needsUpdate = true;
    });

    /* keep outline always white & on top */
    if (outlineRef.current) {
      const m = outlineRef.current.material;
      m.depthTest = false;
      m.depthWrite = false;
      m.toneMapped = false;
      m.vertexColors = false;
      m.color.set("#ffffff");
      m.needsUpdate = true;
    }
  }, [obj, wireframe, selectedHulls, group, localIdx, originalMat]);

  /* hidden? */
  if (hiddenHulls.has(meshKey)) return null;

  return (
    <primitive
      object={obj.scene}
      userData={{ isModelMesh: true, isHull: true }}
      onPointerDown={(e) => {
        if ((e.button ?? e.nativeEvent.button) !== 0) return; // left‑click only
        const mesh = e.object;
        if (!mesh.isMesh || mesh.userData.isHelper) return;
        e.stopPropagation();

        onPickForHide(meshKey);

        const volume =
          group === "nonSelect"
            ? hullVolumes.nonSelect[localIdx]
            : hullVolumes.select[localIdx];

        const { vertices, faces } = analyseHull(obj.scene);
        onSelectHull(meshKey, group, localIdx, { vertices, faces, volume });
      }}
    >
      {selectedHulls.some((h) => h.meshKey === meshKey) && firstMesh && (
        <Edges
          ref={outlineRef}
          geometry={firstMesh.geometry}
          scale={1.01}
          color="white"
          lineWidth={1}
          renderOrder={1_000}
        />
      )}
    </primitive>
  );
}
