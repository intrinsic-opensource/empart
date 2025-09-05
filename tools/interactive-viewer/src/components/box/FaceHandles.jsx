// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// FaceHandles.jsx
// -----------------------------------------------------------------------------
// Renders clickable box faces for snapping and selection interactions.
// Highlights hovered and active faces; emits pick and hover events.
// -----------------------------------------------------------------------------

import React, { useMemo } from "react";
import * as THREE from "three";

/**
 * A single box’s “clickable faces” (selections + snapping preview).
 *
 * Props
 * -----
 * box                { center: THREE.Vector3, size: {x,y,z}, coarseness }
 * boxIdx             integer index in parent list
 * hover              {axis:'x'|'y'|'z', sign:±1} | null
 * active             same shape as `hover`           (while in snap‑mode)
 * onHover(face)      ⇢ void     — hover preview
 * onPick(face)       ⇢ void     — enter snap‑mode
 * setSelectedBoxIndex(idx)      — select this box
 */
export default function FaceHandles({
  box,
  boxIdx,
  hover,
  active,
  onHover,
  onPick,
  setSelectedBoxIndex,
}) {
  /* static geometry / material */
  const planeGeo = useMemo(() => new THREE.PlaneGeometry(1, 1), []);
  const basicMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
        depthWrite: false,
        alphaTest: 0.1,
      }),
    [],
  );

  /* 6 planes = 3 axes × 2 signs */
  const faces = [
    { axis: "x", sign: 1 },
    { axis: "x", sign: -1 },
    { axis: "y", sign: 1 },
    { axis: "y", sign: -1 },
    { axis: "z", sign: 1 },
    { axis: "z", sign: -1 },
  ];

  return faces.map((f, i) => {
    /** world‑space centre of this face */
    const pos = new THREE.Vector3(
      box.center.x + (f.axis === "x" ? (f.sign * box.size.x) / 2 : 0),
      box.center.y + (f.axis === "y" ? (f.sign * box.size.y) / 2 : 0),
      box.center.z + (f.axis === "z" ? (f.sign * box.size.z) / 2 : 0),
    );

    /** swap colour / opacity based on hover & active */
    const mat = useMemo(() => {
      const m = basicMat.clone();
      m.depthTest = false;
      if (active && active.axis === f.axis && active.sign === f.sign) {
        m.color.set("magenta"); // snapping
        m.opacity = 0.4;
      } else if (hover && hover.axis === f.axis && hover.sign === f.sign) {
        m.opacity = 0.15; // hover
      } else {
        m.color.set("white"); // idle
        m.opacity = 0.35;
      }
      return m;
    }, [hover, active]);

    return (
      <mesh
        key={i}
        geometry={planeGeo}
        material={mat}
        position={pos}
        rotation={[
          f.axis === "y" ? Math.PI / 2 : 0,
          f.axis === "x" ? -Math.PI / 2 : 0,
          0,
        ]}
        scale={[
          f.axis === "x" ? box.size.z : box.size.x,
          f.axis === "y" ? box.size.z : box.size.y,
          1,
        ]}
        renderOrder={999}
        onPointerOver={!active ? () => onHover(f) : undefined}
        onPointerOut={!active ? () => onHover(null) : undefined}
        onPointerDown={
          !active
            ? (e) => {
                e.stopPropagation();
                setSelectedBoxIndex(boxIdx);
                onPick(f);
              }
            : undefined
        }
        pointerEvents={active ? "none" : "auto"}
      />
    );
  });
}
