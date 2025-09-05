// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// BoxSelectControls.jsx
// -----------------------------------------------------------------------------
// React component for drawing axis-aligned bounding boxes on a 3D model surface.
//
// Features:
// • Enables click-and-drag creation of bounding boxes in 'corner' or 'center' mode
// • Does not add box if too small (less than 1mm in any dimension)
// • Provides visual feedback with a drag helper box and preview sphere
// • Emits box data ({ center, size, coarseness }) to the parent via `onAddBox`
// • Temporarily disables orbit controls while dragging
//
// -----------------------------------------------------------------------------

import React, { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";

/**
 * Lets the user drag‑out axis‑aligned boxes on the surface of the model.
 *
 * Props
 * -----
 * enabled      boolean                    // user can start a drag
 * onAddBox     ({ center, size, coarseness }) => void
 * orbitRef     React.ref<OrbitControls>   // for temporarily disabling orbit
 * mode         'corner' | 'center'        // how the drag behaves
 */
export default function BoxSelectControls({
  enabled,
  onAddBox,
  orbitRef,
  mode = "corner",
}) {
  const { gl, camera, raycaster, pointer, scene } = useThree();
  const modeRef = useRef(mode);
  const helperRef = useRef(); // cyan live‑box
  const highlightRef = useRef(); // tiny magenta sphere
  const isDragging = useRef(false);
  const dragStartPt = useRef(null);

  /* keep latest mode string */
  useEffect(() => void (modeRef.current = mode), [mode]);

  /* helper: freeze/unfreeze orbit controls */
  const setOrbitEnabled = useCallback(
    (v) => orbitRef?.current && (orbitRef.current.enabled = v),
    [orbitRef],
  );

  /* helper: translate DOM coords → NDC */
  const updatePointer = (ev) => {
    const r = gl.domElement.getBoundingClientRect();
    pointer.set(
      ((ev.clientX - r.left) / r.width) * 2 - 1,
      -((ev.clientY - r.top) / r.height) * 2 + 1,
    );
  };

  /* helper: ray‑pick first model‑mesh under cursor */
  const pick = () => {
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(scene.children, true);
    return hits.find((h) => h.object.userData.isModelMesh);
  };

  /* ─────────────── event handlers ─────────────── */
  const onPointerMove = (e) => {
    updatePointer(e);
    const hit = pick();

    /* preview sphere */
    if (highlightRef.current) {
      highlightRef.current.visible = enabled && !!hit;
      if (hit) highlightRef.current.position.copy(hit.point);
    }

    /* drag‑resize */
    if (!isDragging.current || !hit) return;
    const p0 = dragStartPt.current;
    const p1 = hit.point.clone();

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();

    if (modeRef.current === "corner") {
      const min = p0.clone().min(p1);
      const max = p0.clone().max(p1);
      center.copy(min).add(max).multiplyScalar(0.5);
      size
        .copy(max)
        .sub(min)
        .max(new THREE.Vector3(0.002, 0.002, 0.002));
    } else {
      // 'center' mode
      const delta = p1.sub(p0);
      size.set(
        Math.max(Math.abs(delta.x) * 2, 0.002),
        Math.max(Math.abs(delta.y), 0.002),
        Math.max(Math.abs(delta.z) * 2, 0.002),
      );
      center.set(p0.x, p0.y + delta.y / 2, p0.z);
    }

    helperRef.current.position.copy(center);
    helperRef.current.scale.copy(size);
  };

  const onPointerDown = (e) => {
    if (!enabled || e.button !== 0) return;
    updatePointer(e);
    const hit = pick();
    if (!hit) return;

    isDragging.current = true;
    dragStartPt.current = hit.point.clone();

    helperRef.current.visible = true;
    helperRef.current.position.copy(hit.point);
    helperRef.current.scale.setScalar(0.0001);
    setOrbitEnabled(false);
  };

  const onPointerUp = () => {
    if (!isDragging.current) return;
    isDragging.current = false;

    const { position, scale } = helperRef.current;

    // ────────────── reject degenerate “point” boxes ──────────────
    const MIN = 0.001; // 1 mm in model units
    const tooSmall = scale.x <= MIN && scale.y <= MIN && scale.z <= MIN;
    if (tooSmall) {
      // ignore accidental clicks
      helperRef.current.visible = false;
      setOrbitEnabled(true);
      return;
    }

    onAddBox({
      center: position.clone(),
      size: { x: scale.x, y: scale.y, z: scale.z },
      coarseness: 0,
    });

    helperRef.current.visible = false;
    setOrbitEnabled(true);
  };

  /* register global listeners once */
  useEffect(() => {
    const el = gl.domElement;
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    return () => {
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [enabled]); // eslint‑disable‑line react-hooks/exhaustive-deps

  /* ─────────────── render helpers ─────────────── */
  return (
    <>
      {/* drag helper box */}
      <mesh ref={helperRef} visible={false} userData={{ isHelper: true }}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="cyan" wireframe />
      </mesh>

      {/* hover‑preview sphere (≈ 2 mm) */}
      <mesh
        ref={highlightRef}
        visible={false}
        userData={{ isHelper: true }}
        renderOrder={1}
      >
        <sphereGeometry args={[0.0015, 15, 15]} />
        <meshBasicMaterial color="magenta" toneMapped={false} />
      </mesh>
    </>
  );
}
