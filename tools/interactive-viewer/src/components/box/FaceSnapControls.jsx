// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// FaceSnapControls.jsx
// -----------------------------------------------------------------------------
// Renders a temporary snapping preview while the user is in "snap-mode".
//
// Features:
// • Displays a preview sphere that follows the pointer on the model surface
// • On click, adjusts a selected box face based on hit location
// • Exits snap-mode after update or on Escape key
//
// Mounted only when snap-mode is active.
// -----------------------------------------------------------------------------

import React, { useRef, useEffect, useCallback } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import * as THREE from "three";

/**
 * FaceSnapControls
 * -------------------------------------------------------------
 * A lightweight component that is mounted **only while** the user
 * is in “snap‑mode”.  It renders a single preview sphere that
 * follows the pointer, listens for a click on the model surface and
 * then updates the currently‑selected bounding box before exiting
 * snap‑mode again.
 *
 * Props
 * -----
 * @param {React.MutableRefObject<THREE.Scene>} sceneRef – root scene ref
 * @param {object|null}  activeFace   – { axis:'x'|'y'|'z', sign:±1 } or null
 * @param {function}     setActiveFace – setter to leave snap‑mode (pass null)
 * @param {object|null}  selectedBox  – { center:THREE.Vector3, size:{x,y,z} }
 * @param {function}     updateBox    – callback(updatedBox) → void
 */
export default function FaceSnapControls({
  sceneRef,
  activeFace,
  setActiveFace,
  selectedBox,
  updateBox,
}) {
  const { camera, raycaster, pointer } = useThree();
  const previewRef = useRef(null);
  const ignoreNextDown = useRef(false);

  /* ───────────────── frame loop ───────────────── */
  useFrame(() => {
    if (!activeFace || !previewRef.current) return;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster
      .intersectObjects(sceneRef.current?.children ?? [], true)
      .find((h) => h.object.userData.isModelMesh);

    previewRef.current.visible = Boolean(hit);
    if (hit) previewRef.current.position.copy(hit.point);
  });

  /* ───────────────── click handler ───────────────── */
  const onPointerDown = useCallback(
    (e) => {
      // The very first pointerdown after enabling snap‑mode belongs to
      // the face that triggered it – ignore that one.
      if (ignoreNextDown.current) {
        ignoreNextDown.current = false;
        return;
      }
      if (!activeFace || !selectedBox) return;

      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster
        .intersectObjects(sceneRef.current?.children ?? [], true)
        .find((h) => h.object.userData.isModelMesh);
      if (!hit) return;

      /* 1 ▸ compute new extents  */
      const p = hit.point;
      const { axis, sign } = activeFace;
      const half = {
        x: selectedBox.size.x / 2,
        y: selectedBox.size.y / 2,
        z: selectedBox.size.z / 2,
      };
      const min = new THREE.Vector3(
        selectedBox.center.x - half.x,
        selectedBox.center.y - half.y,
        selectedBox.center.z - half.z,
      );
      const max = new THREE.Vector3(
        selectedBox.center.x + half.x,
        selectedBox.center.y + half.y,
        selectedBox.center.z + half.z,
      );

      if (sign > 0) max[axis] = p[axis];
      else min[axis] = p[axis];

      // tiny outward padding so the face never inverts
      const EPS = 5e-4;
      min[axis] -= EPS * sign;
      max[axis] += EPS * sign;

      /* 2 ▸ derive new center & size */
      const newCenter = new THREE.Vector3(
        (min.x + max.x) / 2,
        (min.y + max.y) / 2,
        (min.z + max.z) / 2,
      );
      const newSize = {
        x: max.x - min.x,
        y: max.y - min.y,
        z: max.z - min.z,
      };

      updateBox({ ...selectedBox, center: newCenter, size: newSize });
      setActiveFace(null); // leave snap‑mode
    },
    [
      activeFace,
      selectedBox,
      sceneRef,
      raycaster,
      pointer,
      camera,
      updateBox,
      setActiveFace,
    ],
  );

  /* ───────────────── lifecycle ───────────────── */
  useEffect(() => {
    if (!activeFace) return undefined;

    // Ignore the pointer‑down that started snap‑mode
    ignoreNextDown.current = true;

    window.addEventListener("pointerdown", onPointerDown);
    const onEsc = (e) => e.key === "Escape" && setActiveFace(null);
    window.addEventListener("keydown", onEsc);

    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onEsc);
    };
  }, [activeFace, onPointerDown, setActiveFace]);

  /* ───────────────── render ───────────────── */
  if (!activeFace) return null; // render nothing when snap‑mode is off

  return (
    <mesh ref={previewRef} visible={false} userData={{ isHelper: true }}>
      <sphereGeometry args={[0.002, 12, 12]} />
      <meshBasicMaterial color="magenta" />
    </mesh>
  );
}
