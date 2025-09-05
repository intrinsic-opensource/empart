// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// useBoundingBoxes.js
// ---------------------------------------------------------------------------
// React hook for managing bounding boxes in a 3D scene.
//
// Features:
// • Stores and updates box size, position, coarseness, and selection state
// • Supports adding, deleting, and clearing boxes
// • Integrates with Three.js to snap boxes to geometry
// • Exposes UI-related flags (sliders, visibility, interaction state)

import { useCallback, useState } from "react";
import { Vector3 } from "three";
import { snapBoxesToGeometry } from "../utils";

export default function useBoundingBoxes({ sceneRef } = {}) {
  /* ───── core state ───── */
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const [defaultSize, setDefaultSize] = useState({ x: 0.1, y: 0.1, z: 0.1 });
  const [defaultCoarseness, setDefaultCoarseness] = useState(1);
  const [showCanvasSliders, setShowCanvasSliders] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [boxMode, setBoxMode] = useState("center");
  const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);
  const [hoverFace, setHoverFace] = useState(null);
  const [activeFace, setActiveFace] = useState(null);
  const [errors, setErrors] = useState([]);

  /* ───── simple helpers ───── */
  const handleAddBox = useCallback((box) => {
    setBoundingBoxes((prev) => [...prev, box]);
  }, []);

  const updateBoxPosition = useCallback((idx, axis, value) => {
    setBoundingBoxes((bs) =>
      bs.map((b, i) =>
        i === idx
          ? {
              ...b,
              center: new Vector3(
                axis === "x" ? value : b.center.x,
                axis === "y" ? value : b.center.y,
                axis === "z" ? value : b.center.z,
              ),
            }
          : b,
      ),
    );
  }, []);

  const updateBoxSize = useCallback((idx, axis, value) => {
    setBoundingBoxes((bs) =>
      bs.map((b, i) =>
        i === idx ? { ...b, size: { ...b.size, [axis]: value } } : b,
      ),
    );
  }, []);

  const updateBoxCoarsenessValue = useCallback((idx, value) => {
    setBoundingBoxes((bs) =>
      bs.map((b, i) => (i === idx ? { ...b, coarseness: value } : b)),
    );
  }, []);

  const updateDefaultSizeValue = useCallback((axis, value) => {
    setDefaultSize((ds) => ({ ...ds, [axis]: value }));
  }, []);

  const updateDefaultCoarsenessValue = useCallback((value) => {
    setDefaultCoarseness(value);
  }, []);

  const handleDeleteBox = useCallback((idx) => {
    setBoundingBoxes((bs) => bs.filter((_, i) => i !== idx));
  }, []);

  const handleClear = useCallback(() => setBoundingBoxes([]), []);

  const handleSnapBoxes = useCallback(() => {
    if (!sceneRef?.current) return;
    setBoundingBoxes((bs) => snapBoxesToGeometry(bs, sceneRef.current));
  }, [sceneRef]);

  return {
    // state slices
    boundingBoxes,
    setBoundingBoxes,
    defaultSize,
    setDefaultSize,
    defaultCoarseness,
    setDefaultCoarseness,
    showCanvasSliders,
    setShowCanvasSliders,
    showBoxes,
    setShowBoxes,
    boxMode,
    setBoxMode,
    selectedBoxIndex,
    setSelectedBoxIndex,
    hoverFace,
    setHoverFace,
    activeFace,
    setActiveFace,
    errors,
    setErrors,

    handleAddBox,
    updateBoxPosition,
    updateBoxSize,
    updateBoxCoarsenessValue,
    updateDefaultSizeValue,
    updateDefaultCoarsenessValue,
    handleDeleteBox,
    handleClear,
    handleSnapBoxes,
  };
}
