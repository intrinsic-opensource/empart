// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------
// useKeyboardShortcuts.js
// -----------------------------------------------------------
// React hook that registers global keyboard shortcuts for hull and box editing.
//
// Features:
// • Escape - Clears hull and box selections
// • M      - Merges selected hulls
// • Backspace - Deletes selected hulls or active bounding box
// • H      - Hides selected hulls and updates side panel metadata

import { useEffect, useRef } from "react";

/**
 * Handles the global hot‑keys for hull / box editing.
 *
 * @param {Object}  p
 * @param {Array}   p.selectedHulls           – array of currently‑selected hull meta objects
 * @param {Function}p.clearHullSelections     – () ⇒ void
 * @param {Function}p.deleteHull              – (group, idx, meshKey) ⇒ void
 * @param {number?} p.selectedBoxIndex        – index of the active bounding box or null
 * @param {Function}p.handleDeleteBox         – (boxIdx) ⇒ void
 * @param {Function}p.handleMergeHulls        – () ⇒ void  (merges *all* selected hulls)
 * @param {Function}p.setSelectedBoxIndex     – (null | number) ⇒ void
 * @param {Function}p.setActiveFace           – (face | null) ⇒ void   (exits snap‑mode)
 * @param {Function}p.setHiddenHulls          – (Set) ⇒ void  (add keys to hidden set)
 * @param {Function}p.setAllHulls             – (Array) ⇒ void (metadata used by side panel)
 */
export default function useKeyboardShortcuts({
  selectedHulls,
  clearHullSelections,
  deleteHull,
  selectedBoxIndex,
  handleDeleteBox,
  handleMergeHulls,
  setSelectedBoxIndex,
  setActiveFace,
  setHiddenHulls,
  setAllHulls,
}) {
  // we need a stable ref because the handler lives outside React render
  const selectedHullsRef = useRef(selectedHulls);
  const boxIndexRef = useRef(selectedBoxIndex);

  // keep refs in‑sync with latest props
  useEffect(
    () => void (selectedHullsRef.current = selectedHulls),
    [selectedHulls],
  );
  useEffect(
    () => void (boxIndexRef.current = selectedBoxIndex),
    [selectedBoxIndex],
  );

  useEffect(() => {
    function onKeyDown(e) {
      switch (e.key) {
        // ────────────────────────────────────────────
        // Escape → clear any selection
        // ────────────────────────────────────────────
        case "Escape": {
          clearHullSelections();
          if (boxIndexRef.current != null) {
            setSelectedBoxIndex(null);
            setActiveFace(null);
          }
          break;
        }

        // ────────────────────────────────────────────
        // M → merge currently selected hulls
        // ────────────────────────────────────────────
        case "m":
        case "M": {
          if (selectedHullsRef.current.length > 1) {
            e.preventDefault(); // keep browser focus shortcuts away
            handleMergeHulls();
          }
          break;
        }

        // ────────────────────────────────────────────
        // Backspace → delete selected hull(s) OR the active box
        // ────────────────────────────────────────────
        case "Backspace": {
          // delete hulls first
          selectedHullsRef.current.forEach((h) => {
            deleteHull(h.group, h.idx, h.meshKey);
          });

          // then delete box (if any)
          const idx = boxIndexRef.current;
          if (idx != null) {
            handleDeleteBox(idx);
            setSelectedBoxIndex(null);
          }
          break;
        }

        // ────────────────────────────────────────────
        // H → hide selected hulls (push them to Hidden‑set)
        // ────────────────────────────────────────────
        case "h":
        case "H": {
          const hulls = selectedHullsRef.current;
          if (!hulls.length) return;

          // 1. remember the metadata so the side panel can list them later
          setAllHulls((prev) => {
            const next = [...prev];
            hulls.forEach((h) => {
              if (!next.some((x) => x.meshKey === h.meshKey)) next.push(h);
            });
            return next;
          });

          // 2. add their meshKeys to the Hidden set
          setHiddenHulls((prev) => {
            const next = new Set(prev);
            hulls.forEach((h) => next.add(h.meshKey));
            return next;
          });

          // 3. clear the visual selection afterwards
          clearHullSelections();
          break;
        }
        default:
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    clearHullSelections,
    deleteHull,
    handleDeleteBox,
    handleMergeHulls,
    setAllHulls,
    setHiddenHulls,
    setActiveFace,
    setSelectedBoxIndex,
  ]);
}
