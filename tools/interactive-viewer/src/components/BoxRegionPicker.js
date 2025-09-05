// Copyright 2025 Intrinsic Innovation LLC

// components/BoxRegionPicker.js
// ---------------------------------------------------------------------------
// BoxRegionPicker.js
// ---------------------------------------------------------------------------
// Helper component for selecting what regions to include for error analysis
// in the interactive viewer.

import React, { useState, useRef, useEffect } from "react";

const dropdownStyles = {
  wrapper: { position: "relative", width: "100%" },
  trigger: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "6px 10px",
    border: "1px solid #dee2e6",
    borderRadius: "6px",
    background: "#f8f9fa",
    fontSize: 12,
    cursor: "pointer",
    width: "100%",
  },
  panel: {
    position: "absolute",
    top: "100%",
    left: 0,
    zIndex: 9999,
    width: "100%",
    background: "#fff",
    border: "1px solid #dee2e6",
    borderRadius: "6px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    maxHeight: 200,
    overflowY: "auto",
    padding: "8px",
  },
  item: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 12,
    marginBottom: 4,
  },
  footer: {
    marginTop: 6,
    display: "flex",
    justifyContent: "space-between",
    fontSize: 11,
  },
};

export default function BoxRegionPicker({
  boxCount,
  selected,
  onChange,
  getCountForBox,
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const onClick = (e) => {
      if (!ref.current?.contains(e.target)) setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, []);

  const toggleBox = (idx) => {
    const next = new Set(selected);
    next.has(idx) ? next.delete(idx) : next.add(idx);
    onChange(next);
  };

  const allSelected = selected.size === boxCount;
  const noneSelected = selected.size === 0;

  return (
    <div style={dropdownStyles.wrapper} ref={ref}>
      <button
        style={dropdownStyles.trigger}
        onClick={() => setOpen((o) => !o)}
        data-testid="box-filter-trigger"
      >
        <span>
          Box Filter ({selected.size}/{boxCount})
        </span>
        <span>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div style={dropdownStyles.panel}>
          {Array.from({ length: boxCount }).map((_, i) => (
            <label key={i} style={dropdownStyles.item}>
              <input
                type="checkbox"
                checked={selected.has(i)}
                onChange={() => toggleBox(i)}
              />
              Box {i + 1}{" "}
              <span style={{ opacity: 0.6 }}>({getCountForBox(i)} hulls)</span>
            </label>
          ))}

          <div style={dropdownStyles.footer}>
            <button onClick={() => onChange(new Set())}>None</button>
            <button
              onClick={() => onChange(new Set([...Array(boxCount).keys()]))}
            >
              All
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
