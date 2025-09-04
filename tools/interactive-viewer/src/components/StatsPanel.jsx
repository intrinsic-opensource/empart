// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// StatsPanel.jsx
// -----------------------------------------------------------------------------
// Displays a floating panel with mesh stats and timing info.
// -----------------------------------------------------------------------------

import React from "react";
import { useEffect, useRef } from "react";

/**
 * Floating stats panel.
 * Pass an object like { vertices, faces, hulls }.
 */
export default function StatsPanel({ stats = {}, timings = {} }) {
  const fmt = (v) => (v != null ? v.toLocaleString() : "–");
  const fmtTime = (ms) => (ms != null ? `${(ms / 1000).toFixed(2)} s` : "–");

  return (
    <div
      data-testid="stats-panel"
      key={timings.process}
      style={{
        position: "absolute",
        bottom: 20,
        right: 120,
        zIndex: 10,
        background: "rgba(0,0,0,0.65)",
        color: "#fff",
        padding: "8px 12px",
        borderRadius: 6,
        fontFamily: "monospace",
        fontSize: 12,
        lineHeight: 1.4,
        pointerEvents: "none",
        whiteSpace: "nowrap",
      }}
    >
      <div>Vertices: {fmt(stats.vertices)}</div>
      <div>Faces: &nbsp;&nbsp;{fmt(stats.faces)}</div>
      <div>Hulls: &nbsp;&nbsp;{fmt(stats.hulls)}</div>
      <div>RT Factor: &nbsp;&nbsp;{fmt(stats.rt_factor)}</div>
      {timings.process !== null && (
        <>
          <hr style={{ opacity: 0.25 }} />
          <div>process:&nbsp;&nbsp;{fmtTime(timings.process)}</div>
          <div>run-error:&nbsp;{fmtTime(timings.runError)}</div>
          <div>perf:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{fmtTime(timings.perf)}</div>
        </>
      )}
    </div>
  );
}
