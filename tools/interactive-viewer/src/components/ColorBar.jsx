// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// ColorBar.js
// -----------------------------------------------------------------------------
// React component that renders a vertical color gradient legend with labeled ticks.
//
// Features:
// • Configurable size, tick count, and color stops
// • Generates a linear gradient using CSS
// • Displays numeric tick labels alongside the bar (in mm units)
// -----------------------------------------------------------------------------

import React from "react";
function ColorBar({
  min,
  max,
  width = 30,
  height = 200,
  ticks = 5,
  colors = ["blue", "white", "red"], // default: blue-white-red
}) {
  /* ---------- build gradient string ---------- */
  const step = 100 / (colors.length - 1);
  const stops = colors
    .map((c, i) => `${c} ${Math.round(i * step)}%`)
    .join(", ");
  const gradient = `linear-gradient(to top, ${stops})`;

  /* ---------- tick positions & labels ---------- */
  const tickValues = Array.from(
    { length: ticks },
    (_, i) => (min + (max - min) * ((ticks - 1 - i) / (ticks - 1))) * 1000,
  );
  const tickPositions = Array.from(
    { length: ticks },
    (_, i) => `${((ticks - 1 - i) / (ticks - 1)) * 100}%`,
  );

  /* ---------- render ---------- */
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        background: "rgba(255,255,255,0.9)",
        padding: 8,
        borderRadius: 4,
      }}
    >
      <div style={{ display: "flex", alignItems: "center" }}>
        <div style={{ position: "relative", width, height }}>
          <div
            style={{
              width: "100%",
              height: "100%",
              background: gradient,
              border: "1px solid #ccc",
              borderRadius: 2,
            }}
          />
          {tickPositions.map((pos, idx) => (
            <div
              key={idx}
              style={{
                position: "absolute",
                left: -4,
                width: 4,
                height: 1,
                background: "#000",
                top: pos,
                transform: "translateY(-50%)",
              }}
            />
          ))}
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            height,
            marginLeft: 8,
          }}
        >
          {tickValues.map((v, i) => (
            <div key={i} style={{ fontSize: 12 }}>
              {v.toFixed(1)} mm
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ColorBar;
