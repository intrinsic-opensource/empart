// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// components/SliderWithButtons.jsx
// -----------------------------------------------------------------------------
// Reusable slider component with increment and decrement buttons.
//
// Features:
// • Accepts min, max, step, and value props
// • Calls `onChange` with updated value on slider move or button press
// -----------------------------------------------------------------------------

import React from "react";

const SliderWithButtons = ({
  value,
  min,
  max,
  step,
  onChange,
  testid, // now accepted
}) => {
  const base = testid || "slider";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
      <button
        data-testid={`${base}-decrement`}
        onClick={() => onChange(Math.max(min, value - step))}
        style={{ padding: "4px 8px", cursor: "pointer" }}
      >
        –
      </button>

      <input
        data-testid={`${base}-slider`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flexGrow: 1, cursor: "pointer" }}
      />

      <button
        data-testid={`${base}-increment`}
        onClick={() => onChange(Math.min(max, value + step))}
        style={{ padding: "4px 8px", cursor: "pointer" }}
      >
        +
      </button>
    </div>
  );
};

export default SliderWithButtons;
