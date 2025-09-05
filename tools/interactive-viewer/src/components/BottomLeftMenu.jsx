// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// BottomLeftMenu.jsx
// ---------------------------------------------------------------------------
// This component provides a bottom-left menu for the interactive viewer,
// allowing users to manage hulls, bounding boxes, and error ranges.
// -----------------------------------------------------------------------------

import React, { useRef, useEffect, useState, useCallback } from "react";
import { Trash2, Box as BoxIcon, Layers, GitBranch } from "lucide-react";
import SliderWithButtons from "./SliderWithButtons";

/* ─── design tokens ───────────────────────────────────────────── */
const colors = {
  bg: "#ffffff",
  surface: "#f8f9fa",
  border: "#dee2e6",
  primary: "#007bff",
  danger: "#dc3545",
  text: "#212529",
};

const styles = {
  container: {
    position: "absolute",
    bottom: "12px",
    left: "12px",
    width: "300px",
    maxHeight: "500px",
    overflowY: "auto",
    background: colors.bg,
    padding: "8px",
    borderRadius: "6px",
    fontFamily: "system-ui, sans-serif",
    fontSize: "11px",
    color: colors.text,
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  collapsedTab: {
    position: "absolute",
    bottom: "12px",
    left: "12px",
    padding: "4px 8px",
    background: colors.primary,
    color: "#fff",
    cursor: "pointer",
    borderRadius: "4px",
    fontSize: "11px",
  },
  collapseBtn: {
    alignSelf: "flex-end",
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "12px",
    marginBottom: "4px",
  },
  sectionTitle: {
    fontWeight: "600",
    fontSize: "12px",
    marginTop: "6px",
    marginBottom: "4px",
    display: "flex",
    alignItems: "center",
    gap: "4px",
  },
  detailsContent: {
    padding: "4px 8px",
    borderLeft: `2px solid ${colors.primary}`,
    marginTop: "4px",
    fontSize: "11px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  hideButton: {
    padding: "2px 6px",
    fontSize: "11px",
    cursor: "pointer",
    borderRadius: "4px",
    border: `1px solid ${colors.primary}`,
    background: colors.primary,
    color: "#fff",
    marginLeft: "4px",
  },
  detailsWrap: { marginBottom: "8px" },
  sliderRow: {
    display: "flex",
    alignItems: "center",
    gap: "4px",
    marginBottom: "4px",
    fontSize: "11px",
  },
  summary: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "4px 6px",
    borderRadius: "4px",
    cursor: "pointer",
    background: colors.surface,
    border: `1px solid ${colors.border}`,
    fontSize: "11px",
  },
  clearButton: {
    padding: "4px 8px",
    border: `1px solid ${colors.danger}`,
    borderRadius: "4px",
    background: "#fff",
    color: colors.danger,
    cursor: "pointer",
    alignSelf: "flex-start",
    fontSize: "11px",
  },
};
styles.detailsColumn = {
  ...styles.detailsContent,
  flexDirection: "column", // stack children
  alignItems: "stretch", // full-width sliders
  gap: "4px", // a bit of breathing room
};

function formatMetric(v) {
  if (v == null) return "-";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e3) return v.toExponential(2);
  return v.toFixed(4);
}

export default function BottomLeftMenu({
  mergeCost,
  sumVolume,
  unionVolume,
  showCanvasSliders,
  setShowCanvasSliders,
  errorMin,
  errorMax,
  setErrorMin,
  setErrorMax,
  defaultSize,
  updateDefaultSizeValue,
  defaultCoarseness,
  updateDefaultCoarsenessValue,
  boundingBoxes,
  updateBoxSize,
  updateBoxPosition,
  updateBoxCoarsenessValue,
  handleDeleteBox,
  handleClear,
  selectedBoxIndex,
  setSelectedBoxIndex,
  meshDimensions,
  selectedHulls,
  onDeleteHull,
  onMergeHulls,
  hiddenHulls,
  setHiddenHulls,
  allHulls,
  setAllHulls,
  setUnionGeom,
  setSumVolume,
  setUnionVolume,
  setCost,
}) {
  const containerRef = useRef(null);
  const detailsRefs = useRef([]);
  const [collapsed, setCollapsed] = useState(false);

  const visibleHulls = selectedHulls.filter((h) => !hiddenHulls.has(h.meshKey));
  const hiddenSelected = allHulls.filter((h) => hiddenHulls.has(h.meshKey));

  const rememberHull = useCallback(
    (hull) =>
      setAllHulls((all) =>
        all.some((x) => x.meshKey === hull.meshKey) ? all : [...all, hull],
      ),
    [setAllHulls],
  );

  const hideSelectedHulls = useCallback(() => {
    visibleHulls.forEach(rememberHull);
    setHiddenHulls((prev) => {
      const next = new Set(prev);
      visibleHulls.forEach((h) => next.add(h.meshKey));
      return next;
    });
    setUnionGeom(null);
    setSumVolume(0);
    setUnionVolume(0);
    setCost(0);
  }, [
    visibleHulls,
    rememberHull,
    setHiddenHulls,
    setUnionGeom,
    setSumVolume,
    setUnionVolume,
    setCost,
  ]);

  useEffect(() => {
    if (collapsed) return;
    const el = detailsRefs.current[selectedBoxIndex];
    if (el && containerRef.current) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedBoxIndex, collapsed]);

  if (collapsed) {
    return (
      <button
        data-testid="menu-open-btn"
        style={styles.collapsedTab}
        onClick={() => setCollapsed(false)}
      >
        Show Menu
      </button>
    );
  }

  return (
    <div style={styles.container} ref={containerRef}>
      <button
        data-testid="menu-collapse-btn"
        style={styles.collapseBtn}
        onClick={() => setCollapsed(true)}
        title="Collapse Menu"
      >
        ✕
      </button>

      {/* Selected Hulls */}
      {visibleHulls.length > 0 && (
        <>
          <div style={styles.sectionTitle}>
            <Layers size={14} />
            <span>Selected Hulls ({visibleHulls.length})</span>
          </div>

          {visibleHulls.length > 1 && (
            <>
              {mergeCost != null && (
                <div
                  style={{
                    fontSize: "12px",
                    marginBottom: "6px",
                    color: colors.text,
                    padding: "4px 6px",
                    background: colors.surface,
                    border: `1px solid ${colors.border}`,
                    borderRadius: "4px",
                  }}
                >
                  <strong>Merge Cost:</strong> {formatMetric(mergeCost)}
                </div>
              )}
              <div style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
                <button
                  data-testid="merge-hulls-btn"
                  onClick={onMergeHulls}
                  style={{
                    padding: "6px 12px",
                    fontSize: "12px",
                    fontWeight: 600,
                    borderRadius: "4px",
                    background: colors.primary,
                    color: "#fff",
                    cursor: "pointer",
                    lineHeight: 1.2,
                  }}
                >
                  Merge Hulls
                </button>

                <button
                  data-testid="hide-hulls-btn"
                  onClick={hideSelectedHulls}
                  style={{
                    padding: "6px 12px",
                    fontSize: "12px",
                    fontWeight: 600,
                    border: "1px solid #ff9800",
                    borderRadius: "4px",
                    background: "#fff",
                    color: "#ff9800",
                    cursor: "pointer",
                    lineHeight: 1.2,
                  }}
                >
                  Hide Hulls
                </button>
              </div>
            </>
          )}

          {visibleHulls.map((h, idx) => (
            <div
              key={h.meshKey}
              style={styles.detailsContent}
              ref={(el) => (detailsRefs.current[idx] = el)}
            >
              <div style={{ lineHeight: 1.35 }}>
                <div>
                  Hull {h.idx + 1} <em>({h.group})</em>
                </div>
                <div>Verts: {h.stats.vertices.toLocaleString()}</div>
                <div>Faces: {h.stats.faces.toLocaleString()}</div>
                <div>Vol: {formatMetric(h.stats.volume)}</div>
              </div>
              <div style={{ display: "flex", alignItems: "center" }}>
                <Trash2
                  data-testid={`delete-hull-btn-${h.meshKey}`}
                  size={16}
                  color={colors.danger}
                  onClick={() => onDeleteHull(h.group, h.idx, h.meshKey)}
                />
                <button
                  data-testid={`hide-hull-btn-${h.meshKey}`}
                  style={styles.hideButton}
                  onClick={() => {
                    rememberHull(h);
                    setHiddenHulls((prev) => {
                      const next = new Set(prev);
                      next.add(h.meshKey);
                      return next;
                    });
                  }}
                >
                  Hide
                </button>
              </div>
            </div>
          ))}
        </>
      )}

      {/* Hidden Hulls */}
      <div>
        <div style={styles.sectionTitle}>
          <Layers size={14} />
          <span>Hidden Hulls ({hiddenSelected.length})</span>
        </div>
        {hiddenSelected.length === 0 ? (
          <div style={styles.detailsContent}>
            <em>No hidden hulls</em>
          </div>
        ) : (
          hiddenSelected.map((h) => (
            <div key={h.meshKey} style={styles.detailsContent}>
              <div style={{ lineHeight: 1.35 }}>
                <div>
                  Hull {h.idx + 1} <em>({h.group})</em>
                </div>
              </div>
              <button
                data-testid={`show-hull-btn-${h.meshKey}`}
                style={styles.hideButton}
                onClick={() =>
                  setHiddenHulls((prev) => {
                    const next = new Set(prev);
                    next.delete(h.meshKey);
                    return next;
                  })
                }
              >
                Show
              </button>
            </div>
          ))
        )}
      </div>

      {/* Error Range Slider */}
      <details style={styles.detailsWrap}>
        <summary data-testid="error-range-summary">Error Range</summary>
        {["Min", "Max"].map((label, i) => (
          <div key={label} style={styles.sliderRow}>
            <span style={{ width: "30px" }}>{label}:</span>
            <SliderWithButtons
              testid={i === 0 ? "error-min-slider" : "error-max-slider"}
              value={i === 0 ? errorMin : errorMax}
              min={i === 0 ? 0 : errorMin + 0.001}
              max={i === 0 ? errorMax - 0.001 : 0.1}
              step={0.001}
              onChange={i === 0 ? setErrorMin : setErrorMax}
            />
            <span>{(i === 0 ? errorMin : errorMax).toFixed(3)}</span>
          </div>
        ))}
      </details>

      {/* Default Box Size */}
      <details style={styles.detailsWrap}>
        <summary testid="default-box-size-summary">Default Box Size</summary>
        {["x", "y", "z"].map((axis) => {
          const dim = meshDimensions[axis] || 1;
          const step = dim * 0.01;
          return (
            <div key={axis} style={styles.sliderRow}>
              <BoxIcon size={12} />
              <span style={{ width: "20px" }}>{axis.toUpperCase()}:</span>
              <SliderWithButtons
                testid={`default-size-slider-${axis}`}
                value={defaultSize[axis]}
                min={step}
                max={dim}
                step={step}
                onChange={(v) => updateDefaultSizeValue(axis, v)}
              />
              <span>{defaultSize[axis].toFixed(3)}</span>
            </div>
          );
        })}
      </details>

      {/* Default Coarseness */}
      <div>
        <div style={styles.sectionTitle}># Hulls (Non-Selected)</div>
        <div style={styles.sliderRow}>
          <Layers size={12} />
          <SliderWithButtons
            testid="default-coarseness-slider"
            value={defaultCoarseness}
            min={0}
            max={10000}
            step={1}
            onChange={updateDefaultCoarsenessValue}
          />
          <span>{defaultCoarseness.toFixed(0)}</span>
        </div>
      </div>

      {/* Individual Boxes */}
      <div>
        <div style={styles.sectionTitle}>Boxes</div>
        {boundingBoxes.map((b, i) => {
          const open = selectedBoxIndex === i;
          return (
            <details
              key={i}
              ref={(el) => (detailsRefs.current[i] = el)}
              open={open}
              onClick={() => setSelectedBoxIndex(i)}
              style={{ marginBottom: "6px" }}
            >
              <summary data-testid={`box-summary-${i}`} style={styles.summary}>
                <span>Box {i + 1}</span>
                <Trash2
                  data-testid={`delete-box-btn-${i}`}
                  size={14}
                  color={colors.danger}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteBox(i);
                  }}
                />
              </summary>

              {/* Size */}
              <div style={styles.detailsColumn}>
                <div style={styles.sectionTitle}>Size</div>
                {["x", "y", "z"].map((ax) => {
                  const dim = meshDimensions[ax] || 1;
                  const step = dim * 0.005;
                  return (
                    <div key={ax} style={styles.sliderRow}>
                      <span style={{ width: "20px" }}>{ax.toUpperCase()}:</span>
                      <SliderWithButtons
                        testid={`box-${i}-size-slider-${ax}`}
                        value={b.size[ax]}
                        // min={step}
                        // max={dim}
                        min={0}
                        max={1}
                        step={step}
                        onChange={(v) => updateBoxSize(i, ax, v)}
                      />
                      <span>{b.size[ax].toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>

              {/* Position */}
              <div style={styles.detailsColumn}>
                <div style={styles.sectionTitle}>Position</div>
                {["x", "y", "z"].map((ax) => {
                  const dim = meshDimensions[ax] || 1;
                  const step = dim * 0.005;
                  const min = -dim / 2;
                  const max = dim / 2;
                  return (
                    <div key={ax} style={styles.sliderRow}>
                      <span style={{ width: "20px" }}>{ax.toUpperCase()}:</span>
                      <SliderWithButtons
                        testid={`box-${i}-pos-slider-${ax}`}
                        value={b.center[ax]}
                        // min={min}
                        // max={max}
                        min={-1}
                        max={1}
                        step={step}
                        onChange={(v) => updateBoxPosition(i, ax, v)}
                      />
                      <span>{b.center[ax].toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>

              {/* Hull Count */}
              <div style={styles.detailsColumn}>
                <div style={styles.sectionTitle}># Hulls</div>
                <div style={styles.sliderRow}>
                  <GitBranch size={12} />
                  <SliderWithButtons
                    testid={`box-${i}-coarseness-slider`}
                    value={b.coarseness}
                    min={0}
                    max={128}
                    step={1}
                    onChange={(v) => updateBoxCoarsenessValue(i, v)}
                  />
                  <span>{b.coarseness.toFixed(0)}</span>
                </div>
              </div>
            </details>
          );
        })}
      </div>

      <button
        data-testid="clear-all-btn"
        style={styles.clearButton}
        onClick={handleClear}
      >
        Clear All
      </button>
    </div>
  );
}
