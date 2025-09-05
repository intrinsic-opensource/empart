// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// TopRightControls.jsx
// ---------------------------------------------------------------------------
// This component provides a set of controls for interacting with the 3D viewer,
// including options for processing, error evaluation, mesh preprocessing,
// visibility toggles, and import/export functionality.

import React from "react";
import {
  Eye,
  EyeOff,
  Grid,
  Database,
  Slash,
  UploadCloud,
  ServerCog,
  Download,
  Scissors,
} from "lucide-react";
import BoxRegionPicker from "./BoxRegionPicker";

// ---------------------------------------------------------------------------
// Colour & style constants
// ---------------------------------------------------------------------------
const colors = {
  background: "#ffffff",
  surface: "#f8f9fa",
  border: "#dee2e6",
  primary: "#007bff",
  primaryLight: "#e7f1ff",
  secondary: "#6c757d",
  success: "#28a745",
  danger: "#dc3545",
  warning: "#ffc107",
  text: "#212529",
};

const styles = {
  container: {
    position: "absolute",
    top: "16px",
    right: "16px",
    zIndex: 1000,
    background: colors.background,
    padding: "12px",
    borderRadius: "8px",
    fontFamily: "system-ui, sans-serif",
    fontSize: "13px",
    color: colors.text,
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    width: "240px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    overflow: "visible",
  },
  button: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "6px 10px",
    border: `1px solid ${colors.border}`,
    borderRadius: "6px",
    cursor: "pointer",
    background: colors.surface,
    transition: "background 0.2s, border-color 0.2s",
    fontSize: "12px",
    color: colors.text,
  },
  buttonActive: {
    background: colors.primaryLight,
    borderColor: colors.primary,
    color: colors.primary,
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  spinner: {
    border: "3px solid rgba(0, 123, 255, 0.2)",
    borderTop: "3px solid #007bff",
    borderRadius: "50%",
    width: "16px",
    height: "16px",
    animation: "spin 0.8s linear infinite",
    position: "absolute",
    top: "8px",
    left: "8px",
  },
  group: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "6px",
  },
  decimateRow: {
    display: "grid",
    gridTemplateColumns: "1fr auto",
    gap: "6px",
  },
  hiddenInput: { display: "none" },
  section: { display: "flex", flexDirection: "column", gap: "6px" },
  sectionLabel: { fontSize: 12, fontWeight: 600, color: colors.secondary },
  checkboxGroup: { display: "flex", flexDirection: "column", gap: "4px" },
};

// Helper to sanitise a label into a test‑ID
const makeTestId = (label) =>
  "btn-" +
  label
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w-]/g, "");

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
const TopRightControls = ({
  errorBoxFilter,
  setErrorBoxFilter,
  boxToHullMap,
  showBoxes,
  setShowBoxes,
  wireframeMode,
  setWireframeMode,
  boxMode,
  setBoxMode,
  showErrorForward,
  setShowErrorForward,
  showErrorReverse,
  setShowErrorReverse,
  handleSend,
  handleProcessError,
  meshes,
  boundingBoxes,
  handleSnapBoxes,
  handleExport,
  handlePerfCheck,
  handleExportState,
  handleImportState,
  isLoading,
  showOriginal,
  setShowOriginal,
  showApproximation,
  setShowApproximation,
  handleDecimate,
  handleWatertight,
  originalUrl,
  watertight,
  algorithm,
  setAlgorithm,
  setIsEditing,
}) => {
  const [decimateTris, setDecimateTris] = React.useState(10000);
  const [watertightPitch, setWatertightPitch] = React.useState(0.002);

  const renderButton = (
    icon,
    label,
    active,
    onClick,
    disabled,
    fullWidth = false,
  ) => {
    const base = { ...styles.button, ...(fullWidth && { width: "100%" }) };
    if (active) Object.assign(base, styles.buttonActive);
    if (disabled) Object.assign(base, styles.buttonDisabled);
    return (
      <button
        key={label}
        data-testid={makeTestId(label)}
        style={base}
        onClick={onClick}
        disabled={disabled}
      >
        {icon}
        <span>{label}</span>
      </button>
    );
  };

  // Sync editing mode with visibility toggles
  React.useEffect(() => {
    if (
      showOriginal &&
      !showApproximation &&
      !showErrorForward &&
      !showErrorReverse
    ) {
      setIsEditing?.(true);
    } else {
      setIsEditing?.(false);
    }
  }, [
    showOriginal,
    showApproximation,
    setIsEditing,
    showErrorForward,
    showErrorReverse,
  ]);

  return (
    <>
      {/* Local keyframe for tiny spinner */}
      <style>
        {`@keyframes spin { to { transform: rotate(360deg); } }
          button:focus { outline: none; }`}
      </style>

      <div style={styles.container}>
        {/* ----------------------------------------------------------------- */}
        {/*  Watertight badge & algorithm selector                           */}
        {/* ----------------------------------------------------------------- */}
        <div
          style={{
            alignSelf: "flex-end",
            fontSize: 11,
            padding: "2px 6px",
            borderRadius: 4,
            background:
              watertight == null
                ? colors.warning
                : watertight
                  ? colors.success
                  : colors.danger,
            color: "#fff",
          }}
        >
          {watertight == null ? "—" : watertight ? "Watertight" : "Open"}
        </div>

        {/* ----------------------------------------------------------------- */}
        {/*  Processing                                                       */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Processing</span>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <label htmlFor="algorithm-select" style={{ fontSize: 12 }}>
              Algo:
            </label>
            <select
              id="algorithm-select"
              data-testid="algorithm-select"
              value={algorithm}
              onChange={(e) => setAlgorithm?.(e.target.value)}
              style={{
                flexGrow: 1,
                padding: "6px 8px",
                fontSize: 12,
                border: `1px solid ${colors.border}`,
                borderRadius: 6,
                background: colors.surface,
                cursor: "pointer",
              }}
            >
              <option value="vhacd">VHACD</option>
              <option value="coacd">CoACD</option>
            </select>
          </div>
          {renderButton(
            <Database size={14} />,
            "Run",
            false,
            handleSend,
            false,
            true,
          )}
        </div>

        {/* ----------------------------------------------------------------- */}
        {/*  Error evaluation                                                 */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Error Evaluation</span>

          <BoxRegionPicker
            boxCount={boundingBoxes.length}
            selected={errorBoxFilter}
            onChange={setErrorBoxFilter}
            getCountForBox={(i) =>
              (boxToHullMap?.nonSelect?.[i]?.length || 0) +
              (boxToHullMap?.select?.[i]?.length || 0)
            }
          />

          <div style={styles.group}>
            {renderButton(
              <Slash size={14} />,
              "Run Error (on true)",
              false,
              () => handleProcessError(true),
              !meshes?.nonSelect,
            )}
            {renderButton(
              <Slash size={14} />,
              "Run Error (on approx)",
              false,
              () => handleProcessError(false),
              !meshes?.nonSelect,
            )}
          </div>
        </div>

        {/* ----------------------------------------------------------------- */}
        {/*  Performance                                                      */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Performance</span>
          {renderButton(
            <ServerCog size={14} />,
            "Compute RTF",
            false,
            handlePerfCheck,
            !meshes?.nonSelect,
            true,
          )}
        </div>

        {/* ----------------------------------------------------------------- */}
        {/*  Mesh Preprocessing                                                */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Mesh Preprocessing</span>

          <div style={styles.decimateRow}>
            <input
              data-testid="decimate-input"
              type="number"
              min="100"
              step="100"
              value={decimateTris}
              onChange={(e) => setDecimateTris(Number(e.target.value))}
              style={{
                padding: "6px 8px",
                fontSize: 12,
                border: `1px solid ${colors.border}`,
                borderRadius: 6,
                width: "100%",
              }}
              title="Target triangle count"
            />
            {renderButton(
              <Scissors size={14} />,
              "Decimate",
              false,
              () => handleDecimate(decimateTris),
              !originalUrl,
            )}
          </div>

          <div style={styles.decimateRow}>
            <input
              data-testid="voxelize-input"
              type="number"
              step="0.0005"
              min="0.0001"
              value={watertightPitch}
              onChange={(e) => setWatertightPitch(Number(e.target.value))}
              style={{
                padding: "6px 8px",
                fontSize: 12,
                border: `1px solid ${colors.border}`,
                borderRadius: 6,
                width: "100%",
              }}
              title="Voxel size for marching‑cubes"
            />
            {renderButton(
              <Grid size={14} />,
              "Voxelize",
              false,
              () => handleWatertight(watertightPitch),
              !originalUrl,
            )}
          </div>
        </div>

        {isLoading && <div style={styles.spinner} />}

        {/* ----------------------------------------------------------------- */}
        {/*  Display & boxes                                                  */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Utils</span>

          <div style={styles.group}>
            {renderButton(
              showBoxes ? <Eye size={14} /> : <EyeOff size={14} />,
              showBoxes ? "Hide Boxes" : "Show Boxes",
              showBoxes,
              () => setShowBoxes((b) => !b),
            )}
            {renderButton(
              <Grid size={14} />,
              wireframeMode ? "Wireframe Off" : "Wireframe On",
              wireframeMode,
              () => setWireframeMode((w) => !w),
            )}
            {renderButton(
              <Grid size={14} />,
              boxMode === "corner" ? "Corner → Center" : "Center → Corner",
              boxMode === "center",
              () => setBoxMode((m) => (m === "corner" ? "center" : "corner")),
            )}
            {renderButton(
              <Slash size={14} />,
              "Snap",
              false,
              handleSnapBoxes,
              !boundingBoxes?.length,
            )}
          </div>
        </div>
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Export </span>
          {renderButton(
            <UploadCloud size={14} />,
            "Export Parts",
            false,
            handleExport,
            !meshes?.nonSelect,
            true,
          )}
        </div>
        {/* ----------------------------------------------------------------- */}
        {/*  Visibility toggles                                               */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Visibility</span>

          <div style={styles.checkboxGroup}>
            {[
              ["Original", showOriginal, setShowOriginal],
              ["Approximation", showApproximation, setShowApproximation],
              ["Error (On True)", showErrorForward, setShowErrorForward],
              ["Error (On Approx)", showErrorReverse, setShowErrorReverse],
            ].map(([label, checked, toggle]) => (
              <label
                key={label}
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggle((v) => !v)}
                />
                {label}
              </label>
            ))}
          </div>
        </div>

        {/* ----------------------------------------------------------------- */}
        {/*  Import/Export                                                            */}
        {/* ----------------------------------------------------------------- */}
        <div style={styles.section}>
          <span style={styles.sectionLabel}>Import/Export</span>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "6px",
            }}
          >
            <label
              htmlFor="stateFile"
              data-testid="load-state-btn"
              style={styles.button}
            >
              <UploadCloud size={14} color={colors.primary} />
              <span>Import Project</span>
            </label>
            <input
              id="stateFile"
              data-testid="state-file-input"
              type="file"
              accept=".gltf,.json,application/json"
              style={styles.hiddenInput}
              onChange={handleImportState}
            />

            {renderButton(
              <Download size={14} />,
              "Export project",
              false,
              handleExportState,
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default TopRightControls;
