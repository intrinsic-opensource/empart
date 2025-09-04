// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// src/components/FileLoader.jsx
// -----------------------------------------------------------------------------
// React component for loading a `.glb` file via a styled file input.
//
// Features:
// • Hidden file input with a custom clickable label
// • Styled container with icon and "Load GLB" text
// • Calls `onFileSelect` callback when a file is chosen
// -----------------------------------------------------------------------------

import React from "react";
import { UploadCloud } from "lucide-react";

const colors = {
  background: "#ffffff",
  surface: "#f8f9fa",
  border: "#dee2e6",
  primary: "#007bff",
  text: "#212529",
};

const styles = {
  container: {
    position: "absolute",
    top: "10px",
    left: "80px",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    background: colors.surface,
    padding: "6px 10px",
    borderRadius: "6px",
    fontFamily: "system-ui, sans-serif",
    fontSize: "12px",
    color: colors.text,
    border: `1px solid ${colors.border}`,
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    cursor: "pointer",
  },
  input: {
    display: "none",
  },
  label: {
    display: "flex",
    alignItems: "center",
    gap: "4px",
    cursor: "pointer",
  },
};

const FileLoader = ({ onFileSelect }) => (
  <div style={styles.container}>
    <label htmlFor="glbFile" style={styles.label}>
      <UploadCloud size={16} color={colors.primary} />
      <span>Load GLB</span>
    </label>
    <input
      id="glbFile"
      type="file"
      accept=".glb"
      style={styles.input}
      onChange={onFileSelect}
    />
  </div>
);

export default FileLoader;
