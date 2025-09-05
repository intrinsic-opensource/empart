// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// file.js
// ---------------------------------------------------------------------------
// Converts blob or remote URLs to Base64, with helpers for decoding and downloading.

import { Buffer } from "buffer";

/**
 * Fetches a blob URL (or remote URL) and returns Base‑64 body.
 */
export async function blobUrlToBase64(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed fetch ${url}: ${res.statusText}`);

  if (typeof window !== "undefined" && typeof FileReader !== "undefined") {
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onerror = () => reject(r.error ?? new Error("FileReader failed"));
      r.onload = () => {
        const s = /** @type {string} */ (r.result);
        resolve(s.slice(s.indexOf(",") + 1));
      };
      r.readAsDataURL(blob);
    });
  }

  const buf = await res.arrayBuffer();
  return Buffer.from(buf).toString("base64");
}

export function decodeBlobUrl(base64, mime = "application/octet-stream") {
  const bin = atob(base64);
  const u8 = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  const blob = new Blob([u8], { type: mime });
  return URL.createObjectURL(blob);
}

export function downloadBlobWithPrompt(blob, fallback = "export.zip") {
  const name = window.prompt("Save file as…", fallback);
  if (!name) return;
  const href = URL.createObjectURL(blob);
  Object.assign(document.createElement("a"), {
    href,
    download: name.trim(),
  }).click();
  URL.revokeObjectURL(href);
}
