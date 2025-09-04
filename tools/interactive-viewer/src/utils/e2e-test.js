// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// setupE2ELogging.js   (pure JavaScript)
// -----------------------------------------------------------------------------
// This script sets up instrumentation for end-to-end (E2E) testing by
// intercepting console.e2e logs, user inputs (pointer, wheel, key, and click events),
// and saving the logs to be copied to the clipboard.

export const isE2EMode = () =>
  typeof window !== "undefined" && window.location.search.includes("e2e");

export function setupE2ELogging() {
  /* ───────────── capture [E2E] logs ───────────── */
  const e2eLogs = [];
  console.e2e = (...args) => {
    const txt = args.map(String).join(" ");
    if (txt.startsWith("[E2E]")) e2eLogs.push(txt);
    console.log(...args); // still log to console
  };
  if (!isE2EMode()) {
    return;
  }
  const pct = (n) => (n * 100).toFixed(4) + "%";
  const btnName = (b) => ({ 0: "LMB", 1: "MMB", 2: "RMB" })[b] ?? `BTN${b}`;

  /* ───────────── pointer logging ───────────── */
  const logPointer = (ev) => {
    const rect = ev.target.getBoundingClientRect();
    const relX = ev.clientX - rect.left;
    const relY = ev.clientY - rect.top;
    console.e2e(
      `[E2E] ${ev.type.padEnd(11)} ${btnName(ev.button).padEnd(4)} ` +
        `(${relX.toFixed(4)}, ${relY.toFixed(4)}) → ` +
        `${pct(relX / rect.width)}, ${pct(relY / rect.height)}`,
    );
  };
  window.addEventListener("pointerdown", logPointer, true);
  window.addEventListener("pointerup", logPointer, true);

  /* ───────────── wheel / zoom logging ───────────── */
  window.addEventListener(
    "wheel",
    (ev) => {
      const rect = ev.target.getBoundingClientRect();
      const relX = ev.clientX - rect.left;
      const relY = ev.clientY - rect.top;
      const pct = (n) => (n * 100).toFixed(4) + "%";

      const dir = ev.deltaY < 0 ? "zoom-in" : "zoom-out";
      console.e2e(
        `[E2E] wheel        ${dir.padEnd(8)} ` +
          `(${relX.toFixed(4)}, ${relY.toFixed(4)}) → ` +
          `${pct(relX / rect.width)}, ${pct(relY / rect.height)} ` +
          `(deltaY=${ev.deltaY.toFixed(2)})`,
      );
    },
    { passive: true, capture: true },
  );

  document.addEventListener("pointermove", (e) => {
    const pctX = ((e.clientX / window.innerWidth) * 100).toFixed(4);
    const pctY = ((e.clientY / window.innerHeight) * 100).toFixed(4);
    console.e2e(
      `[E2E] pointermove  (${e.clientX}, ${e.clientY}) → ${pctX}%, ${pctY}%`,
    );
  });

  /* ───────────── button clicks ───────────── */
  document.addEventListener(
    "click",
    (ev) => {
      const btnEl = ev.target.closest ? ev.target.closest("button") : null;
      if (!btnEl) return;

      // first try data-testid, then aria-label, then text
      const label =
        btnEl.getAttribute("testid") ||
        btnEl.getAttribute("data-testid") ||
        btnEl.getAttribute("aria-label") ||
        btnEl.textContent.trim().replace(/\s+/g, " ");

      console.e2e(`[E2E] button click → "${label}"`);
    },
    true,
  );

  /* ───────────── key logging ───────────── */
  window.addEventListener(
    "keydown",
    (ev) => {
      if (["Escape", "Backspace", "h", "H", "m", "M"].includes(ev.key)) {
        console.e2e(`[E2E] keydown      "${ev.key}"`);
      }
    },
    true,
  );

  /* ───────────── clipboard UI ───────────── */
  const copyLogs = () => {
    navigator.clipboard.writeText(e2eLogs.join("\n")).then(() => {
      btn.textContent = "Copied ✔";
      setTimeout(() => (btn.textContent = "Copy E2E log"), 1500);
    });
  };

  const btn = document.createElement("button");
  btn.textContent = "Copy E2E log";
  Object.assign(btn.style, {
    position: "fixed",
    bottom: "20px",
    left: "20px",
    zIndex: 10000,
    padding: "6px 10px",
    fontSize: "12px",
    cursor: "pointer",
  });
  btn.setAttribute("translate", "no");
  btn.addEventListener("click", copyLogs);
  document.body.appendChild(btn);

  console.e2e("[E2E] logging hooks installed ✅");
}

setupE2ELogging();
