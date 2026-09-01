import { useState } from "react";
import { useModalA11y } from "../hooks/useModalA11y";
import { DEPT_LABELS } from "./ui";

const QUICK_REASONS = ["Capacity constraint", "Clinical judgment", "Family/social factors", "Other"];

export function OverrideReasonModal({
  patientId,
  from,
  to,
  busy,
  onConfirm,
  onCancel,
}: {
  patientId: string;
  from: string;
  to: string;
  busy: boolean;
  onConfirm: (reason: string) => void;
  onCancel: () => void;
}) {
  const [reason, setReason] = useState("");
  // Cancel is disabled while busy; Escape mirrors that same gating.
  const containerRef = useModalA11y(busy ? () => {} : onCancel);

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-[70] flex items-center justify-center bg-black/40 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="w-[420px] rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl">
        <p className="mb-3 text-sm font-bold text-[var(--color-ink)]">
          Move Patient — Nurse Override <span className="font-mono font-normal text-[var(--color-ink-faint)]">({patientId})</span>
        </p>
        <div className="mb-3.5 flex items-center gap-2.5">
          <div className="flex-1 rounded-lg bg-[var(--color-surface-muted)] p-2.5 text-center">
            <p className="text-[9px] text-[var(--color-ink-faint)]">AI RECOMMENDED</p>
            <p className="mt-0.5 text-[13px] font-bold text-[var(--color-ink)]">{DEPT_LABELS[from] ?? from}</p>
          </div>
          <span className="text-[var(--color-ink-faint)]">→</span>
          <div className="flex-1 rounded-lg p-2.5 text-center" style={{ background: "var(--color-override-50)" }}>
            <p className="text-[9px] font-semibold" style={{ color: "var(--color-override-600)" }}>NURSE MOVING TO</p>
            <p className="mt-0.5 text-[13px] font-bold" style={{ color: "var(--color-override-600)" }}>{DEPT_LABELS[to] ?? to}</p>
          </div>
        </div>
        <div className="mb-2 flex flex-wrap gap-1.5">
          {QUICK_REASONS.map((qr) => (
            <button
              key={qr}
              onClick={() => setReason(qr)}
              className="rounded-full border border-[var(--color-border)] px-2.5 py-1 text-[11px] font-medium transition"
              style={{ background: reason === qr ? "var(--color-brand-50)" : "transparent" }}
            >
              {qr}
            </button>
          ))}
        </div>
        <textarea
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Reason for override…"
          className="mb-3.5 h-14 w-full resize-none rounded-lg border border-[var(--color-border)] px-3 py-2 text-xs outline-none focus:border-[var(--color-brand-500)]"
        />
        <div className="flex gap-2">
          <button
            onClick={onCancel}
            disabled={busy}
            className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-xs font-semibold text-[var(--color-ink-soft)] disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(reason)}
            disabled={busy}
            className="flex-1 rounded-lg px-4 py-2 text-xs font-bold text-white disabled:opacity-50"
            style={{ background: "var(--color-override-500)" }}
          >
            {busy ? "Applying…" : "Confirm Move"}
          </button>
        </div>
      </div>
    </div>
  );
}
