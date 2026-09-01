import { useState } from "react";
import { useModalA11y } from "../hooks/useModalA11y";
import type { SimVitals } from "../types";

const VITAL_FIELDS: { key: keyof SimVitals; label: string }[] = [
  { key: "hr", label: "Heart Rate" },
  { key: "rr", label: "Resp Rate" },
  { key: "spo2", label: "SpO₂" },
  { key: "sbp", label: "Systolic BP" },
  { key: "dbp", label: "Diastolic BP" },
  { key: "temp", label: "Temp" },
  { key: "pain", label: "Pain" },
];

export function VitalsEditModal({
  patientId,
  busy,
  onSave,
  onClose,
}: {
  patientId: string;
  busy: boolean;
  onSave: (vitals: SimVitals) => void;
  onClose: () => void;
}) {
  const [form, setForm] = useState<Record<string, string>>({});
  // Cancel is disabled while busy; Escape mirrors that same gating.
  const containerRef = useModalA11y(busy ? () => {} : onClose);

  const save = () => {
    const vitals: SimVitals = {};
    for (const f of VITAL_FIELDS) {
      const raw = form[f.key];
      if (raw !== undefined && raw.trim() !== "") (vitals as Record<string, number>)[f.key] = Number(raw);
    }
    onSave(vitals);
  };

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="w-[420px] rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl">
        <p className="text-sm font-bold text-[var(--color-ink)]">
          Update Vitals — <span className="font-mono">{patientId}</span>
        </p>
        <p className="mb-4 mt-1 text-[11px] text-[var(--color-ink-faint)]">Condition changes trigger automatic re-triage.</p>
        <div className="grid grid-cols-2 gap-2.5">
          {VITAL_FIELDS.map((f) => (
            <label key={f.key} className="flex flex-col gap-1">
              <span className="text-[9.5px] font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">{f.label}</span>
              <input
                type="number"
                value={form[f.key] ?? ""}
                onChange={(e) => setForm((s) => ({ ...s, [f.key]: e.target.value }))}
                className="rounded-lg border border-[var(--color-border)] px-2.5 py-1.5 font-mono text-[13px] outline-none focus:border-[var(--color-brand-500)]"
              />
            </label>
          ))}
        </div>
        <div className="mt-4 flex gap-2">
          <button
            onClick={onClose}
            disabled={busy}
            className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-xs font-semibold text-[var(--color-ink-soft)] disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={save}
            disabled={busy}
            className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-xs font-bold text-white hover:bg-[var(--color-brand-600)] disabled:opacity-50"
          >
            {busy ? "Saving…" : "Save & Re-triage"}
          </button>
        </div>
      </div>
    </div>
  );
}
