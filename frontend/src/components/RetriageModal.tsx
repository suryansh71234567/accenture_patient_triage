import { useModalA11y } from "../hooks/useModalA11y";
import { AcuityPill, DEPT_LABELS } from "./ui";
import type { OperationalDecision, SimVitals } from "../types";

/**
 * Simplified from the mockup's side-by-side previous-vs-new VITALS
 * comparison: the real API only carries previous_operational_department /
 * previous_nurse_override / previous_override_reason on a retriage event,
 * not a snapshot of the patient's previous vitals. Showing a fabricated
 * "before" vitals column isn't an option, so this shows current vitals once
 * plus the real before/after department + override history instead.
 */
export function RetriageModal({
  patientId,
  vitals,
  acuity,
  decision,
  onAcknowledge,
  onClose,
}: {
  patientId: string;
  vitals: SimVitals;
  acuity: number;
  decision: OperationalDecision;
  onAcknowledge: () => void;
  onClose: () => void;
}) {
  const hadOverride = Boolean(decision.previous_nurse_override);
  const containerRef = useModalA11y(onClose);

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="w-[520px] rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl">
        <p className="text-[15px] font-bold text-[var(--color-ink)]">
          Re-Triage — <span className="font-mono">{patientId}</span>
        </p>
        <p className="mb-4 mt-0.5 text-xs text-[var(--color-ink-faint)]">Condition changed. AI has reassessed.</p>

        <div className="mb-3.5 grid grid-cols-4 gap-2 font-mono text-[11px] text-[var(--color-ink-soft)]">
          {vitals.hr != null && <VitalChip label="HR" value={vitals.hr} />}
          {vitals.spo2 != null && <VitalChip label="SpO₂" value={`${vitals.spo2}%`} />}
          {vitals.sbp != null && vitals.dbp != null && <VitalChip label="BP" value={`${vitals.sbp}/${vitals.dbp}`} />}
          {vitals.rr != null && <VitalChip label="RR" value={vitals.rr} />}
        </div>

        <div className="mb-3.5 rounded-xl border p-3" style={{ borderColor: "var(--color-brand-100)", background: "var(--color-brand-50)" }}>
          <p className="text-[9.5px] font-bold uppercase tracking-wide" style={{ color: "var(--color-brand-700)" }}>New AI Assessment</p>
          <div className="mt-1 flex items-center gap-2">
            <AcuityPill acuity={acuity} />
            <p className="text-[13px] font-bold text-[var(--color-ink)]">
              → {DEPT_LABELS[decision.operational_department] ?? decision.operational_department}
            </p>
          </div>
          {decision.previous_operational_department && (
            <p className="mt-1.5 text-[11px] text-[var(--color-ink-faint)]">
              Previously: {DEPT_LABELS[decision.previous_operational_department] ?? decision.previous_operational_department}
            </p>
          )}
          <p className="mt-1.5 text-[11.5px] text-[var(--color-ink-soft)]">{decision.recommendation_summary}</p>
        </div>

        {hadOverride && (
          <div className="mb-3.5 rounded-xl border p-3" style={{ borderColor: "var(--color-override-50)", background: "var(--color-override-50)" }}>
            <p className="text-[9.5px] font-bold uppercase tracking-wide" style={{ color: "var(--color-override-600)" }}>
              Previous Nurse Decision
            </p>
            <p className="mt-1 text-[12px] font-semibold text-[var(--color-ink)]">
              {DEPT_LABELS[decision.previous_operational_department ?? ""] ?? decision.previous_operational_department}
            </p>
            {decision.previous_override_reason && (
              <p className="mt-0.5 text-[10.5px] italic" style={{ color: "var(--color-override-600)" }}>
                "{decision.previous_override_reason}"
              </p>
            )}
            <p className="mt-2 text-[11px] font-bold" style={{ color: "var(--color-warn-600)" }}>
              RE-TRIAGE REQUIRES REVIEW — previous override not automatically changed.
            </p>
          </div>
        )}

        <div className="flex gap-2">
          <button
            onClick={onClose}
            className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-xs font-semibold text-[var(--color-ink-soft)]"
          >
            Close
          </button>
          <button
            onClick={onAcknowledge}
            className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-xs font-bold text-white"
          >
            Acknowledge &amp; Requeue → {DEPT_LABELS[decision.operational_department] ?? decision.operational_department}
          </button>
        </div>
      </div>
    </div>
  );
}

function VitalChip({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg bg-[var(--color-surface-muted)] px-2 py-1.5 text-center">
      <p className="text-[9px] text-[var(--color-ink-faint)]">{label}</p>
      <p className="font-semibold text-[var(--color-ink)]">{value}</p>
    </div>
  );
}
