import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useModalA11y } from "../hooks/useModalA11y";
import { useSession } from "../state/SessionContext";
import { AcuityPill, DEPT_LABELS } from "./ui";
import type { SimVitals, TriageResult } from "../types";

/**
 * The mockup's flow implies "Run Assessment" previews a recommendation
 * before anything is committed. The real triageSimulated() endpoint isn't a
 * preview — calling it already runs the clinical assessment AND commits the
 * operational placement server-side (the patient moves ARRIVED -> TRIAGED
 * immediately). So "Confirm" here just acknowledges the already-committed
 * placement (no second API call); "Override Placement" is the one real
 * follow-up action, calling the same overrideDepartment() endpoint the
 * department board's drag-and-drop uses.
 */
export function TriageModal({
  patientId,
  age,
  sex,
  chiefComplaint,
  vitals,
  hospitalId,
  departmentOptions,
  activation,
  onClose,
  onDone,
}: {
  patientId: string;
  age: number;
  sex: string;
  chiefComplaint: string;
  vitals: SimVitals;
  hospitalId: string;
  departmentOptions: string[];
  /**
   * Present only for a chart-only patient with no live simulation record yet
   * (e.g. opened via the Patients screen drawer or PatientWorkspace, not an
   * already-ARRIVED queue patient). When set, runAssessment() first calls
   * the real manualArrival() to bring this patient into the live queue
   * (reusing the chart's own vitals server-side — see manual_arrival()'s
   * stored-record fallback), then triages exactly as normal — and fires
   * automatically on mount so the outer "Triage Patient" click is the only
   * click the nurse needs.
   */
  activation?: { chiefComplaint: string; age: number; sex: string; acuity: number | null };
  onClose: () => void;
  onDone: () => void;
}) {
  const { bumpMutationTick } = useSession();
  const containerRef = useModalA11y(onClose);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<TriageResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overriding, setOverriding] = useState(false);
  const [overrideDept, setOverrideDept] = useState("");
  const [overrideReason, setOverrideReason] = useState("");

  const runAssessment = async () => {
    setSubmitting(true);
    setError(null);
    try {
      if (activation) {
        try {
          await api.manualArrival({
            patient_id: patientId,
            chief_complaint: activation.chiefComplaint,
            age: activation.age,
            sex: activation.sex,
            acuity: activation.acuity ?? 3,
            hospital_id: hospitalId,
          });
        } catch (e) {
          // A patient already active in this hospital's simulation (e.g. a
          // double-click that slipped past the disabled button, or a
          // concurrent registration from elsewhere) is not a real failure
          // here — the patient already exists in queue either way, so fall
          // through to triage. Any other error is real and must surface.
          const msg = e instanceof Error ? e.message : String(e);
          if (!/already active/i.test(msg)) throw e;
        }
      }
      const r = await api.triageSimulated(patientId, hospitalId);
      setResult(r);
      setOverrideDept(r.operational_decision.operational_department);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  // Guarded by a ref (not just the empty dep array) because React StrictMode
  // intentionally double-invokes effects in development — without this,
  // activation would fire manualArrival() + triageSimulated() twice per
  // mount. The backend's own collision lock now makes a genuine duplicate
  // patient impossible either way (see _MANUAL_ARRIVAL_LOCK), but there's no
  // reason to ever send the duplicate request in the first place.
  const activationStartedRef = useRef(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (activation && !activationStartedRef.current) {
      activationStartedRef.current = true;
      runAssessment();
    }
  }, []);

  const confirmOverride = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.overrideDepartment(patientId, overrideDept, overrideReason, hospitalId);
      bumpMutationTick();
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const op = result?.operational_decision;

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="max-h-[86vh] w-[560px] overflow-y-auto rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl">
        <p className="text-[15px] font-bold text-[var(--color-ink)]">
          Triage — <span className="font-mono">{patientId}</span>
        </p>
        <p className="mb-4 mt-0.5 text-xs text-[var(--color-ink-faint)]">
          {age}{sex} · {chiefComplaint}
        </p>

        <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Vitals</p>
        <div className="mb-4 grid grid-cols-3 gap-2 font-mono text-[13px]">
          {vitals.hr != null && <VitalChip label="HR" value={vitals.hr} />}
          {vitals.spo2 != null && <VitalChip label="SpO₂" value={`${vitals.spo2}%`} />}
          {vitals.rr != null && <VitalChip label="RR" value={vitals.rr} />}
          {vitals.sbp != null && <VitalChip label="SBP" value={vitals.sbp} />}
          {vitals.dbp != null && <VitalChip label="DBP" value={vitals.dbp} />}
          {vitals.temp != null && <VitalChip label="Temp" value={vitals.temp} />}
        </div>

        {error && (
          <div className="mb-3 rounded-lg border border-[var(--color-critical-100)] bg-[var(--color-critical-50)] px-3 py-2 text-xs text-[var(--color-critical-600)]">{error}</div>
        )}

        {!result && (
          <button
            onClick={runAssessment}
            disabled={submitting}
            className="w-full rounded-xl bg-[var(--color-brand-500)] px-4 py-2.5 text-[13px] font-bold text-white disabled:opacity-50"
          >
            {submitting ? "Running…" : "Run AI Clinical Assessment"}
          </button>
        )}

        {result && op && (
          <>
            <div className="mb-3.5 overflow-hidden rounded-xl border border-[var(--color-border)]">
              <div className="flex items-center justify-between px-3 py-2.5" style={{ background: "var(--color-brand-50)" }}>
                <span className="text-[9.5px] font-bold uppercase tracking-wide" style={{ color: "var(--color-brand-700)" }}>
                  AI Recommendation
                </span>
                <AcuityPill acuity={result.clinical_assessment.acuity_tier} />
              </div>
              <div className="px-3 py-2.5">
                <p className="text-[15px] font-bold text-[var(--color-ink)]">
                  {DEPT_LABELS[op.operational_department] ?? op.operational_department}
                </p>
                <p className="mt-1.5 text-[11.5px] leading-relaxed text-[var(--color-ink-soft)]">
                  {op.recommendation_summary}
                </p>
              </div>
            </div>

            {!overriding ? (
              <div className="flex gap-2">
                <button
                  onClick={onDone}
                  className="flex-1 rounded-lg px-4 py-2.5 text-xs font-bold text-white"
                  style={{ background: "var(--color-good-500)" }}
                >
                  Confirm — {DEPT_LABELS[op.operational_department] ?? op.operational_department}
                </button>
                <button
                  onClick={() => setOverriding(true)}
                  className="flex-1 rounded-lg border px-4 py-2.5 text-xs font-semibold"
                  style={{ borderColor: "var(--color-override-500)", background: "var(--color-override-50)", color: "var(--color-override-600)" }}
                >
                  Override Placement
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <select
                  value={overrideDept}
                  onChange={(e) => setOverrideDept(e.target.value)}
                  className="rounded-lg border border-[var(--color-border)] px-2.5 py-2 text-xs"
                >
                  {departmentOptions.map((d) => (
                    <option key={d} value={d}>{DEPT_LABELS[d] ?? d}</option>
                  ))}
                </select>
                <textarea
                  value={overrideReason}
                  onChange={(e) => setOverrideReason(e.target.value)}
                  placeholder="Reason for override…"
                  className="h-[52px] resize-none rounded-lg border border-[var(--color-border)] px-2.5 py-2 text-xs"
                />
                <button
                  onClick={confirmOverride}
                  disabled={submitting}
                  className="rounded-lg px-4 py-2.5 text-xs font-bold text-white disabled:opacity-50"
                  style={{ background: "var(--color-override-500)" }}
                >
                  {submitting ? "Applying…" : `Confirm Override — ${DEPT_LABELS[overrideDept] ?? overrideDept}`}
                </button>
              </div>
            )}
          </>
        )}

        <button
          onClick={onClose}
          className="mt-3.5 w-full rounded-lg px-4 py-2 text-xs font-semibold text-[var(--color-ink-faint)]"
        >
          {result ? "Close" : "Cancel"}
        </button>
      </div>
    </div>
  );
}

function VitalChip({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg bg-[var(--color-surface-muted)] px-2 py-1.5">
      <p className="text-[9px] uppercase text-[var(--color-ink-faint)]">{label}</p>
      <p className="font-semibold text-[var(--color-ink)]">{value}</p>
    </div>
  );
}
