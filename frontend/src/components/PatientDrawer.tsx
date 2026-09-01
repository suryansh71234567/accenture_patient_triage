import { useState } from "react";
import { Link } from "react-router-dom";
import { useModalA11y } from "../hooks/useModalA11y";
import { AcuityPill, DEPT_LABELS, aiDeptOf } from "./ui";
import type { OperationalDecision } from "../types";

export interface DrawerVital {
  label: string;
  value: string | number | null;
}

/**
 * Lightweight quick-view drawer (mockup's Patient Drawer). PatientWorkspace
 * remains the full page for the observation-recording workflow; this is a
 * fast glance from a list/board without leaving the page. Vitals are passed
 * in pre-normalized ({label, value}[]) by the caller since Dashboard/Live
 * Hospital (SimVitals, short keys) and the Patients table (PatientVitals,
 * long keys — or no simulation record at all) are genuinely different real
 * shapes; the drawer itself doesn't guess a mapping between them.
 */
export function PatientDrawer({
  patientId,
  age,
  sex,
  chiefComplaint,
  vitals,
  acuity,
  status,
  decision,
  onClose,
  onTriage,
  onEditVitals,
  onAdmit,
}: {
  patientId: string;
  age: number | null;
  sex: string | null;
  chiefComplaint: string;
  vitals: DrawerVital[];
  acuity: number | null;
  /** undefined = no simulation record for this patient at all (chart-only). */
  status?: "ARRIVED" | "TRIAGED" | "IN_TREATMENT" | "DISCHARGED" | string;
  decision?: OperationalDecision | null;
  onClose: () => void;
  onTriage?: () => void;
  onEditVitals?: () => void;
  onAdmit?: () => void;
}) {
  const [tab, setTab] = useState<"live" | "record">("live");
  const containerRef = useModalA11y(onClose);

  return (
    <>
      <div onClick={onClose} className="fixed inset-0 z-50 bg-black/30" />
      <div
        ref={containerRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={`Patient ${patientId} details`}
        className="fixed right-0 top-0 z-[51] flex h-full w-[440px] flex-col bg-[var(--color-surface)] shadow-[-16px_0_40px_rgba(0,0,0,.12)]"
      >
        <div className="px-5 pt-4">
          <div className="flex items-start justify-between">
            <div>
              <p className="font-mono text-base font-bold text-[var(--color-ink)]">{patientId}</p>
              <p className="mt-0.5 text-[11.5px] text-[var(--color-ink-faint)]">
                {age ?? "—"} {sex ?? ""} · {chiefComplaint}
              </p>
            </div>
            <button onClick={onClose} aria-label="Close patient details" className="text-lg text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]">×</button>
          </div>
          <div className="mt-4 flex w-fit gap-1 rounded-lg bg-[var(--color-surface-muted)] p-0.5">
            <button
              onClick={() => setTab("live")}
              className="rounded-md px-3.5 py-1.5 text-[11.5px] font-semibold transition"
              style={tab === "live" ? { background: "#fff", color: "var(--color-ink)" } : { color: "var(--color-ink-faint)" }}
            >
              Live Status
            </button>
            <button
              onClick={() => setTab("record")}
              className="rounded-md px-3.5 py-1.5 text-[11.5px] font-semibold transition"
              style={tab === "record" ? { background: "#fff", color: "var(--color-ink)" } : { color: "var(--color-ink-faint)" }}
            >
              Record
            </button>
          </div>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
          {tab === "live" ? (
            <>
              <div className="grid grid-cols-2 gap-2">
                {vitals.map((v) => (
                  <div key={v.label} className="rounded-lg bg-[var(--color-surface-muted)] px-2.5 py-2">
                    <p className="text-[9.5px] uppercase tracking-wide text-[var(--color-ink-faint)]">{v.label}</p>
                    <p className="font-mono text-sm font-semibold text-[var(--color-ink)]">{v.value ?? "—"}</p>
                  </div>
                ))}
              </div>

              {/* undefined = chart-only, never entered the live queue — same
                  "not yet triaged" treatment as an ARRIVED simulation
                  patient, since Triage Patient can now bring either into an
                  active visit and triage them in one action. */}
              {(status === "ARRIVED" || status === undefined) && (
                <>
                  <EmptyLine text="Not yet triaged. Clinical and operational assessment has not been performed." tone="warn" />
                  {onTriage && (
                    <button onClick={onTriage} className="w-full rounded-lg bg-[var(--color-brand-500)] px-4 py-2.5 text-xs font-bold text-white">
                      Triage Patient
                    </button>
                  )}
                </>
              )}

              {status === "TRIAGED" && decision && (
                <>
                  <div className="overflow-hidden rounded-xl border border-[var(--color-border)]">
                    <div className="px-3 py-2.5" style={{ background: "var(--color-brand-50)" }}>
                      <p className="text-[9.5px] font-bold uppercase tracking-wide" style={{ color: "var(--color-brand-700)" }}>AI Recommendation</p>
                      <p className="mt-0.5 text-sm font-bold text-[var(--color-ink)]">{DEPT_LABELS[aiDeptOf(decision)] ?? aiDeptOf(decision)}</p>
                    </div>
                    <div className="border-t border-[var(--color-border)] px-3 py-2.5">
                      <p className="text-[9.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Current Queue</p>
                      <p className="mt-0.5 text-sm font-bold text-[var(--color-ink)]">{DEPT_LABELS[decision.operational_department] ?? decision.operational_department}</p>
                    </div>
                    {decision.nurse_override && (
                      <div className="border-t border-[var(--color-border)] px-3 py-2.5" style={{ background: "var(--color-override-50)" }}>
                        <p className="text-[9.5px] font-bold uppercase tracking-wide" style={{ color: "var(--color-override-600)" }}>Nurse Override</p>
                        <p className="mt-0.5 text-[13px] font-semibold text-[var(--color-ink)]">
                          {DEPT_LABELS[aiDeptOf(decision)] ?? aiDeptOf(decision)} → {DEPT_LABELS[decision.operational_department] ?? decision.operational_department}
                        </p>
                        {decision.override_reason && <p className="mt-0.5 text-[11px] italic" style={{ color: "var(--color-override-600)" }}>"{decision.override_reason}"</p>}
                      </div>
                    )}
                  </div>
                  <div>
                    <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">AI Assessment Reasoning</p>
                    <p className="rounded-lg bg-[var(--color-surface-muted)] px-3 py-2.5 text-xs leading-relaxed text-[var(--color-ink)]">{decision.recommendation_summary}</p>
                  </div>
                  {decision.confirmation_required && (
                    <EmptyLine text="Nurse confirmation required — limited capacity in target department." tone="warn" />
                  )}
                  <div className="flex gap-2">
                    {onEditVitals && (
                      <button onClick={onEditVitals} className="flex-1 rounded-lg border border-[var(--color-border)] px-4 py-2.5 text-xs font-semibold">
                        Update Vitals
                      </button>
                    )}
                    {onAdmit && (
                      <button onClick={onAdmit} className="flex-1 rounded-lg px-4 py-2.5 text-xs font-bold text-white" style={{ background: "var(--color-good-500)" }}>
                        Admit to {DEPT_LABELS[decision.operational_department] ?? decision.operational_department}
                      </button>
                    )}
                  </div>
                </>
              )}

              {status === "IN_TREATMENT" && decision && (
                <EmptyLine
                  text={`Admitted to ${DEPT_LABELS[decision.operational_department] ?? decision.operational_department}. No longer in active queue.`}
                  tone="good"
                />
              )}

              {status === "DISCHARGED" && (
                <EmptyLine
                  text={
                    decision
                      ? `Discharged from ${DEPT_LABELS[decision.operational_department] ?? decision.operational_department}. No longer an active patient.`
                      : "Discharged. No longer an active patient."
                  }
                  tone="good"
                />
              )}
            </>
          ) : (
            <>
              <div>
                <p className="mb-1 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Chief Complaint</p>
                <p className="text-[13px] text-[var(--color-ink)]">{chiefComplaint}</p>
              </div>
              {decision?.retriage && (
                <div>
                  <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide" style={{ color: "var(--color-brand-600)" }}>Re-Triage History</p>
                  <div className="space-y-1.5 rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-[11.5px]">
                    <p><b>Previous:</b> {DEPT_LABELS[decision.previous_operational_department ?? ""] ?? decision.previous_operational_department}</p>
                    <p><b>Updated:</b> {acuity != null && <AcuityPill acuity={acuity} />} {DEPT_LABELS[decision.operational_department] ?? decision.operational_department}</p>
                    {decision.previous_nurse_override && (
                      <p className="text-[10.5px]" style={{ color: "var(--color-override-600)" }}>
                        Previous nurse override on record: {decision.previous_override_reason}
                      </p>
                    )}
                  </div>
                </div>
              )}
              {decision?.nurse_override && (
                <div>
                  <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide" style={{ color: "var(--color-override-600)" }}>Operational Decision History</p>
                  <div className="rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-[11.5px] text-[var(--color-ink)]">
                    Nurse moved patient from AI-recommended {DEPT_LABELS[aiDeptOf(decision)] ?? aiDeptOf(decision)} to {DEPT_LABELS[decision.operational_department] ?? decision.operational_department}.
                    <br />Reason: "{decision.override_reason}"
                  </div>
                </div>
              )}
              {!decision?.retriage && !decision?.nurse_override && (
                <p className="text-[11.5px] text-[var(--color-ink-faint)]">No prior overrides or re-triage events on record.</p>
              )}
              <Link
                to={`/patients/${patientId}`}
                className="block rounded-lg border border-[var(--color-border)] px-4 py-2.5 text-center text-xs font-semibold text-[var(--color-brand-600)] hover:bg-[var(--color-surface-muted)]"
              >
                Open full patient workspace →
              </Link>
            </>
          )}
        </div>
      </div>
    </>
  );
}

function EmptyLine({ text, tone = "neutral" }: { text: string; tone?: "neutral" | "warn" | "good" }) {
  const style =
    tone === "warn"
      ? { background: "var(--color-warn-50)", color: "var(--color-warn-600)" }
      : tone === "good"
        ? { background: "var(--color-good-50)", color: "var(--color-good-600)" }
        : { background: "var(--color-surface-muted)", color: "var(--color-ink-soft)" };
  return (
    <div className="rounded-lg px-3 py-2.5 text-xs" style={style}>
      {text}
    </div>
  );
}
