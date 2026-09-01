import { useState } from "react";
import { api } from "../api/client";
import { useModalA11y } from "../hooks/useModalA11y";
import { useSession } from "../state/SessionContext";
import { DEPT_LABELS } from "./ui";

type AdmitPhase = "confirm" | "applying" | "success" | "failure";

/**
 * Purpose-built admission modal (decision: separate from the generic
 * PendingActionModal, not a variant of it). Only shown when the patient's
 * operational_decision.confirmation_required is true — otherwise callers
 * admit directly with no modal, matching the real backend's own signal for
 * when a nurse actually needs to review before committing.
 *
 * Deliberately does NOT go through SessionContext.proposeAction(): that
 * path would also pop the generic PendingActionModal for the same
 * server-side confirmation gate, stacking two confirmation dialogs for one
 * action. Instead this orchestrates the same real two-step
 * executeTool -> confirmTool handshake directly, so exactly one modal is
 * shown. mutationTick is still bumped afterward so every other page's poll
 * picks up the change, same as the generic path does.
 */
export function AdmissionConfirmModal({
  patientId,
  department,
  reason,
  hospitalId,
  onClose,
  onSuccess,
}: {
  patientId: string;
  department: string;
  reason: string;
  hospitalId: string;
  onClose: () => void;
  onSuccess: () => void;
}) {
  const { sessionId, bumpMutationTick } = useSession();
  const [phase, setPhase] = useState<AdmitPhase>("confirm");
  // No Cancel/Close affordance exists during "applying" (the request is
  // already in flight) — Escape mirrors that same absence rather than
  // introducing a way to dismiss the modal that the mouse doesn't have.
  const containerRef = useModalA11y(phase === "applying" ? () => {} : onClose);

  const runAdmit = async () => {
    if (!sessionId) return;
    setPhase("applying");
    try {
      const kwargs = { patient_id: patientId, department, hospital_id: hospitalId };
      const first = await api.executeTool(sessionId, "admit_simulated_patient", kwargs);
      let finalStatus = first.status;
      if (first.status === "awaiting_confirmation") {
        const confirmed = await api.confirmTool(sessionId, true);
        finalStatus = confirmed.response_type === "error" ? "failed" : "executed";
      }
      if (finalStatus === "executed") {
        bumpMutationTick();
        setPhase("success");
        onSuccess();
      } else {
        setPhase("failure");
      }
    } catch {
      setPhase("failure");
    }
  };

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-[70] flex items-center justify-center bg-black/40 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="w-[420px] rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl">
        {phase === "confirm" && (
          <>
            <p className="text-sm font-bold text-[var(--color-ink)]">
              Confirm Admission — <span className="font-mono">{patientId}</span>
            </p>
            <div className="my-2.5 rounded-lg px-3 py-2 text-xs" style={{ background: "var(--color-warn-50)", color: "var(--color-warn-600)" }}>
              Nurse confirmation required — limited capacity in target department.
            </div>
            <div className="mb-4 rounded-xl border border-[var(--color-border)] p-3">
              <p className="text-[9.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Destination</p>
              <p className="mt-0.5 text-[15px] font-bold text-[var(--color-ink)]">{DEPT_LABELS[department] ?? department}</p>
              {reason && <p className="mt-1.5 text-[11.5px] text-[var(--color-ink-soft)]">{reason}</p>}
            </div>
            <div className="flex gap-2">
              <button
                onClick={onClose}
                className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-xs font-semibold text-[var(--color-ink-soft)]"
              >
                Cancel
              </button>
              <button
                onClick={runAdmit}
                className="flex-1 rounded-lg bg-[var(--color-good-500)] px-4 py-2 text-xs font-bold text-white"
              >
                Confirm Admission
              </button>
            </div>
          </>
        )}

        {phase === "applying" && (
          <div className="py-3 text-center">
            <p className="text-[13px] font-semibold text-[var(--color-ink)]">
              Admitting {patientId} to {DEPT_LABELS[department] ?? department}…
            </p>
            <p className="mt-1.5 text-xs text-[var(--color-ink-faint)]">Applying — please wait.</p>
          </div>
        )}

        {phase === "success" && (
          <div className="py-2.5 text-center">
            <p className="text-[13px] font-bold" style={{ color: "var(--color-good-600)" }}>✓ Admission complete</p>
            <p className="mt-1.5 text-[11.5px] text-[var(--color-ink-soft)]">
              {patientId} admitted to {DEPT_LABELS[department] ?? department}. Capacity updated.
            </p>
            <button
              onClick={onClose}
              className="mt-3.5 w-full rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-xs font-bold text-white"
            >
              Done
            </button>
          </div>
        )}

        {phase === "failure" && (
          <div className="py-2.5 text-center">
            <p className="text-[13px] font-bold text-[var(--color-critical-600)]">Admission failed</p>
            <p className="mt-1.5 text-[11.5px] text-[var(--color-ink-soft)]">Could not admit {patientId}. Please try again.</p>
            <div className="mt-3.5 flex gap-2">
              <button
                onClick={onClose}
                className="flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-xs font-semibold text-[var(--color-ink-soft)]"
              >
                Cancel
              </button>
              <button
                onClick={runAdmit}
                className="flex-1 rounded-lg bg-[var(--color-critical-500)] px-4 py-2 text-xs font-bold text-white"
              >
                Try Again
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
