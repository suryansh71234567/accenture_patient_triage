import { useState } from "react";
import { api } from "../api/client";
import { useSession } from "../state/SessionContext";
import { AdmissionConfirmModal } from "./AdmissionConfirmModal";
import { OverrideReasonModal } from "./OverrideReasonModal";
import { RetriageModal } from "./RetriageModal";
import { VitalsEditModal } from "./VitalsEditModal";
import { AcuityPill, Button, DEPT_LABELS, acuityMeta, aiDeptOf, deptStatus, fmtWaitMinutes } from "./ui";
import type { OperationalDecision, SimVitals, SimulationDashboard } from "../types";

type QueuePatient = {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  acuity: number;
  status: string;
  vitals?: SimVitals;
  arrival_time_min?: number;
  clinical_assessment?: Record<string, unknown> | null;
  operational_decision?: OperationalDecision | null;
  metadata?: Record<string, unknown> | null;
};

function getDeptForPatient(p: QueuePatient): string {
  // Groups by the CURRENT operational destination (reflects any nurse
  // override), not the pure clinical preference — mirrors the backend's
  // patient_flow.department_of().
  const op = p.operational_decision;
  const ca = p.clinical_assessment as Record<string, string> | null | undefined;
  return op?.operational_department ?? ca?.department ?? "UNKNOWN";
}

function groupByDept(patients: QueuePatient[], deptOrder: string[]): Map<string, QueuePatient[]> {
  const map = new Map<string, QueuePatient[]>();
  for (const dept of deptOrder) map.set(dept, []);
  for (const p of patients) {
    const dept = getDeptForPatient(p);
    if (!map.has(dept)) map.set(dept, []);
    map.get(dept)!.push(p);
  }
  return map;
}

/**
 * Hospital-wide department queue board: same-department drag reorder,
 * cross-department drag = nurse override (Phase 9 APIs, reused unchanged).
 * `compact` trims column height for the Dashboard's more scannable layout;
 * LiveHospital uses the full-height version.
 */
export function DepartmentQueueBoard({
  dash,
  hospitalId,
  onChanged,
  compact = false,
}: {
  dash: SimulationDashboard;
  hospitalId: string;
  onChanged: () => Promise<unknown>;
  compact?: boolean;
}) {
  const { sessionId, proposeAction, bumpMutationTick } = useSession();
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState<{ patientId: string; sourceDept: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overrideTarget, setOverrideTarget] = useState<{ patientId: string; from: string; to: string; targetIndex: number } | null>(null);
  const [vitalsPatient, setVitalsPatient] = useState<QueuePatient | null>(null);
  const [retriagePatient, setRetriagePatient] = useState<QueuePatient | null>(null);
  const [admitTarget, setAdmitTarget] = useState<{ patientId: string; department: string; reason: string } | null>(null);

  const fullQueue = ((dash as unknown as { full_queue?: QueuePatient[] }).full_queue ?? []) as QueuePatient[];
  const triagedPatients = fullQueue.filter((p) => p.status === "TRIAGED");
  // Always rendered (even empty) for whichever departments this hospital
  // actually has — read live from the dashboard response, never a fixed
  // list, so a patient can be dragged into a currently-empty queue for any
  // hospital regardless of its department names. Shared by Dashboard and
  // LiveHospital so both pages present the same queues the same way.
  const queueDepts = dash.departments.map((d) => d.name);
  const byDept = groupByDept(triagedPatients, queueDepts);

  const admit = async (patientId: string, department?: string) => {
    if (!sessionId) return;
    setBusy(true);
    try {
      const outcome = await proposeAction("admit_simulated_patient", {
        patient_id: patientId,
        ...(department ? { department } : {}),
        ...(hospitalId ? { hospital_id: hospitalId } : {}),
      });
      if (outcome.status === "executed") await onChanged();
    } finally {
      setBusy(false);
    }
  };

  const requestAdmit = (p: QueuePatient) => {
    const op = p.operational_decision;
    const dept = op?.operational_department ?? "";
    if (op?.confirmation_required) {
      setAdmitTarget({ patientId: p.patient_id, department: dept, reason: op.recommendation_summary });
    } else {
      admit(p.patient_id, dept);
    }
  };

  // Same department = reorder priority; different department = nurse
  // override of the operational destination. Infeasible moves surface the
  // backend's rejection reason rather than silently no-op'ing.
  const dropOnDepartment = async (targetDept: string, targetIndex: number) => {
    const active = drag;
    setDrag(null);
    if (!active) return;
    setError(null);

    if (active.sourceDept === targetDept) {
      setBusy(true);
      try {
        await api.reorderDepartmentQueue(active.patientId, targetDept, targetIndex, hospitalId);
        bumpMutationTick();
        await onChanged();
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setBusy(false);
      }
      return;
    }

    setOverrideTarget({ patientId: active.patientId, from: active.sourceDept, to: targetDept, targetIndex });
  };

  const confirmOverride = async (reason: string) => {
    if (!overrideTarget) return;
    setBusy(true);
    try {
      await api.overrideDepartment(overrideTarget.patientId, overrideTarget.to, reason, hospitalId);
      bumpMutationTick();
      await onChanged();
      setOverrideTarget(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const saveVitals = async (patientId: string, vitals: SimVitals) => {
    setBusy(true);
    setError(null);
    try {
      if (Object.keys(vitals).length > 0) {
        await api.updateSimulatedVitals(patientId, vitals, hospitalId);
      }
      await api.triageSimulated(patientId, hospitalId);
      bumpMutationTick();
      setVitalsPatient(null);
      await onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const retriageOnly = async (patientId: string) => {
    setBusy(true);
    setError(null);
    try {
      await api.triageSimulated(patientId, hospitalId);
      bumpMutationTick();
      await onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const requestMove = (p: QueuePatient, sourceDept: string, targetDept: string) => {
    if (targetDept === sourceDept) return;
    // Routes through the exact same nurse-override confirmation flow a
    // cross-department drag uses — a keyboard move is not a shortcut that
    // skips the reason prompt, it's the same action via a different input.
    setOverrideTarget({ patientId: p.patient_id, from: sourceDept, to: targetDept, targetIndex: 0 });
  };

  const listHeight = compact ? "max-h-[340px]" : "max-h-[620px]";
  // `xl:` is a viewport breakpoint, not a container query — on Dashboard the
  // board's actual container is much narrower than the viewport (it shares
  // the row with a Recent Activity panel), so forcing 4 columns there at
  // wide viewports crams each column too narrow for its card content.
  // compact mode caps at 2 columns instead; the full-width LiveHospital
  // board keeps 4.
  const gridCols = compact ? "md:grid-cols-2" : "md:grid-cols-2 xl:grid-cols-4";

  return (
    <div className="space-y-3">
      {error && (
        <div className="rounded-lg border border-[var(--color-critical-100)] bg-[var(--color-critical-50)] px-3 py-2 text-xs text-[var(--color-critical-600)]">{error}</div>
      )}
      <div className={`grid grid-cols-1 gap-4 ${gridCols}`}>
        {[...byDept.entries()].map(([dept, patients]) => {
          const capacity = dash.departments.find((d) => d.name === dept);
          const status = capacity ? deptStatus(capacity.occupied, capacity.capacity) : null;
          return (
            <div
              key={dept}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); dropOnDepartment(dept, patients.length); }}
              className="flex flex-col rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]"
            >
              <div className="rounded-t-xl border-b border-[var(--color-border)] bg-[var(--color-surface-muted)] px-3.5 py-2.5">
                <div className="mb-2 flex items-baseline justify-between gap-2">
                  <p className="text-[13px] font-bold text-[var(--color-ink)]">{DEPT_LABELS[dept] ?? dept}</p>
                  {status && (
                    <span
                      className="rounded-[5px] px-[7px] py-[2px] text-[9.5px] font-bold tracking-[.05em]"
                      style={{ background: status.bg, color: status.color }}
                    >
                      {status.label}
                    </span>
                  )}
                </div>
                {capacity && (
                  <>
                    <div className="mb-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                      <div className="h-full rounded-full" style={{ width: `${status?.pct ?? 0}%`, background: status?.color }} />
                    </div>
                    <div className="flex items-center justify-between text-[10.5px] text-[var(--color-ink-faint)]">
                      <span className="font-mono">{capacity.occupied}/{capacity.capacity} beds</span>
                      <span>{patients.length} in queue</span>
                    </div>
                  </>
                )}
              </div>
              <div className={`min-h-[110px] flex-1 space-y-2 overflow-y-auto p-2.5 ${listHeight}`}>
                {patients.length === 0 ? (
                  <div className="flex h-full min-h-[90px] items-center justify-center rounded-xl border border-dashed border-[var(--color-border)] px-3 text-center text-[11px] text-[var(--color-ink-faint)]">
                    Drop patient here
                  </div>
                ) : (
                  patients.map((p, idx) => (
                    <PatientQueueCard
                      key={p.patient_id}
                      p={p}
                      busy={busy}
                      isDragging={drag?.patientId === p.patient_id}
                      onDragStart={() => setDrag({ patientId: p.patient_id, sourceDept: dept })}
                      onDragEnd={() => setDrag(null)}
                      onDrop={() => dropOnDepartment(dept, idx)}
                      onAdmit={() => requestAdmit(p)}
                      onEditVitals={() => setVitalsPatient(p)}
                      onRetriage={() => retriageOnly(p.patient_id)}
                      onOpenRetriageDetail={() => setRetriagePatient(p)}
                      simTimeMinutes={dash.sim_time_minutes}
                      moveTargets={queueDepts.filter((d) => d !== dept)}
                      onMoveTo={(targetDept) => requestMove(p, dept, targetDept)}
                    />
                  ))
                )}
              </div>
            </div>
          );
        })}
      </div>

      {overrideTarget && (
        <OverrideReasonModal
          patientId={overrideTarget.patientId}
          from={overrideTarget.from}
          to={overrideTarget.to}
          busy={busy}
          onConfirm={confirmOverride}
          onCancel={() => setOverrideTarget(null)}
        />
      )}

      {vitalsPatient && (
        <VitalsEditModal
          patientId={vitalsPatient.patient_id}
          busy={busy}
          onSave={(vitals) => saveVitals(vitalsPatient.patient_id, vitals)}
          onClose={() => setVitalsPatient(null)}
        />
      )}

      {retriagePatient && retriagePatient.operational_decision && (
        <RetriageModal
          patientId={retriagePatient.patient_id}
          vitals={retriagePatient.vitals ?? {}}
          acuity={retriagePatient.acuity}
          decision={retriagePatient.operational_decision}
          onAcknowledge={() => setRetriagePatient(null)}
          onClose={() => setRetriagePatient(null)}
        />
      )}

      {admitTarget && (
        <AdmissionConfirmModal
          patientId={admitTarget.patientId}
          department={admitTarget.department}
          reason={admitTarget.reason}
          hospitalId={hospitalId}
          onClose={() => setAdmitTarget(null)}
          onSuccess={() => { setAdmitTarget(null); onChanged(); }}
        />
      )}
    </div>
  );
}

function PatientQueueCard({
  p, busy, isDragging, onDragStart, onDragEnd, onDrop, onAdmit, onEditVitals, onRetriage, onOpenRetriageDetail, simTimeMinutes,
  moveTargets, onMoveTo,
}: {
  p: QueuePatient;
  busy: boolean;
  isDragging: boolean;
  onDragStart: () => void;
  onDragEnd: () => void;
  onDrop: () => void;
  onAdmit: () => void;
  onEditVitals: () => void;
  onRetriage: () => void;
  onOpenRetriageDetail: () => void;
  simTimeMinutes: number;
  /** Departments this card's patient isn't already in — a keyboard-usable
   * alternative to cross-department drag, not a shortcut around it: picking
   * one opens the same nurse-override confirmation drag-and-drop uses. */
  moveTargets: string[];
  onMoveTo: (targetDept: string) => void;
}) {
  const op = p.operational_decision;
  const aiDept = op ? aiDeptOf(op) : undefined;
  const currentDept = op?.operational_department;
  const isRetriage = Boolean(op?.retriage);
  const priorOverrideBeforeRetriage = isRetriage && op?.previous_nurse_override;

  const waitLabel =
    p.arrival_time_min != null ? fmtWaitMinutes(Math.max(0, simTimeMinutes - p.arrival_time_min)) : null;

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); onDrop(); }}
      style={{ borderLeftColor: acuityMeta(p.acuity).color, borderLeftWidth: 3 }}
      className={`cursor-move rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-2.5 text-xs transition ${
        isDragging ? "opacity-40" : "hover:border-[var(--color-brand-300)]"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 text-[var(--color-ink-faint)]" title="Drag to reorder or move to another department">⠿</span>
          <AcuityPill acuity={p.acuity} />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {waitLabel && <span className="font-mono text-[10px] text-[var(--color-ink-faint)]">{waitLabel}</span>}
          <Button size="sm" disabled={busy} onClick={onAdmit}>
            {op?.confirmation_required ? "Review" : "Admit"}
          </Button>
        </div>
      </div>
      <div className="mt-1 flex items-baseline justify-between gap-2">
        <p className="truncate font-mono text-sm font-semibold text-[var(--color-ink)]">{p.patient_id}</p>
        <span className="shrink-0 text-[10.5px] text-[var(--color-ink-faint)]">{p.age}y {p.sex}</span>
      </div>
      <p className="mt-0.5 truncate text-[11px] text-[var(--color-ink-faint)]">{p.chief_complaint}</p>
      {p.vitals && (p.vitals.hr != null || p.vitals.spo2 != null || p.vitals.sbp != null) && (
        <div className="mt-1 flex gap-2.5 font-mono text-[10px] text-[var(--color-ink-faint)]">
          {p.vitals.hr != null && <span>HR {p.vitals.hr}</span>}
          {p.vitals.spo2 != null && <span>SpO₂ {p.vitals.spo2}%</span>}
          {p.vitals.sbp != null && p.vitals.dbp != null && <span>{p.vitals.sbp}/{p.vitals.dbp}</span>}
        </div>
      )}

      {isRetriage && (
        <button
          onClick={onOpenRetriageDetail}
          className="mt-1.5 w-full rounded-md px-2 py-1 text-left text-[10px] font-semibold leading-relaxed"
          style={
            priorOverrideBeforeRetriage
              ? { background: "var(--color-warn-50)", color: "var(--color-warn-600)" }
              : { background: "var(--color-retriage-50)", color: "var(--color-retriage-600)" }
          }
        >
          🔄 Re-triaged — new AI recommendation, requires nurse review
          {priorOverrideBeforeRetriage && (
            <div className="mt-0.5 font-normal italic">
              Prior nurse override to {DEPT_LABELS[op?.previous_operational_department ?? ""] ?? op?.previous_operational_department} no longer applies to this new assessment.
            </div>
          )}
        </button>
      )}

      {op?.nurse_override ? (
        <div
          className="mt-1.5 rounded-md px-2 py-1 text-[10px] leading-relaxed"
          style={{ background: "var(--color-override-50)", color: "var(--color-override-600)" }}
        >
          <div>AI recommended: <strong>{DEPT_LABELS[aiDept ?? ""] ?? aiDept}</strong></div>
          <div className="font-semibold">
            ✋ Nurse override: {DEPT_LABELS[aiDept ?? ""] ?? aiDept} → {DEPT_LABELS[currentDept ?? ""] ?? currentDept}
          </div>
          {op.override_reason && <div className="italic">"{op.override_reason}"</div>}
        </div>
      ) : (
        <div className="mt-1.5 text-[10px] leading-relaxed text-[var(--color-ink-faint)]">
          <div>AI recommended: <span className="font-medium text-[var(--color-ink-soft)]">{DEPT_LABELS[aiDept ?? ""] ?? aiDept}</span></div>
          <div>Currently queued: <span className="font-medium text-[var(--color-ink-soft)]">{DEPT_LABELS[currentDept ?? ""] ?? currentDept}</span></div>
        </div>
      )}

      <div className="mt-1.5 flex gap-1.5">
        <button
          disabled={busy}
          onClick={onEditVitals}
          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-raised)] disabled:opacity-30 transition"
        >
          Vitals
        </button>
        <button
          disabled={busy}
          onClick={onRetriage}
          title="Re-run triage using this patient's current vitals"
          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-raised)] disabled:opacity-30 transition"
        >
          Re-triage
        </button>
        {moveTargets.length > 0 && (
          <select
            aria-label={`Move ${p.patient_id} to a different department`}
            title="Keyboard-accessible alternative to dragging this card to another department"
            disabled={busy}
            value=""
            onChange={(e) => {
              const targetDept = e.target.value;
              if (targetDept) onMoveTo(targetDept);
              e.target.value = "";
            }}
            className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-raised)] disabled:opacity-30 transition"
          >
            <option value="">Move to ▾</option>
            {moveTargets.map((d) => (
              <option key={d} value={d}>
                {DEPT_LABELS[d] ?? d}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}
