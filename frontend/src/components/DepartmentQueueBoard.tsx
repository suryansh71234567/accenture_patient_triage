import { useState } from "react";
import { api } from "../api/client";
import { useSession } from "../state/SessionContext";
import { Badge, Button, DEPT_LABELS, acuityLabel, acuityTone } from "./ui";
import type { OperationalDecision, SimulationDashboard } from "../types";

// The four nurse-facing operational queues — always rendered (even empty)
// for whichever of these departments the hospital actually has, so a
// patient can be dragged into a currently-empty queue. Shared by Dashboard
// and LiveHospital so both pages present the same queues the same way
// (Phase 9's reorder/override APIs, unchanged).
const QUEUE_DEPARTMENTS = ["ICU", "CICU", "ADMITTED_GEN", "ED_OBS"];

type QueuePatient = {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  acuity: number;
  status: string;
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
  const { sessionId, proposeAction } = useSession();
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState<{ patientId: string; sourceDept: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fullQueue = ((dash as unknown as { full_queue?: QueuePatient[] }).full_queue ?? []) as QueuePatient[];
  const triagedPatients = fullQueue.filter((p) => p.status === "TRIAGED");
  const hospitalDeptNames = new Set(dash.departments.map((d) => d.name));
  const queueDepts = QUEUE_DEPARTMENTS.filter((d) => hospitalDeptNames.has(d));
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
        await onChanged();
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setBusy(false);
      }
      return;
    }

    const reason = window.prompt(
      `Override AI routing: move ${active.patientId} from ${DEPT_LABELS[active.sourceDept] ?? active.sourceDept} to ${DEPT_LABELS[targetDept] ?? targetDept}.\nReason (optional):`
    );
    if (reason === null) return; // nurse cancelled — do nothing
    setBusy(true);
    try {
      await api.overrideDepartment(active.patientId, targetDept, reason, hospitalId);
      await onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const listHeight = compact ? "max-h-[340px]" : "max-h-[620px]";

  return (
    <div className="space-y-3">
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">{error}</div>
      )}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[...byDept.entries()].map(([dept, patients]) => (
          <div
            key={dept}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); dropOnDepartment(dept, patients.length); }}
            className="flex flex-col rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)]"
          >
            <div className="flex items-center justify-between rounded-t-2xl border-b border-[var(--color-border)] bg-[var(--color-surface-muted)] px-3.5 py-2.5">
              <p className="text-sm font-bold text-[var(--color-ink)]">{DEPT_LABELS[dept] ?? dept}</p>
              <Badge tone="neutral">{patients.length} patient{patients.length !== 1 ? "s" : ""}</Badge>
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
                    onAdmit={admit}
                  />
                ))
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PatientQueueCard({
  p, busy, isDragging, onDragStart, onDragEnd, onDrop, onAdmit,
}: {
  p: QueuePatient;
  busy: boolean;
  isDragging: boolean;
  onDragStart: () => void;
  onDragEnd: () => void;
  onDrop: () => void;
  onAdmit: (patientId: string, department?: string) => void;
}) {
  const op = p.operational_decision;
  const aiDept = op?.ai_operational_department;
  const currentDept = op?.operational_department;

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); onDrop(); }}
      className={`cursor-move rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-2.5 text-xs transition ${
        isDragging ? "opacity-40" : "hover:border-[var(--color-brand-300)]"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 text-[var(--color-ink-faint)]" title="Drag to reorder or move to another department">⠿</span>
          <Badge tone={acuityTone(p.acuity)}>{acuityLabel(p.acuity)}</Badge>
          <p className="truncate text-sm font-semibold text-[var(--color-ink)]">{p.patient_id}</p>
        </div>
        <Button size="sm" disabled={busy} onClick={() => onAdmit(p.patient_id, currentDept ?? "")}>
          {op?.confirmation_required ? "Review" : "Admit"}
        </Button>
      </div>
      <p className="mt-1 truncate text-[11px] text-[var(--color-ink-faint)]">
        {p.age}y {p.sex} · {p.chief_complaint}
      </p>

      {op?.nurse_override ? (
        <div className="mt-1.5 rounded-md bg-purple-50 px-2 py-1 text-[10px] leading-relaxed text-purple-700">
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
    </div>
  );
}
