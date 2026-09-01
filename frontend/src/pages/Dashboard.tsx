import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { AcuityPill, DEPT_LABELS, DeptGauge, EmptyState, Spinner, fmtWaitMinutes } from "../components/ui";
import { PatientDrawer, type DrawerVital } from "../components/PatientDrawer";
import { VitalsEditModal } from "../components/VitalsEditModal";
import { AdmissionConfirmModal } from "../components/AdmissionConfirmModal";
import type { OperationalDecision, SimVitals } from "../types";

type FullQueueEntry = {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  acuity: number;
  status: string;
  vitals?: SimVitals;
  arrival_time_min?: number;
  operational_decision?: OperationalDecision | null;
};

type DeptPreview = {
  name: string;
  queueCount: number;
  patients: FullQueueEntry[];
};

/** Top-3-by-acuity, department-grouped read-only preview — the mockup's "Live Queues by
 * Department" panel. Full drag/reorder/override/admit interactivity lives exclusively on the
 * Live Hospital screen's DepartmentQueueBoard; this panel only opens the patient drawer. */
function buildDeptPreviews(departments: { name: string }[], fullQueue: FullQueueEntry[]): DeptPreview[] {
  const byDept = new Map<string, FullQueueEntry[]>();
  for (const d of departments) byDept.set(d.name, []);
  for (const p of fullQueue) {
    if (p.status !== "TRIAGED") continue;
    const dept = p.operational_decision?.operational_department ?? "UNKNOWN";
    if (!byDept.has(dept)) byDept.set(dept, []);
    byDept.get(dept)!.push(p);
  }
  return departments.map((d) => {
    const list = (byDept.get(d.name) ?? []).slice().sort((a, b) => (a.acuity ?? 9) - (b.acuity ?? 9));
    return { name: d.name, queueCount: list.length, patients: list.slice(0, 3) };
  });
}

export function Dashboard() {
  const { mutationTick, hospitalId, proposeAction } = useSession();
  const { data, loading, refetch } = usePoll(() => api.dashboard(hospitalId), 6000, [mutationTick, hospitalId]);
  const [drawerPatientId, setDrawerPatientId] = useState<string | null>(null);
  const [vitalsTarget, setVitalsTarget] = useState<string | null>(null);
  const [admitTarget, setAdmitTarget] = useState<{ patientId: string; department: string; reason: string } | null>(null);

  if (loading && !data) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
      </div>
    );
  }
  if (!data) return <EmptyState title="Could not reach the hospital simulation service." />;

  const fullQueue = (data.full_queue as unknown as FullQueueEntry[]) ?? [];
  const criticalCount = fullQueue.filter((p) => p.acuity <= 2).length;
  const deptPreviews = buildDeptPreviews(data.departments, fullQueue);
  const simEntry = fullQueue.find((p) => p.patient_id === drawerPatientId) ?? null;

  return (
    <div className="mx-auto max-w-[1400px] space-y-5 p-[22px_24px]">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-[var(--color-ink-faint)]">
          Simulation time <span className="font-mono font-semibold text-[var(--color-ink)]">{data.time}</span>
        </span>
        {criticalCount > 0 && (
          <span className="flex items-center gap-1.5 font-semibold text-[var(--color-critical-600)]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-critical-500)]" />
            {criticalCount} critical / high-acuity patient{criticalCount !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-3">
        <PipelineStat
          label="Waiting for Triage"
          caption="not yet clinically assessed"
          value={data.untriaged_count}
          colorVar="--color-warn-500"
        />
        <PipelineStat
          label="Triaged · Awaiting Admission"
          caption="queued in departments"
          value={data.triaged_count}
          colorVar="--color-brand-500"
        />
        <PipelineStat
          label="In Treatment"
          caption="admitted, occupying beds"
          value={data.admitted_count}
          colorVar="--color-good-500"
        />
      </div>

      <div>
        <p className="mb-2.5 text-[13px] font-bold text-[var(--color-ink)]">Hospital Capacity</p>
        <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${data.departments.length || 1},1fr)` }}>
          {data.departments.map((d) => (
            <DeptGauge key={d.name} {...d} />
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[2.2fr_1fr]">
        <div>
          <p className="mb-2.5 text-[13px] font-bold text-[var(--color-ink)]">Live Queues by Department</p>
          <div className="flex flex-col gap-2.5">
            {deptPreviews.map((d) => (
              <div key={d.name} className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3.5 py-3">
                <p className="mb-2 text-xs font-bold text-[var(--color-ink-soft)]">
                  {DEPT_LABELS[d.name] ?? d.name}{" "}
                  <span className="font-mono font-normal text-[var(--color-ink-faint)]">({d.queueCount})</span>
                </p>
                <div className="flex flex-col gap-1.5">
                  {d.patients.length === 0 ? (
                    <p className="rounded-lg bg-[var(--color-surface-muted)] px-2.5 py-2 text-[11px] text-[var(--color-ink-faint)]">
                      No patients currently queued.
                    </p>
                  ) : (
                    d.patients.map((p) => (
                      <button
                        key={p.patient_id}
                        onClick={() => setDrawerPatientId(p.patient_id)}
                        className="flex items-center gap-2.5 rounded-lg bg-[var(--color-surface-muted)] px-[9px] py-[7px] text-left hover:bg-[var(--color-surface-raised)]"
                      >
                        <AcuityPill acuity={p.acuity} withLabel={false} />
                        <span className="shrink-0 font-mono text-xs font-semibold text-[var(--color-ink)]">{p.patient_id}</span>
                        <span className="min-w-0 flex-1 truncate text-[11.5px] text-[var(--color-ink-soft)]">{p.chief_complaint}</span>
                        {p.operational_decision?.nurse_override && (
                          <span
                            className="shrink-0 rounded-[5px] px-[6px] py-[2px] text-[9.5px] font-bold"
                            style={{ background: "var(--color-override-50)", color: "var(--color-override-600)" }}
                          >
                            OVERRIDE
                          </span>
                        )}
                        {p.operational_decision?.retriage && (
                          <span
                            className="shrink-0 rounded-[5px] px-[6px] py-[2px] text-[9.5px] font-bold"
                            style={{ background: "var(--color-retriage-50)", color: "var(--color-retriage-600)" }}
                          >
                            RE-TRIAGED
                          </span>
                        )}
                        <span className="shrink-0 font-mono text-[10.5px] text-[var(--color-ink-faint)]">
                          {p.arrival_time_min != null ? fmtWaitMinutes(Math.max(0, data.sim_time_minutes - p.arrival_time_min)) : ""}
                        </span>
                      </button>
                    ))
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-2.5 text-[13px] font-bold text-[var(--color-ink)]">Recent Activity</p>
          <div className="max-h-[420px] overflow-y-auto rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] py-1.5">
            {data.recent_events.length === 0 ? (
              <p className="px-3.5 py-3 text-xs text-[var(--color-ink-faint)]">No events yet.</p>
            ) : (
              [...data.recent_events].reverse().map((e, i) => (
                <div key={i} className="border-b border-[var(--color-border)] px-3.5 py-2.5 text-[11.5px] leading-[1.4] text-[var(--color-ink)] last:border-0">
                  {e}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {drawerPatientId && simEntry && (
        <PatientDrawer
          patientId={drawerPatientId}
          age={simEntry.age}
          sex={simEntry.sex}
          chiefComplaint={simEntry.chief_complaint}
          acuity={simEntry.acuity}
          status={simEntry.status as "ARRIVED" | "TRIAGED" | "IN_TREATMENT" | "DISCHARGED" | undefined}
          decision={simEntry.operational_decision}
          vitals={simVitals(simEntry.vitals)}
          onClose={() => setDrawerPatientId(null)}
          onEditVitals={simEntry.status === "TRIAGED" ? () => setVitalsTarget(drawerPatientId) : undefined}
          onAdmit={
            simEntry.status === "TRIAGED" && simEntry.operational_decision
              ? async () => {
                  const op = simEntry.operational_decision!;
                  if (op.confirmation_required) {
                    setAdmitTarget({ patientId: drawerPatientId, department: op.operational_department, reason: op.recommendation_summary });
                    return;
                  }
                  const outcome = await proposeAction("admit_simulated_patient", {
                    patient_id: drawerPatientId,
                    department: op.operational_department,
                    hospital_id: hospitalId,
                  });
                  if (outcome.status === "executed") {
                    setDrawerPatientId(null);
                    refetch();
                  }
                }
              : undefined
          }
        />
      )}

      {vitalsTarget && (
        <VitalsEditModal
          patientId={vitalsTarget}
          busy={false}
          onSave={async (vitals) => {
            await api.updateSimulatedVitals(vitalsTarget, vitals, hospitalId);
            await api.triageSimulated(vitalsTarget, hospitalId);
            setVitalsTarget(null);
            refetch();
          }}
          onClose={() => setVitalsTarget(null)}
        />
      )}

      {admitTarget && (
        <AdmissionConfirmModal
          patientId={admitTarget.patientId}
          department={admitTarget.department}
          reason={admitTarget.reason}
          hospitalId={hospitalId}
          onClose={() => setAdmitTarget(null)}
          onSuccess={() => { setAdmitTarget(null); setDrawerPatientId(null); refetch(); }}
        />
      )}
    </div>
  );
}

function simVitals(v: SimVitals | undefined): DrawerVital[] {
  return [
    { label: "HR", value: v?.hr ?? null },
    { label: "SpO₂", value: v?.spo2 != null ? `${v.spo2}%` : null },
    { label: "RR", value: v?.rr ?? null },
    { label: "BP", value: v?.sbp != null && v?.dbp != null ? `${v.sbp}/${v.dbp}` : null },
    { label: "Temp", value: v?.temp ?? null },
    { label: "Pain", value: v?.pain ?? null },
  ];
}

function PipelineStat({
  label,
  caption,
  value,
  colorVar,
}: {
  label: string;
  caption: string;
  value: number;
  colorVar: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-5 py-[18px]">
      <p className="text-[10.5px] font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">{label}</p>
      <p className="mt-2 font-mono text-[34px] font-semibold leading-none" style={{ color: `var(${colorVar})` }}>
        {value}
      </p>
      <p className="mt-1 text-[11.5px] text-[var(--color-ink-faint)]">{caption}</p>
    </div>
  );
}
