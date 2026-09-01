import { useMemo, useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { PatientDrawer, type DrawerVital } from "../components/PatientDrawer";
import { TriageModal } from "../components/TriageModal";
import { VitalsEditModal } from "../components/VitalsEditModal";
import { AdmissionConfirmModal } from "../components/AdmissionConfirmModal";
import { AcuityPill, DEPT_LABELS, EmptyState, Spinner, aiDeptOf, fmtWaitMinutes } from "../components/ui";
import type { OperationalDecision, PatientSummary, SimVitals } from "../types";

type QueueEntry = {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  acuity: number;
  status: string;
  arrival_time_min?: number;
  vitals?: SimVitals;
  operational_decision?: OperationalDecision | null;
};

/**
 * Table matches the mockup's columns exactly (Status/Acuity/AI Dept/Current
 * Dept/Waiting). /api/patients (chart-based records) carries none of those
 * fields — they only exist for patients the simulation has actually seen
 * (/api/simulation/dashboard's full_queue). Joined here by patient_id;
 * where no simulation record exists, those columns render "—" rather than
 * fabricating a department/status/wait time or dropping the patient.
 */
export function PatientList() {
  const { hospitalId } = useSession();
  const { data: patients, loading } = usePoll(() => api.listPatients(), 15000);
  const { data: dash, refetch: refetchDash } = usePoll(() => api.dashboard(hospitalId), 8000, [hospitalId]);
  const { data: hospitals } = usePoll(() => api.listHospitals(), 60000, []);
  const hospitalName = hospitals?.find((h) => h.hospital_id === hospitalId)?.hospital_name ?? hospitalId;
  const [search, setSearch] = useState("");
  const [drawerPatientId, setDrawerPatientId] = useState<string | null>(null);
  const [triageTarget, setTriageTarget] = useState<QueueEntry | null>(null);
  const [vitalsTarget, setVitalsTarget] = useState<string | null>(null);
  const [admitTarget, setAdmitTarget] = useState<{ patientId: string; department: string; reason: string } | null>(null);

  const fullQueue = ((dash?.full_queue as unknown as QueueEntry[]) ?? []);
  const byId = useMemo(() => new Map(fullQueue.map((p) => [p.patient_id, p])), [fullQueue]);

  const filtered = useMemo(() => {
    if (!patients) return patients;
    const q = search.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => p.patient_id.toLowerCase().includes(q) || p.chief_complaint.toLowerCase().includes(q));
  }, [patients, search]);

  const drawerChart = patients?.find((p) => p.patient_id === drawerPatientId) ?? null;
  const drawerSim = drawerPatientId ? byId.get(drawerPatientId) ?? null : null;
  const departmentOptions = dash?.departments.map((d) => d.name) ?? [];

  const COLS = "grid-cols-[90px_55px_1.6fr_90px_90px_1fr_1fr_90px]";

  return (
    <div className="mx-auto max-w-[1300px] space-y-3.5 p-[22px_24px]">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-[14px] font-bold text-[var(--color-ink)]">Patients — {hospitalName}</h1>
        {patients && patients.length > 0 && (
          <div className="flex shrink-0 items-center gap-2">
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by ID or complaint…"
              aria-label="Search patients"
              className="w-[260px] rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-xs text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-brand-400)]"
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-2 text-[11.5px] font-semibold text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-muted)]"
              >
                Clear
              </button>
            )}
          </div>
        )}
      </div>

      {loading && !patients ? (
        <div className="flex justify-center py-16">
          <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
        </div>
      ) : !patients || patients.length === 0 ? (
        <EmptyState
          title="No patient records loaded."
          subtitle="Ask the assistant to look up a patient, or ingest hospital records to populate this list."
        />
      ) : (
        <div className="overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className={`grid ${COLS} gap-2 bg-[var(--color-surface-muted)] px-4 py-[9px] text-[9.5px] font-bold uppercase tracking-[.04em] text-[var(--color-ink-faint)]`}>
            <span>ID</span><span>Age/Sex</span><span>Complaint</span><span>Status</span><span>Acuity</span><span>AI Dept</span><span>Current Dept</span><span>Waiting</span>
          </div>
          {filtered && filtered.length === 0 && (
            <div className="px-6 py-6 text-center text-xs text-[var(--color-ink-faint)]">No patients match your search.</div>
          )}
          {filtered!.map((p) => {
            const sim = byId.get(p.patient_id) ?? null;
            const acuity = sim?.acuity ?? p.acuity;
            const op = sim?.operational_decision;
            const waitLabel =
              sim?.arrival_time_min != null && dash ? fmtWaitMinutes(Math.max(0, dash.sim_time_minutes - sim.arrival_time_min)) : null;
            return (
              <button
                key={p.patient_id}
                onClick={() => setDrawerPatientId(p.patient_id)}
                className={`grid w-full ${COLS} items-center gap-2 border-t border-[var(--color-border)] px-4 py-2.5 text-left hover:bg-[var(--color-surface-muted)]`}
              >
                <span className="truncate font-mono text-xs font-semibold text-[var(--color-ink)]">{p.patient_id}</span>
                <span className="text-[11px] text-[var(--color-ink-faint)]">{p.age ?? "—"}{p.sex ?? ""}</span>
                <span className="truncate text-[11.5px] text-[var(--color-ink)]">{p.chief_complaint}</span>
                <span className="truncate text-[10.5px] capitalize text-[var(--color-ink-faint)]">{sim?.status?.toLowerCase() ?? "—"}</span>
                <span className="w-fit"><AcuityPill acuity={acuity} withLabel={false} /></span>
                <span className="truncate text-[11px] text-[var(--color-ink)]">{op ? (DEPT_LABELS[aiDeptOf(op)] ?? aiDeptOf(op)) : "—"}</span>
                <span className="truncate text-[11px] font-semibold text-[var(--color-ink)]">
                  {op ? (DEPT_LABELS[op.operational_department] ?? op.operational_department) : "—"}
                  {op?.nurse_override && <span style={{ color: "oklch(48% 0.15 300)" }}> ★</span>}
                </span>
                <span className="truncate font-mono text-[10.5px] text-[var(--color-ink-faint)]">{waitLabel ?? "—"}</span>
              </button>
            );
          })}
        </div>
      )}

      {drawerPatientId && drawerChart && (
        <PatientDrawer
          patientId={drawerPatientId}
          age={drawerChart.age}
          sex={drawerChart.sex}
          chiefComplaint={drawerChart.chief_complaint}
          acuity={drawerSim?.acuity ?? drawerChart.acuity}
          status={drawerSim?.status as "ARRIVED" | "TRIAGED" | "IN_TREATMENT" | "DISCHARGED" | undefined}
          decision={drawerSim?.operational_decision}
          vitals={drawerVitals(drawerChart, drawerSim)}
          onClose={() => setDrawerPatientId(null)}
          onTriage={drawerSim?.status === "ARRIVED" ? () => setTriageTarget(drawerSim) : undefined}
          onEditVitals={drawerSim?.status === "TRIAGED" ? () => setVitalsTarget(drawerPatientId) : undefined}
          onAdmit={
            drawerSim?.status === "TRIAGED" && drawerSim.operational_decision
              ? () => {
                  const op = drawerSim.operational_decision!;
                  setAdmitTarget({ patientId: drawerPatientId, department: op.operational_department, reason: op.recommendation_summary });
                }
              : undefined
          }
        />
      )}

      {triageTarget && (
        <TriageModal
          patientId={triageTarget.patient_id}
          age={triageTarget.age}
          sex={triageTarget.sex}
          chiefComplaint={triageTarget.chief_complaint}
          vitals={triageTarget.vitals ?? {}}
          hospitalId={hospitalId}
          departmentOptions={departmentOptions}
          onClose={() => setTriageTarget(null)}
          onDone={() => { setTriageTarget(null); refetchDash(); }}
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
            refetchDash();
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
          onSuccess={() => { setAdmitTarget(null); setDrawerPatientId(null); refetchDash(); }}
        />
      )}
    </div>
  );
}

function drawerVitals(p: PatientSummary, sim: QueueEntry | null): DrawerVital[] {
  if (sim?.vitals) {
    return [
      { label: "HR", value: sim.vitals.hr ?? null },
      { label: "SpO₂", value: sim.vitals.spo2 != null ? `${sim.vitals.spo2}%` : null },
      { label: "RR", value: sim.vitals.rr ?? null },
      { label: "BP", value: sim.vitals.sbp != null && sim.vitals.dbp != null ? `${sim.vitals.sbp}/${sim.vitals.dbp}` : null },
      { label: "Temp", value: sim.vitals.temp ?? null },
      { label: "Pain", value: sim.vitals.pain ?? null },
    ];
  }
  return [
    { label: "HR", value: p.vitals.heart_rate },
    { label: "SpO₂", value: p.vitals.spo2 != null ? `${p.vitals.spo2}%` : null },
    { label: "RR", value: p.vitals.resp_rate },
    { label: "BP", value: p.vitals.sbp != null && p.vitals.dbp != null ? `${p.vitals.sbp}/${p.vitals.dbp}` : null },
    { label: "Temp", value: p.vitals.temperature },
    { label: "Pain", value: p.vitals.pain_score },
  ];
}

