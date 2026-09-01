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

type PatientRow = {
  patient_id: string;
  age: number | null;
  sex: string | null;
  chief_complaint: string;
  acuity: number | null;
  chart: PatientSummary | null;
  sim: QueueEntry | null;
};

type TriageTarget = {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  vitals: SimVitals;
  /** Set only for a chart-only row with no live simulation record — see TriageModal's `activation` prop. */
  activation?: { chiefComplaint: string; age: number; sex: string; acuity: number | null };
};

// A chart-only patient needs real age/sex/complaint to be brought into the
// live queue via manualArrival() — never fabricate a placeholder value, so
// Triage Patient simply isn't offered for the (rare) record missing one.
function canActivate(row: PatientRow): boolean {
  return row.age != null && row.sex != null && row.chief_complaint.trim() !== "";
}

/**
 * Table matches the mockup's columns exactly (Status/Acuity/AI Dept/Current
 * Dept/Waiting). Two real, independent sources feed this list: /api/patients
 * (static chart records) and /api/simulation/dashboard's full_queue (patients
 * the live simulation actually knows about — created via Register Patient,
 * Random Arrival, or a scenario, e.g. PAT-101). Neither is a subset of the
 * other, so the table is their union, deduped by patient_id. Where a patient
 * exists in both, the live simulation record wins (more current/authoritative
 * — status/acuity/complaint/demographics all prefer `sim` over `chart`).
 * Where a patient exists only in the simulation, chart-equivalent display
 * fields (age/sex/complaint) come straight off that same live record instead
 * of being left blank. Where no simulation record exists at all, Status/AI
 * Dept/Current Dept/Waiting render "—" rather than fabricating them.
 */
export function PatientList() {
  const { hospitalId } = useSession();
  const { data: patients, loading } = usePoll(() => api.listPatients(), 15000);
  const { data: dash, refetch: refetchDash } = usePoll(() => api.dashboard(hospitalId), 8000, [hospitalId]);
  const { data: hospitals } = usePoll(() => api.listHospitals(), 60000, []);
  const hospitalName = hospitals?.find((h) => h.hospital_id === hospitalId)?.hospital_name ?? hospitalId;
  const [search, setSearch] = useState("");
  const [drawerPatientId, setDrawerPatientId] = useState<string | null>(null);
  const [triageTarget, setTriageTarget] = useState<TriageTarget | null>(null);
  const [vitalsTarget, setVitalsTarget] = useState<string | null>(null);
  const [admitTarget, setAdmitTarget] = useState<{ patientId: string; department: string; reason: string } | null>(null);

  const fullQueue = ((dash?.full_queue as unknown as QueueEntry[]) ?? []);
  const byId = useMemo(() => new Map(fullQueue.map((p) => [p.patient_id, p])), [fullQueue]);
  const chartById = useMemo(() => new Map((patients ?? []).map((p) => [p.patient_id, p])), [patients]);

  const rows: PatientRow[] = useMemo(() => {
    const ids = new Set<string>([...chartById.keys(), ...fullQueue.map((p) => p.patient_id)]);
    return [...ids].map((id) => {
      const chart = chartById.get(id) ?? null;
      const sim = byId.get(id) ?? null;
      return {
        patient_id: id,
        age: sim?.age ?? chart?.age ?? null,
        sex: sim?.sex ?? chart?.sex ?? null,
        chief_complaint: sim?.chief_complaint ?? chart?.chief_complaint ?? "",
        acuity: sim?.acuity ?? chart?.acuity ?? null,
        chart,
        sim,
      };
    });
  }, [chartById, fullQueue, byId]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) => r.patient_id.toLowerCase().includes(q) || r.chief_complaint.toLowerCase().includes(q));
  }, [rows, search]);

  const drawerRow = drawerPatientId ? rows.find((r) => r.patient_id === drawerPatientId) ?? null : null;
  const drawerSim = drawerRow?.sim ?? null;
  const departmentOptions = dash?.departments.map((d) => d.name) ?? [];

  const COLS = "grid-cols-[90px_55px_1.6fr_90px_90px_1fr_1fr_90px]";

  return (
    <div className="mx-auto max-w-[1300px] space-y-3.5 p-[22px_24px]">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-[14px] font-bold text-[var(--color-ink)]">Patients — {hospitalName}</h1>
        {rows.length > 0 && (
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

      {loading && !patients && rows.length === 0 ? (
        <div className="flex justify-center py-16">
          <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
        </div>
      ) : rows.length === 0 ? (
        <EmptyState
          title="No patient records loaded."
          subtitle="Ask the assistant to look up a patient, or ingest hospital records to populate this list."
        />
      ) : (
        <div className="overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className={`grid ${COLS} gap-2 bg-[var(--color-surface-muted)] px-4 py-[9px] text-[9.5px] font-bold uppercase tracking-[.04em] text-[var(--color-ink-faint)]`}>
            <span>ID</span><span>Age/Sex</span><span>Complaint</span><span>Status</span><span>Acuity</span><span>AI Dept</span><span>Current Dept</span><span>Waiting</span>
          </div>
          {filtered.length === 0 && (
            <div className="px-6 py-6 text-center text-xs text-[var(--color-ink-faint)]">No patients match your search.</div>
          )}
          {filtered.map((row) => {
            const sim = row.sim;
            const op = sim?.operational_decision;
            const waitLabel =
              sim?.arrival_time_min != null && dash ? fmtWaitMinutes(Math.max(0, dash.sim_time_minutes - sim.arrival_time_min)) : null;
            return (
              <button
                key={row.patient_id}
                onClick={() => setDrawerPatientId(row.patient_id)}
                className={`grid w-full ${COLS} items-center gap-2 border-t border-[var(--color-border)] px-4 py-2.5 text-left hover:bg-[var(--color-surface-muted)]`}
              >
                <span className="truncate font-mono text-xs font-semibold text-[var(--color-ink)]">{row.patient_id}</span>
                <span className="text-[11px] text-[var(--color-ink-faint)]">{row.age ?? "—"}{row.sex ?? ""}</span>
                <span className="truncate text-[11.5px] text-[var(--color-ink)]">{row.chief_complaint}</span>
                <span className="truncate text-[10.5px] capitalize text-[var(--color-ink-faint)]">{sim?.status?.toLowerCase() ?? "—"}</span>
                <span className="w-fit"><AcuityPill acuity={row.acuity} withLabel={false} /></span>
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

      {drawerPatientId && drawerRow && (
        <PatientDrawer
          patientId={drawerPatientId}
          age={drawerRow.age}
          sex={drawerRow.sex}
          chiefComplaint={drawerRow.chief_complaint}
          acuity={drawerRow.acuity}
          status={drawerSim?.status as "ARRIVED" | "TRIAGED" | "IN_TREATMENT" | "DISCHARGED" | undefined}
          decision={drawerSim?.operational_decision}
          vitals={drawerVitals(drawerRow.chart, drawerSim)}
          onClose={() => setDrawerPatientId(null)}
          onTriage={
            drawerSim?.status === "ARRIVED"
              ? () =>
                  setTriageTarget({
                    patient_id: drawerSim.patient_id,
                    age: drawerSim.age,
                    sex: drawerSim.sex,
                    chief_complaint: drawerSim.chief_complaint,
                    vitals: drawerSim.vitals ?? {},
                  })
              : !drawerSim && drawerRow && canActivate(drawerRow)
                ? () =>
                    setTriageTarget({
                      patient_id: drawerRow.patient_id,
                      age: drawerRow.age!,
                      sex: drawerRow.sex!,
                      chief_complaint: drawerRow.chief_complaint,
                      vitals: toSimVitals(drawerRow.chart),
                      activation: { chiefComplaint: drawerRow.chief_complaint, age: drawerRow.age!, sex: drawerRow.sex!, acuity: drawerRow.acuity },
                    })
                : undefined
          }
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
          activation={triageTarget.activation}
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

// `p` is null for a patient that exists only in the live simulation with no
// matching chart record (e.g. PAT-101 created via Random Arrival) — sim.vitals
// covers that case, so a missing chart is only a problem if sim has no
// vitals either, which falls through to explicit "—" rather than a crash.
function drawerVitals(p: PatientSummary | null, sim: QueueEntry | null): DrawerVital[] {
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
  if (p) {
    return [
      { label: "HR", value: p.vitals.heart_rate },
      { label: "SpO₂", value: p.vitals.spo2 != null ? `${p.vitals.spo2}%` : null },
      { label: "RR", value: p.vitals.resp_rate },
      { label: "BP", value: p.vitals.sbp != null && p.vitals.dbp != null ? `${p.vitals.sbp}/${p.vitals.dbp}` : null },
      { label: "Temp", value: p.vitals.temperature },
      { label: "Pain", value: p.vitals.pain_score },
    ];
  }
  return [
    { label: "HR", value: null }, { label: "SpO₂", value: null }, { label: "RR", value: null },
    { label: "BP", value: null }, { label: "Temp", value: null }, { label: "Pain", value: null },
  ];
}

// TriageModal's vitals display expects SimVitals' short keys; the chart
// record carries PatientSummary's long keys — purely a display mapping, the
// same real numbers either way (manualArrival() re-derives its own vitals
// server-side from the chart file regardless of what's shown here).
function toSimVitals(p: PatientSummary | null): SimVitals {
  if (!p) return {};
  return {
    hr: p.vitals.heart_rate ?? undefined,
    rr: p.vitals.resp_rate ?? undefined,
    spo2: p.vitals.spo2 ?? undefined,
    sbp: p.vitals.sbp ?? undefined,
    dbp: p.vitals.dbp ?? undefined,
    temp: p.vitals.temperature ?? undefined,
    pain: p.vitals.pain_score ?? undefined,
  };
}

