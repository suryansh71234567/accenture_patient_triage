import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import {
  Button,
  EmptyState,
  Spinner,
  fmtWaitMinutes,
  formatOperatingMode,
} from "../components/ui";
import { ManualIntakeForm } from "../components/ManualIntakeForm";
import { DepartmentQueueBoard } from "../components/DepartmentQueueBoard";
import { TriageModal } from "../components/TriageModal";
import type { SimVitals } from "../types";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

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
  operational_decision?: Record<string, unknown> | null;
  metadata?: Record<string, unknown> | null;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function hasHistory(p: QueuePatient): boolean {
  return !!(p.metadata as Record<string, unknown> | null)?.has_history;
}

const MODE_BADGE_STYLE: Record<string, { color: string; background: string; borderColor: string }> = {
  NORMAL: { color: "var(--color-good-600)", background: "var(--color-good-50)", borderColor: "var(--color-good-100)" },
  HIGH_LOAD: { color: "var(--color-warn-600)", background: "var(--color-warn-50)", borderColor: "var(--color-warn-100)" },
  CRITICAL: { color: "var(--color-critical-600)", background: "var(--color-critical-50)", borderColor: "var(--color-critical-100)" },
};
const DEFAULT_MODE_BADGE_STYLE = { color: "var(--color-brand-600)", background: "var(--color-brand-50)", borderColor: "var(--color-brand-100)" };

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export function LiveHospital() {
  const { hospitalId } = useSession();
  const { data: dash, loading, refetch } = usePoll(() => api.dashboard(hospitalId), 4000, [hospitalId]);
  const { data: scenarios } = usePoll(() => api.scenarios(hospitalId), 60000, [hospitalId]);
  const { data: hospitals } = usePoll(() => api.listHospitals(), 60000, []);
  const hospitalName = hospitals?.find((h) => h.hospital_id === hospitalId)?.hospital_name ?? hospitalId;
  const [busy, setBusy] = useState(false);
  const [triagePatient, setTriagePatient] = useState<QueuePatient | null>(null);
  const [showIntake, setShowIntake] = useState(false);

  if (loading && !dash) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
      </div>
    );
  }
  if (!dash) return <EmptyState title="Could not reach the hospital simulation service." />;

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); await refetch(); } finally { setBusy(false); }
  };

  const reorder = (patientId: string, newIndex: number) =>
    run(() => api.reorderQueue(patientId, newIndex, "", hospitalId));

  const modeStyle = MODE_BADGE_STYLE[dash.load.operating_mode] ?? DEFAULT_MODE_BADGE_STYLE;

  const fullQueue: QueuePatient[] = (dash.full_queue as unknown as QueuePatient[]) ?? [];
  const triagedPatients = fullQueue.filter(p => p.status === "TRIAGED");
  const waitingPatients = fullQueue.filter(p => p.status === "ARRIVED");
  const departmentOptions = dash.departments.map((d) => d.name);

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-6">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <h1 className="text-[14px] font-bold text-[var(--color-ink)]">Live Hospital — {hospitalName}</h1>
        <div className="flex items-center gap-2">
          <Button size="sm" disabled={busy} onClick={() => setShowIntake(true)}>+ Register Patient</Button>
          <span className="text-[10.5px] text-[var(--color-ink-faint)]">or</span>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.triggerArrival(undefined, hospitalId))}>Random Arrival</Button>
        </div>
      </div>

      {/* ── Scenario Switcher — compact pill row, matching the mock ── */}
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3.5 py-2.5">
        <span className="text-[10.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Scenario</span>
        {scenarios?.map((s) => (
          <button
            key={s.name}
            disabled={busy}
            title={s.description}
            onClick={() => run(() => api.loadScenario(s.name, hospitalId))}
            className={`rounded-full border px-3 py-1.5 text-[11px] font-bold transition ${
              s.name === dash.scenario.name
                ? "border-[var(--color-brand-500)] bg-[var(--color-brand-50)] text-[var(--color-brand-700)]"
                : "border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-brand-300)]"
            }`}
          >
            {s.title}
          </button>
        ))}
        <span className="flex-1" />
        <span className="whitespace-nowrap text-[10.5px] text-[var(--color-ink-faint)]">
          Load ratio <b className="font-mono text-[var(--color-ink)]">{dash.load.lambda.toFixed(2)}×</b> —{" "}
          <span className="font-bold" style={{ color: modeStyle.color }}>{formatOperatingMode(dash.load.operating_mode)}</span>
        </span>
      </div>

      {/* ── Waiting for triage rail ─────────────────────────────────── */}
      <div>
        <p className="mb-2 text-[11.5px] font-bold uppercase tracking-wide" style={{ color: "var(--color-warn-500)" }}>
          Waiting for Triage · {waitingPatients.length}
        </p>
        {waitingPatients.length === 0 ? (
          <EmptyState title="ED queue is clear." subtitle='Use "Random Arrival" or "+ Register Patient" to add patients.' />
        ) : (
          <div className="flex gap-3 overflow-x-auto pb-1.5">
            {waitingPatients.map((p, idx) => (
              <div
                key={p.patient_id}
                style={{ borderColor: "oklch(85% 0.03 45)" }}
                className="flex w-[270px] shrink-0 flex-col gap-2 rounded-xl border bg-[var(--color-surface)] p-3.5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[12.5px] font-semibold text-[var(--color-ink)]">{p.patient_id}</span>
                  <div className="flex items-center gap-1">
                    <span className="text-[10.5px] text-[var(--color-ink-faint)]">{p.age}{p.sex}</span>
                    <button
                      disabled={busy || idx === 0}
                      title="Move earlier in queue"
                      onClick={() => reorder(p.patient_id, idx - 1)}
                      className="rounded border border-[var(--color-border)] px-1 text-[9px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-muted)] disabled:opacity-30"
                    >◀</button>
                    <button
                      disabled={busy || idx === waitingPatients.length - 1}
                      title="Move later in queue"
                      onClick={() => reorder(p.patient_id, idx + 1)}
                      className="rounded border border-[var(--color-border)] px-1 text-[9px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-muted)] disabled:opacity-30"
                    >▶</button>
                  </div>
                </div>
                <p className="line-clamp-2 text-xs text-[var(--color-ink)]">{p.chief_complaint}</p>
                {p.vitals && (
                  <div className="flex gap-2.5 font-mono text-[10.5px] text-[var(--color-ink-soft)]">
                    {p.vitals.hr != null && <span>HR {p.vitals.hr}</span>}
                    {p.vitals.spo2 != null && <span>SpO₂ {p.vitals.spo2}%</span>}
                    {p.vitals.sbp != null && p.vitals.dbp != null && <span>{p.vitals.sbp}/{p.vitals.dbp}</span>}
                  </div>
                )}
                <div className="flex flex-wrap items-center gap-1.5">
                  {hasHistory(p) && (
                    <span className="rounded-full bg-[var(--color-good-50)] px-2 py-0.5 text-[10px] font-semibold text-[var(--color-good-600)]">History</span>
                  )}
                  {Boolean(p.metadata?.queue_note) && (
                    <span className="rounded-full bg-[var(--color-brand-50)] px-2 py-0.5 text-[10px] italic text-[var(--color-brand-600)]">
                      "{p.metadata?.queue_note as string}"
                    </span>
                  )}
                </div>
                <div className="mt-0.5 flex items-center justify-between">
                  <span className="text-[10.5px] font-semibold" style={{ color: "var(--color-warn-500)" }}>
                    {p.arrival_time_min != null
                      ? `waiting ${fmtWaitMinutes(Math.max(0, dash.sim_time_minutes - p.arrival_time_min))}`
                      : ""}
                  </span>
                  <Button size="sm" disabled={busy} onClick={() => setTriagePatient(p)}>Triage</Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Department queues — no outer card, matching the mock: each
          column already carries its own border/background via
          DepartmentQueueBoard, so an enclosing card would just add a
          second, non-mock layer of chrome around it. ── */}
      <div>
        <p className="mb-2 text-[11.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">
          Triaged · {triagedPatients.length} awaiting admission
        </p>
        <DepartmentQueueBoard dash={dash} hospitalId={hospitalId} onChanged={refetch} />
      </div>

      {/* ── Modals ─────────────────────────────────────────────────── */}
      {showIntake && (
        <ManualIntakeForm onSuccess={refetch} onClose={() => setShowIntake(false)} />
      )}

      {triagePatient && (
        <TriageModal
          patientId={triagePatient.patient_id}
          age={triagePatient.age}
          sex={triagePatient.sex}
          chiefComplaint={triagePatient.chief_complaint}
          vitals={triagePatient.vitals ?? {}}
          hospitalId={hospitalId}
          departmentOptions={departmentOptions}
          onClose={() => setTriagePatient(null)}
          onDone={() => { setTriagePatient(null); refetch(); }}
        />
      )}
    </div>
  );
}
