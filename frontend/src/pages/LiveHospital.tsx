import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { Badge, Button, Card, DEPT_LABELS, DeptGauge, EmptyState, RiskBar, Spinner, acuityLabel, acuityTone } from "../components/ui";
import { ManualIntakeForm } from "../components/ManualIntakeForm";
import { DepartmentQueueBoard } from "../components/DepartmentQueueBoard";
import type { TriageResult } from "../types";

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

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export function LiveHospital() {
  const { sessionId, proposeAction, mutationTick, hospitalId } = useSession();
  const { data: dash, loading, refetch } = usePoll(() => api.dashboard(hospitalId), 4000, [mutationTick, hospitalId]);
  const { data: scenarios } = usePoll(() => api.scenarios(hospitalId), 60000, [hospitalId]);
  const [busy, setBusy] = useState(false);
  // A triage result is only meaningful for the hospital whose queue it was
  // computed against, so its owning hospitalId is tracked alongside it —
  // `triageResult` below is derived at render time and reads as empty the
  // instant the selected hospital no longer matches, with no separate effect
  // needed to clear it.
  const [triageResultState, setTriageResultState] = useState<{ hospitalId: string; result: TriageResult } | null>(null);
  const [showIntake, setShowIntake] = useState(false);

  const triageResult = triageResultState?.hospitalId === hospitalId ? triageResultState.result : null;

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

  const triage = async (patientId: string) => {
    setBusy(true);
    try {
      const result = await api.triageSimulated(patientId, hospitalId);
      setTriageResultState({ hospitalId, result });
      await refetch();
    } finally { setBusy(false); }
  };

  const admit = async (patientId: string, department?: string) => {
    if (!sessionId) return;
    setBusy(true);
    try {
      const outcome = await proposeAction("admit_simulated_patient", {
        patient_id: patientId,
        ...(department ? { department } : {}),
        ...(hospitalId ? { hospital_id: hospitalId } : {}),
      });
      if (outcome.status === "executed") { setTriageResultState(null); await refetch(); }
    } finally { setBusy(false); }
  };

  const reorder = async (patientId: string, newIndex: number, note: string) => {
    setBusy(true);
    try { await api.reorderQueue(patientId, newIndex, note, hospitalId); await refetch(); }
    finally { setBusy(false); }
  };

  const fullQueue: QueuePatient[] = (dash.full_queue as unknown as QueuePatient[]) ?? [];
  const triagedPatients = fullQueue.filter(p => p.status === "TRIAGED");
  const waitingPatients = fullQueue.filter(p => p.status === "ARRIVED");

  const modeColor = {
    NORMAL: "text-emerald-600 bg-emerald-50 border-emerald-200",
    HIGH_LOAD: "text-amber-600 bg-amber-50 border-amber-200",
    CRITICAL: "text-red-600 bg-red-50 border-red-200",
  }[dash.load.operating_mode] ?? "text-blue-600 bg-blue-50 border-blue-200";

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-ink)]">Live Hospital</h1>
          <p className="mt-1 text-sm text-[var(--color-ink-faint)]">
            {dash.scenario.title} — {dash.scenario.description}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <span className={`rounded-lg border px-3 py-1.5 text-xs font-semibold ${modeColor}`}>
            {dash.load.operating_mode.replace("_", " ")} · λ {dash.load.lambda.toFixed(2)}
          </span>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.step(15, true, hospitalId))}>+15 min</Button>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.step(60, true, hospitalId))}>+60 min</Button>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.triggerArrival(undefined, hospitalId))}>Random arrival</Button>
          <Button size="sm" disabled={busy} onClick={() => setShowIntake(true)}>+ Register patient</Button>
        </div>
      </div>

      {/* ── Scenario Switcher ─────────────────────────────────────── */}
      <Card title="Scenario" subtitle="Switch hospital conditions — queue and occupancy update immediately">
        <div className="flex flex-wrap gap-2">
          {scenarios?.map((s) => (
            <button
              key={s.name}
              disabled={busy}
              onClick={() => run(() => api.loadScenario(s.name, hospitalId))}
              className={`rounded-lg border px-3 py-2 text-left text-xs transition ${
                s.name === dash.scenario.name
                  ? "border-[var(--color-brand-500)] bg-[var(--color-brand-50)] text-[var(--color-brand-700)]"
                  : "border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-brand-300)]"
              }`}
            >
              <p className="font-semibold">{s.title}</p>
              <p className="mt-0.5 max-w-[220px] text-[10px] text-[var(--color-ink-faint)]">{s.description}</p>
            </button>
          ))}
        </div>
      </Card>

      {/* ── Dept Capacity ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {dash.departments.map((d) => <DeptGauge key={d.name} {...d} />)}
      </div>

      {/* ── Department queues (full width — drag to reorder / override) ── */}
      <Card
        title="Triaged — awaiting admission"
        subtitle={`${triagedPatients.length} patient${triagedPatients.length !== 1 ? "s" : ""} assessed · drag to reorder or move between departments`}
      >
        <DepartmentQueueBoard dash={dash} hospitalId={hospitalId} onChanged={refetch} />
      </Card>

      {/* ── Main content grid ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">

        {/* ── Left col: queues ────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Waiting for triage */}
          <Card
            title="Waiting for triage"
            subtitle={`${waitingPatients.length} patient${waitingPatients.length !== 1 ? "s" : ""} in queue`}
          >
            {waitingPatients.length === 0 ? (
              <EmptyState title="ED queue is clear." subtitle='Use "Random arrival" or "+ Register patient" to add patients.' />
            ) : (
              <ul className="space-y-2">
                {waitingPatients.map((p, idx) => (
                  <li key={p.patient_id} className="flex items-start justify-between gap-2 rounded-xl border border-[var(--color-border)] p-3">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-bold text-[var(--color-ink-faint)] w-5 text-right">{idx + 1}</span>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge tone={acuityTone(p.acuity)}>{acuityLabel(p.acuity)}</Badge>
                          <p className="text-sm font-semibold text-[var(--color-ink)]">{p.patient_id}</p>
                          <span className="text-xs text-[var(--color-ink-faint)]">{p.age}y {p.sex}</span>
                          {hasHistory(p) && (
                            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">History</span>
                          )}
                          {Boolean(p.metadata?.queue_note) && (
                            <span className="rounded-full bg-blue-50 px-2 py-0.5 text-[10px] text-blue-600 italic">
                              "{p.metadata?.queue_note as string}"
                            </span>
                          )}
                        </div>
                        <p className="mt-0.5 text-xs text-[var(--color-ink-faint)]">{p.chief_complaint}</p>
                      </div>
                    </div>
                    <div className="flex gap-1.5 shrink-0">
                      <div className="flex flex-col gap-1">
                        <button disabled={busy || idx === 0} onClick={() => reorder(p.patient_id, idx - 1, "")}
                          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-raised)] disabled:opacity-30 transition">▲</button>
                        <button disabled={busy || idx === waitingPatients.length - 1} onClick={() => reorder(p.patient_id, idx + 1, "")}
                          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-ink-faint)] hover:bg-[var(--color-surface-raised)] disabled:opacity-30 transition">▼</button>
                      </div>
                      <Button size="sm" disabled={busy} onClick={() => triage(p.patient_id)}>Triage</Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>

        {/* ── Right col: event feed ──────────────────────────────── */}
        <div className="space-y-4">
          <Card title="Live event feed" subtitle="System and staff actions in order">
            <ul className="max-h-[480px] space-y-1.5 overflow-y-auto text-xs text-[var(--color-ink-soft)]">
              {[...dash.recent_events].reverse().map((e, i) => (
                <li key={i} className="border-b border-[var(--color-border)] pb-1.5 last:border-0">{e}</li>
              ))}
            </ul>
          </Card>
          <Card title="Queue summary">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-[var(--color-ink-faint)]">Waiting (untriaged)</span>
                <span className="font-bold text-amber-600">{waitingPatients.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--color-ink-faint)]">Triaged (pending admit)</span>
                <span className="font-bold text-blue-600">{triagedPatients.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--color-ink-faint)]">Admitted (in treatment)</span>
                <span className="font-bold text-emerald-600">{dash.admitted_count}</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* ── Triage result panel ────────────────────────────────────── */}
      {triageResult && (
        <TriageResultPanel result={triageResult} busy={busy} onAdmit={admit} onDismiss={() => setTriageResultState(null)} />
      )}

      {/* ── Modals ─────────────────────────────────────────────────── */}
      {showIntake && (
        <ManualIntakeForm onSuccess={refetch} onClose={() => setShowIntake(false)} />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Triage result panel (unchanged structure, kept here)
// ─────────────────────────────────────────────────────────────────────────────

function TriageResultPanel({
  result, busy, onAdmit, onDismiss,
}: {
  result: TriageResult;
  busy: boolean;
  onAdmit: (patientId: string, department?: string) => void;
  onDismiss: () => void;
}) {
  const clin = result.clinical_assessment;
  const op = result.operational_decision;
  const constrained = op.capacity_warning || op.clinical_department !== op.operational_department;

  return (
    <Card
      title={`Assessment — ${result.patient_id}`}
      right={<button onClick={onDismiss} className="text-xs text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]">Dismiss</button>}
    >
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">Clinical assessment</p>
          <RiskBar label="Admission risk" value={clin.reconciled_admission_risk} />
          <RiskBar label="ICU risk" value={clin.reconciled_icu_risk} />
          {clin.red_flags?.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {clin.red_flags.map((f, i) => <Badge key={i} tone="critical">{f}</Badge>)}
            </div>
          )}
          <p className="text-xs text-[var(--color-ink-soft)]">{clin.department_reasoning}</p>
          {clin.confidence_note && (
            <p className="text-[10px] text-[var(--color-ink-faint)] border-t border-[var(--color-border)] pt-1.5">{clin.confidence_note}</p>
          )}
        </div>
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">Routing decision</p>
          <div className="flex items-center gap-2">
            <div className="rounded-lg border border-[var(--color-border)] px-3 py-2 text-center">
              <p className="text-[10px] text-[var(--color-ink-faint)]">Clinically preferred</p>
              <p className="text-sm font-bold text-[var(--color-ink)]">{DEPT_LABELS[op.clinical_department] ?? op.clinical_department}</p>
            </div>
            {constrained && (
              <>
                <span className="text-[var(--color-ink-faint)]">→</span>
                <div className="rounded-lg border border-[var(--color-warn-100)] bg-[var(--color-warn-50)] px-3 py-2 text-center">
                  <p className="text-[10px] text-[var(--color-warn-600)]">Current allocation</p>
                  <p className="text-sm font-bold text-[var(--color-warn-600)]">{DEPT_LABELS[op.operational_department] ?? op.operational_department}</p>
                </div>
              </>
            )}
          </div>
          <p className="text-xs text-[var(--color-ink-soft)]">{op.recommendation_summary}</p>
          {op.confirmation_required && <Badge tone="warn" dot>Staff confirmation required</Badge>}
          <Button disabled={busy} onClick={() => onAdmit(result.patient_id, op.operational_department)}>
            {op.confirmation_required ? "Review & confirm admission" : `Admit to ${DEPT_LABELS[op.operational_department] ?? op.operational_department}`}
          </Button>
        </div>
      </div>
    </Card>
  );
}
