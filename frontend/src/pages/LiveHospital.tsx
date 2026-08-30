import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { Badge, Button, Card, DEPT_LABELS, DeptGauge, EmptyState, RiskBar, Spinner, acuityLabel, acuityTone } from "../components/ui";
import { ManualIntakeForm } from "../components/ManualIntakeForm";
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

const DEPT_GROUP_ORDER = ["ICU", "CICU", "ADMITTED_GEN", "ED_OBS", "DISCHARGE"];

function getDeptForPatient(p: QueuePatient): string {
  const op = p.operational_decision as Record<string, string> | null | undefined;
  const ca = p.clinical_assessment as Record<string, string> | null | undefined;
  return op?.clinical_department ?? ca?.department ?? "UNKNOWN";
}

function groupByDept(patients: QueuePatient[]): Map<string, QueuePatient[]> {
  const map = new Map<string, QueuePatient[]>();
  for (const p of patients) {
    const dept = getDeptForPatient(p);
    if (!map.has(dept)) map.set(dept, []);
    map.get(dept)!.push(p);
  }
  // Sort keys by clinical severity
  const sorted = new Map<string, QueuePatient[]>();
  for (const key of DEPT_GROUP_ORDER) {
    if (map.has(key)) sorted.set(key, map.get(key)!);
  }
  for (const [k, v] of map) {
    if (!sorted.has(k)) sorted.set(k, v);
  }
  return sorted;
}

function hasHistory(p: QueuePatient): boolean {
  return !!(p.metadata as Record<string, unknown> | null)?.has_history;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export function LiveHospital() {
  const { sessionId, proposeAction, mutationTick } = useSession();
  const { data: dash, loading, refetch } = usePoll(() => api.dashboard(), 4000, [mutationTick]);
  const { data: scenarios } = usePoll(() => api.scenarios(), 60000);
  const [busy, setBusy] = useState(false);
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);
  const [showIntake, setShowIntake] = useState(false);
  const [reorderModal, setReorderModal] = useState<QueuePatient | null>(null);

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
      const result = await api.triageSimulated(patientId);
      setTriageResult(result);
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
      });
      if (outcome.status === "executed") { setTriageResult(null); await refetch(); }
    } finally { setBusy(false); }
  };

  const reorder = async (patientId: string, newIndex: number, note: string) => {
    setBusy(true);
    try { await api.reorderQueue(patientId, newIndex, note); await refetch(); }
    finally { setBusy(false); }
  };

  const fullQueue: QueuePatient[] = (dash as Record<string, unknown>).full_queue as QueuePatient[] ?? [];
  const triagedPatients = fullQueue.filter(p => p.status === "TRIAGED");
  const waitingPatients = fullQueue.filter(p => p.status === "ARRIVED");
  const triagedByDept = groupByDept(triagedPatients);

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
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.step(15))}>+15 min</Button>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.step(60))}>+60 min</Button>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.triggerArrival())}>Random arrival</Button>
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
              onClick={() => run(() => api.loadScenario(s.name))}
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

      {/* ── Main content grid ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">

        {/* ── Left col: queues ────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Triaged → grouped by destination dept */}
          <Card
            title="Triaged — awaiting admission"
            subtitle={`${triagedPatients.length} patient${triagedPatients.length !== 1 ? "s" : ""} assessed`}
          >
            {triagedPatients.length === 0 ? (
              <EmptyState title="No triaged patients." subtitle="Run Triage on waiting patients to see results here." />
            ) : (
              <div className="space-y-4">
                {[...triagedByDept.entries()].map(([dept, patients]) => (
                  <div key={dept}>
                    <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-[var(--color-ink-faint)]">
                      {DEPT_LABELS[dept] ?? dept}
                    </p>
                    <ul className="space-y-2">
                      {patients.map((p) => {
                        const ca = p.clinical_assessment as Record<string, unknown> | null;
                        const op = p.operational_decision as Record<string, unknown> | null;
                        const constrained = op?.capacity_warning || op?.clinical_department !== op?.operational_department;
                        return (
                          <li key={p.patient_id} className="rounded-xl border border-[var(--color-border)] p-3 space-y-2">
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge tone={acuityTone(p.acuity)}>{acuityLabel(p.acuity)}</Badge>
                                <p className="text-sm font-semibold text-[var(--color-ink)]">{p.patient_id}</p>
                                <span className="text-xs text-[var(--color-ink-faint)]">{p.age}y {p.sex}</span>
                                {hasHistory(p) && (
                                  <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
                                    History
                                  </span>
                                )}
                              </div>
                              <div className="flex gap-1.5 shrink-0">
                                <button
                                  className="rounded-md border border-[var(--color-border)] px-2 py-1 text-xs text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-raised)] transition"
                                  disabled={busy}
                                  onClick={() => setReorderModal(p)}
                                >
                                  ↕ Reorder
                                </button>
                                <Button size="sm" disabled={busy}
                                  onClick={() => admit(p.patient_id, (op?.operational_department ?? "") as string)}>
                                  {op?.confirmation_required ? "Review & Admit" : `Admit →`}
                                </Button>
                              </div>
                            </div>
                            <p className="text-xs text-[var(--color-ink-soft)]">{p.chief_complaint}</p>
                            {ca && (
                              <div className="grid grid-cols-2 gap-2">
                                <RiskBar label="Admission risk" value={ca.reconciled_admission_risk as number} />
                                <RiskBar label="ICU risk" value={ca.reconciled_icu_risk as number} />
                              </div>
                            )}
                            {constrained && (
                              <div className="flex items-center gap-1.5 text-[10px] text-amber-600">
                                <span>⚠</span>
                                <span>Clinically: <strong>{DEPT_LABELS[(op?.clinical_department as string)] ?? op?.clinical_department}</strong> → Allocated: <strong>{DEPT_LABELS[(op?.operational_department as string)] ?? op?.operational_department}</strong></span>
                              </div>
                            )}
                            {ca?.confidence_note && (
                              <p className="text-[10px] text-[var(--color-ink-faint)] border-t border-[var(--color-border)] pt-1.5">
                                {ca.confidence_note as string}
                              </p>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ))}
              </div>
            )}
          </Card>

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
                          {p.metadata?.queue_note && (
                            <span className="rounded-full bg-blue-50 px-2 py-0.5 text-[10px] text-blue-600 italic">
                              "{p.metadata.queue_note as string}"
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
        <TriageResultPanel result={triageResult} busy={busy} onAdmit={admit} onDismiss={() => setTriageResult(null)} />
      )}

      {/* ── Modals ─────────────────────────────────────────────────── */}
      {showIntake && (
        <ManualIntakeForm onSuccess={refetch} onClose={() => setShowIntake(false)} />
      )}
      {reorderModal && (
        <ReorderModal
          patient={reorderModal}
          queueLength={fullQueue.length}
          currentIndex={fullQueue.findIndex(p => p.patient_id === reorderModal.patient_id)}
          onReorder={(newIdx, note) => { reorder(reorderModal.patient_id, newIdx, note); setReorderModal(null); }}
          onClose={() => setReorderModal(null)}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Reorder Modal
// ─────────────────────────────────────────────────────────────────────────────

function ReorderModal({
  patient, queueLength, currentIndex, onReorder, onClose,
}: {
  patient: QueuePatient;
  queueLength: number;
  currentIndex: number;
  onReorder: (newIndex: number, note: string) => void;
  onClose: () => void;
}) {
  const [newIdx, setNewIdx] = useState(currentIndex);
  const [note, setNote] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="w-full max-w-sm rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl space-y-4">
        <div className="flex items-start justify-between">
          <h2 className="text-base font-bold text-[var(--color-ink)]">Reorder patient</h2>
          <button onClick={onClose} className="text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]">✕</button>
        </div>
        <p className="text-sm text-[var(--color-ink-soft)]">
          <strong>{patient.patient_id}</strong> — {patient.chief_complaint}
        </p>
        <div className="space-y-1">
          <label className="text-xs font-semibold text-[var(--color-ink-soft)] uppercase tracking-wide">
            New position (1–{queueLength})
          </label>
          <input
            type="number" min={0} max={queueLength - 1}
            value={newIdx + 1}
            onChange={e => setNewIdx(Math.max(0, Math.min(queueLength - 1, Number(e.target.value) - 1)))}
            className="w-full rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-brand-400)]"
          />
          <p className="text-[10px] text-[var(--color-ink-faint)]">Current: position {currentIndex + 1}</p>
        </div>
        <div className="space-y-1">
          <label className="text-xs font-semibold text-[var(--color-ink-soft)] uppercase tracking-wide">Clinical note (optional)</label>
          <input
            placeholder="e.g. Clinician override — deteriorating"
            value={note}
            onChange={e => setNote(e.target.value)}
            className="w-full rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-brand-400)]"
          />
        </div>
        <div className="flex gap-2 pt-1">
          <button onClick={onClose}
            className="flex-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-raised)] transition">
            Cancel
          </button>
          <button onClick={() => onReorder(newIdx, note)}
            className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--color-brand-600)] transition">
            Move patient
          </button>
        </div>
      </div>
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
