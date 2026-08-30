import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { Badge, Button, Card, DEPT_LABELS, DeptGauge, EmptyState, RiskBar, Spinner, acuityLabel, acuityTone } from "../components/ui";
import type { TriageResult } from "../types";

export function LiveHospital() {
  const { sessionId, proposeAction, mutationTick } = useSession();
  const { data: dash, loading, refetch } = usePoll(() => api.dashboard(), 5000, [mutationTick]);
  const { data: scenarios } = usePoll(() => api.scenarios(), 60000);
  const [busy, setBusy] = useState(false);
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);

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
    try {
      await fn();
      await refetch();
    } finally {
      setBusy(false);
    }
  };

  const triage = async (patientId: string) => {
    setBusy(true);
    try {
      const result = await api.triageSimulated(patientId);
      setTriageResult(result);
      await refetch();
    } finally {
      setBusy(false);
    }
  };

  const admit = async (patientId: string, department?: string) => {
    if (!sessionId) return;
    setBusy(true);
    try {
      const outcome = await proposeAction("admit_simulated_patient", {
        patient_id: patientId,
        ...(department ? { department } : {}),
      });
      if (outcome.status === "executed") {
        setTriageResult(null);
        await refetch();
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-ink)]">Live Hospital</h1>
          <p className="mt-1 text-sm text-[var(--color-ink-faint)]">
            {dash.scenario.title} — {dash.scenario.description}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.step(15))}>
            + 15 min
          </Button>
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => run(() => api.step(60))}>
            + 60 min
          </Button>
          <Button size="sm" disabled={busy} onClick={() => run(() => api.triggerArrival())}>
            New arrival
          </Button>
        </div>
      </div>

      <Card title="Scenario" subtitle="Switch hospital operating conditions to see how allocation changes">
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

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {dash.departments.map((d) => (
          <DeptGauge key={d.name} {...d} />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="ED waiting queue" subtitle={`${dash.waiting_count} waiting · ${dash.admitted_count} admitted`}>
          {dash.waiting_queue.length === 0 ? (
            <EmptyState title="ED queue is clear." subtitle="Trigger a new arrival to continue the demo." />
          ) : (
            <ul className="space-y-2">
              {dash.waiting_queue.map((p) => (
                <li key={p.patient_id} className="rounded-lg border border-[var(--color-border)] p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-semibold text-[var(--color-ink)]">{p.patient_id}</p>
                        <Badge tone={acuityTone(p.acuity)}>{acuityLabel(p.acuity)}</Badge>
                      </div>
                      <p className="mt-0.5 text-xs text-[var(--color-ink-faint)]">{p.chief_complaint}</p>
                    </div>
                    <Button size="sm" disabled={busy} onClick={() => triage(p.patient_id)}>
                      Triage
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Live event feed" subtitle="What the system and staff have done, in order">
          <ul className="max-h-[420px] space-y-1.5 overflow-y-auto text-xs text-[var(--color-ink-soft)]">
            {[...dash.recent_events].reverse().map((e, i) => (
              <li key={i} className="border-b border-[var(--color-border)] pb-1.5 last:border-0">
                {e}
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {triageResult && (
        <TriageResultPanel result={triageResult} busy={busy} onAdmit={admit} onDismiss={() => setTriageResult(null)} />
      )}
    </div>
  );
}

function TriageResultPanel({
  result,
  busy,
  onAdmit,
  onDismiss,
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
      right={
        <button onClick={onDismiss} className="text-xs text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]">
          Dismiss
        </button>
      }
    >
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">
            Clinical assessment
          </p>
          <RiskBar label="Admission risk" value={clin.reconciled_admission_risk} />
          <RiskBar label="ICU risk" value={clin.reconciled_icu_risk} />
          {clin.red_flags?.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {clin.red_flags.map((f, i) => (
                <Badge key={i} tone="critical">
                  {f}
                </Badge>
              ))}
            </div>
          )}
          <p className="text-xs text-[var(--color-ink-soft)]">{clin.department_reasoning}</p>
        </div>

        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">
            Routing decision
          </p>
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
          {op.confirmation_required && (
            <Badge tone="warn" dot>
              Staff confirmation required
            </Badge>
          )}
          <Button
            disabled={busy}
            onClick={() => onAdmit(result.patient_id, op.operational_department)}
          >
            {op.confirmation_required
              ? "Review & confirm admission"
              : `Admit to ${DEPT_LABELS[op.operational_department] ?? op.operational_department}`}
          </Button>
        </div>
      </div>
    </Card>
  );
}
