import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { Badge, Card, DeptGauge, EmptyState, Spinner, acuityTone } from "../components/ui";
import { ManualIntakeForm } from "../components/ManualIntakeForm";
import { DepartmentQueueBoard } from "../components/DepartmentQueueBoard";

const MODE_TONE: Record<string, "good" | "warn" | "critical"> = {
  NORMAL: "good",
  HIGH_LOAD: "warn",
  CRITICAL: "critical",
};

export function Dashboard() {
  const { mutationTick, hospitalId } = useSession();
  const { data, loading, refetch } = usePoll(() => api.dashboard(hospitalId), 6000, [mutationTick, hospitalId]);
  const { data: patients, refetch: refetchPatients } = usePoll(() => api.listPatients(), 15000, [mutationTick]);
  const [showIntake, setShowIntake] = useState(false);

  if (loading && !data) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
      </div>
    );
  }
  if (!data) return <EmptyState title="Could not reach the hospital simulation service." />;

  const modeTone: "good" | "warn" | "critical" | "brand" = MODE_TONE[data.load.operating_mode] ?? "brand";

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-ink)]">Hospital overview</h1>
          <p className="mt-1 text-sm text-[var(--color-ink-faint)]">
            {data.scenario.title} · {data.time} · updated live
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge tone={modeTone} dot>
            {data.load.operating_mode.replace("_", " ")}
          </Badge>
          <Badge tone="neutral">λ {data.load.lambda.toFixed(2)}</Badge>
          <button
            onClick={() => setShowIntake(true)}
            className="rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--color-brand-600)] transition shadow-sm"
          >
            + Register patient
          </button>
        </div>
      </div>

      {data.load.operating_mode !== "NORMAL" && (
        <div className="rounded-xl border border-[var(--color-warn-100)] bg-[var(--color-warn-50)] px-4 py-3 text-sm text-[var(--color-warn-600)]">
          Hospital load is elevated ({Math.round(data.load.load_ratio * 100)}%). Escalation sensitivity is
          increased — uncertain or disagreeing cases are more readily flagged for human review.
        </div>
      )}

      <Card title="Department capacity" subtitle="Live bed occupancy across the hospital">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {data.departments.map((d) => (
            <DeptGauge key={d.name} {...d} />
          ))}
        </div>
      </Card>

      <Card
        title="Department queues"
        subtitle={`${data.waiting_count} waiting for triage · drag to reorder or move between departments`}
        right={
          <Link to="/live" className="text-xs font-medium text-[var(--color-brand-600)] hover:underline">
            Open Live Hospital →
          </Link>
        }
      >
        <DepartmentQueueBoard dash={data} hospitalId={hospitalId} onChanged={refetch} compact />
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Patients" subtitle="Chart-based patients you can look up and update" className="lg:col-span-2">
          {!patients || patients.length === 0 ? (
            <EmptyState title="No patient records loaded yet." />
          ) : (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {patients.map((p) => (
                <Link
                  key={p.patient_id}
                  to={`/patients/${p.patient_id}`}
                  className="rounded-xl border border-[var(--color-border)] p-3.5 hover:border-[var(--color-brand-500)] hover:shadow-sm"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-[var(--color-ink)]">Patient {p.patient_id}</p>
                    <Badge tone={acuityTone(p.acuity)}>Acuity {p.acuity ?? "—"}</Badge>
                  </div>
                  <p className="mt-1 text-xs text-[var(--color-ink-faint)]">
                    {p.age ? `${p.age}y ${p.sex}` : "—"} · {p.chief_complaint}
                  </p>
                </Link>
              ))}
            </div>
          )}
        </Card>

        <Card title="Recent activity" subtitle="Live hospital event feed">
          <ul className="space-y-1.5 text-xs text-[var(--color-ink-soft)]">
            {data.recent_events.length === 0 && <li className="text-[var(--color-ink-faint)]">No events yet.</li>}
            {[...data.recent_events].reverse().map((e, i) => (
              <li key={i} className="border-b border-[var(--color-border)] pb-1.5 last:border-0">
                {e}
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {showIntake && (
        <ManualIntakeForm
          onSuccess={() => { refetch(); refetchPatients(); }}
          onClose={() => setShowIntake(false)}
        />
      )}
    </div>
  );
}
