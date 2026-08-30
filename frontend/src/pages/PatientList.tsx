import { Link } from "react-router-dom";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { Badge, Card, EmptyState, Spinner, acuityLabel, acuityTone } from "../components/ui";

export function PatientList() {
  const { data: patients, loading } = usePoll(() => api.listPatients(), 15000);

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <div>
        <h1 className="text-2xl font-bold text-[var(--color-ink)]">Patients</h1>
        <p className="mt-1 text-sm text-[var(--color-ink-faint)]">
          Look up a patient, review their assessment, or add a new observation.
        </p>
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
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {patients.map((p) => (
            <Link key={p.patient_id} to={`/patients/${p.patient_id}`}>
              <Card className="h-full transition hover:border-[var(--color-brand-500)] hover:shadow-md">
                <div className="flex items-center justify-between">
                  <p className="text-base font-bold text-[var(--color-ink)]">Patient {p.patient_id}</p>
                  <Badge tone={acuityTone(p.acuity)}>{acuityLabel(p.acuity)}</Badge>
                </div>
                <p className="mt-1 text-xs text-[var(--color-ink-faint)]">
                  {p.age ?? "—"}y · {p.sex ?? "—"}
                </p>
                <p className="mt-3 text-sm text-[var(--color-ink-soft)]">{p.chief_complaint}</p>
                <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
                  <MiniVital label="HR" value={p.vitals.heart_rate} />
                  <MiniVital label="SpO₂" value={p.vitals.spo2} />
                  <MiniVital label="RR" value={p.vitals.resp_rate} />
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function MiniVital({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-lg bg-[var(--color-surface-muted)] py-1.5">
      <p className="text-[10px] text-[var(--color-ink-faint)]">{label}</p>
      <p className="text-sm font-semibold text-[var(--color-ink)]">{value ?? "—"}</p>
    </div>
  );
}
