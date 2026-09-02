import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { TriageModal } from "../components/TriageModal";
import {
  Badge,
  Button,
  Card,
  DEPT_LABELS,
  EmptyState,
  RiskBar,
  Spinner,
  VitalTile,
  acuityLabel,
  acuityTone,
} from "../components/ui";
import type { AssessmentResult, Observation, OperationalDecision, PatientVitals, ResourceCheck, SimVitals } from "../types";

type QueueEntry = {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  status: string;
  vitals?: SimVitals;
  operational_decision?: OperationalDecision | null;
};

const OBS_TYPES: { value: string; label: string; unit: string }[] = [
  { value: "heart_rate", label: "Heart rate", unit: "bpm" },
  { value: "spo2", label: "SpO₂", unit: "%" },
  { value: "resp_rate", label: "Respiratory rate", unit: "/min" },
  { value: "sbp", label: "Systolic BP", unit: "mmHg" },
  { value: "dbp", label: "Diastolic BP", unit: "mmHg" },
  { value: "temperature", label: "Temperature", unit: "°C" },
];

export function PatientWorkspace() {
  const { id = "" } = useParams();
  const { sessionId, proposeAction, mutationTick, hospitalId } = useSession();

  const { data: detail, loading, refetch } = usePoll(() => api.getPatient(id), 20000, [id]);
  // Chart lookup (above) and live simulation state are separate stores (see
  // PatientList.tsx's union) — this page needs both to know whether Triage
  // Patient should activate a chart-only record or triage an already-ARRIVED
  // one, and to hide the action once the patient is already triaged+.
  const { data: dash, refetch: refetchDash } = usePoll(() => api.dashboard(hospitalId), 8000, [hospitalId]);
  const [assessment, setAssessment] = useState<AssessmentResult | null>(null);
  const [resourceCheck, setResourceCheck] = useState<ResourceCheck | null>(null);
  const [assessing, setAssessing] = useState(false);
  const [assessError, setAssessError] = useState<string | null>(null);
  const [staleBanner, setStaleBanner] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [showTriage, setShowTriage] = useState(false);

  const [obsType, setObsType] = useState(OBS_TYPES[0].value);
  const [obsValue, setObsValue] = useState("");
  const [obsBusy, setObsBusy] = useState(false);
  const [obsError, setObsError] = useState<string | null>(null);

  const lastTick = useMemo(() => mutationTick, []);
  useEffect(() => {
    if (mutationTick !== lastTick) {
      refetch();
      if (assessment) setStaleBanner(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mutationTick]);

  // A clinical assessment/resource-check result was computed for whichever
  // hospital was selected when it ran. Switching hospitals mid-view must not
  // keep showing that stale department/risk/resource picture as if it still
  // applied to the newly selected hospital — clear it back to the same
  // "no assessment yet" state a fresh page load starts in.
  const lastHospitalId = useMemo(() => hospitalId, []);
  useEffect(() => {
    if (hospitalId !== lastHospitalId) {
      setAssessment(null);
      setResourceCheck(null);
      setAssessError(null);
      setStaleBanner(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hospitalId]);

  const runAssessment = async () => {
    if (!sessionId) return;
    setAssessing(true);
    setAssessError(null);
    try {
      const result = await api.assessPatient(id, sessionId, hospitalId);
      setAssessment(result.assessment);
      setResourceCheck(result.resource_check);
      setStaleBanner(false);
    } catch (err) {
      setAssessError((err as Error).message);
    } finally {
      setAssessing(false);
    }
  };

  const submitObservation = async () => {
    if (!sessionId || !obsValue.trim()) return;
    setObsBusy(true);
    setObsError(null);
    try {
      const outcome = await proposeAction("add_patient_observation", {
        patient_id: id,
        observation_type: obsType,
        value: Number(obsValue),
      });
      if (outcome.status === "executed") {
        setObsValue("");
        await refetch();
        // The backend's piggybacked reassessment (add_patient_observation's
        // server-side auto-rerun) has no hospital_id to work with and always
        // runs against the default hospital's routing policy — trusting it
        // here would silently show the wrong hospital's operational
        // department/resource picture whenever a non-default hospital is
        // selected. Re-run the same hospital-scoped call the "Run
        // assessment" button uses instead of reading that payload.
        await runAssessment();
      }
      // A "failed" outcome here is a deliberate nurse cancellation/rejection
      // of the confirmation — not an error, so it's left silent by design:
      // the input stays as typed so the nurse can retry, matching how a
      // cancelled confirmation modal elsewhere in the app just closes.
    } catch (err) {
      // An actual thrown failure (network/backend error) is different from a
      // deliberate cancellation above — previously uncaught here, so it
      // surfaced only as a console-level unhandled rejection with no
      // visible feedback to the nurse at all.
      setObsError(err instanceof Error ? err.message : String(err));
    } finally {
      setObsBusy(false);
    }
  };

  if (loading && !detail) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
      </div>
    );
  }
  if (!detail) return <EmptyState title={`No patient record found for "${id}".`} />;

  const { summary, observations } = detail;

  const fullQueue = ((dash?.full_queue as unknown as QueueEntry[]) ?? []);
  const simEntry = fullQueue.find((p) => p.patient_id === summary.patient_id) ?? null;
  const departmentOptions = dash?.departments.map((d) => d.name) ?? [];
  // Triage Patient is only meaningful before triage exists: activate (for a
  // chart-only patient with no live record) or triage directly (for one
  // already ARRIVED). Once TRIAGED/IN_TREATMENT/DISCHARGED there's nothing
  // new to do here — the same real triage state is already correct
  // everywhere else (Patients list, Live Operations, drawer).
  const canTriage =
    !simEntry
      ? summary.age != null && summary.sex != null && summary.chief_complaint.trim() !== ""
      : simEntry.status === "ARRIVED";

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <Link to="/patients" className="text-xs text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]">
        ← All patients
      </Link>

      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-[var(--color-ink)]">Patient {summary.patient_id}</h1>
            <Badge tone={acuityTone(summary.acuity)}>{acuityLabel(summary.acuity)}</Badge>
          </div>
          <p className="mt-1 text-sm text-[var(--color-ink-soft)]">
            {summary.age ?? "—"}y {summary.sex ?? ""} · {summary.chief_complaint}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {canTriage && (
            <Button variant="secondary" onClick={() => setShowTriage(true)}>
              Triage Patient
            </Button>
          )}
          <Button onClick={runAssessment} disabled={assessing}>
            {assessing ? (
              <>
                <Spinner className="h-4 w-4" /> Running assessment…
              </>
            ) : assessment ? (
              "Refresh assessment"
            ) : (
              "Run assessment"
            )}
          </Button>
        </div>
      </div>

      {staleBanner && (
        <div className="rounded-lg border border-[var(--color-brand-100)] bg-[var(--color-brand-50)] px-4 py-2.5 text-xs text-[var(--color-brand-700)]">
          Vitals were updated elsewhere. Refresh the assessment to see the latest clinical picture.
        </div>
      )}

      <Card title="Current vitals" subtitle={`Last updated ${new Date(summary.last_updated).toLocaleTimeString()}`}>
        <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
          <VitalTile icon="❤" label="HR" value={summary.vitals.heart_rate} unit="bpm" danger={isDanger("heart_rate", summary.vitals.heart_rate)} />
          <VitalTile icon="◔" label="SpO₂" value={summary.vitals.spo2} unit="%" danger={isDanger("spo2", summary.vitals.spo2)} />
          <VitalTile icon="↝" label="Resp" value={summary.vitals.resp_rate} unit="/min" danger={isDanger("resp_rate", summary.vitals.resp_rate)} />
          <VitalTile icon="◆" label="SBP" value={summary.vitals.sbp} unit="mmHg" />
          <VitalTile icon="◇" label="DBP" value={summary.vitals.dbp} unit="mmHg" />
          <VitalTile icon="◉" label="Temp" value={summary.vitals.temperature} unit="°C" />
        </div>

        <div className="mt-5 border-t border-[var(--color-border)] pt-4">
          <p className="mb-2 text-xs font-semibold text-[var(--color-ink-soft)]">Record a new observation</p>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={obsType}
              onChange={(e) => setObsType(e.target.value)}
              className="rounded-lg border border-[var(--color-border)] px-2.5 py-1.5 text-sm"
            >
              {OBS_TYPES.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
            <input
              type="number"
              value={obsValue}
              onChange={(e) => setObsValue(e.target.value)}
              placeholder={OBS_TYPES.find((o) => o.value === obsType)?.unit}
              className="w-28 rounded-lg border border-[var(--color-border)] px-2.5 py-1.5 text-sm"
            />
            <Button size="sm" onClick={submitObservation} disabled={obsBusy || !obsValue.trim()}>
              Record
            </Button>
            <span className="text-[11px] text-[var(--color-ink-faint)]">Timestamped automatically. You'll confirm before it's saved.</span>
          </div>
          {obsError && (
            <div className="mt-2 rounded-lg border border-[var(--color-critical-100)] bg-[var(--color-critical-50)] px-3 py-2 text-xs text-[var(--color-critical-600)]">
              {obsError}
            </div>
          )}
        </div>
      </Card>

      <Card title="Observation timeline" subtitle="Most recent first">
        <ObservationTimeline observations={observations} />
      </Card>

      {assessError && (
        <div className="rounded-lg border border-[var(--color-critical-100)] bg-[var(--color-critical-50)] px-4 py-3 text-sm text-[var(--color-critical-600)]">
          {assessError}
        </div>
      )}

      {assessment && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card title="Clinical assessment" subtitle={assessment.confidence_note}>
            <div className="space-y-3">
              <RiskBar label="Hospital admission risk" value={assessment.reconciled_admission_risk} emphasized />
              <RiskBar label="ICU risk" value={assessment.reconciled_icu_risk} emphasized />
            </div>

            {assessment.red_flags.length > 0 && (
              <div className="mt-4">
                <p className="mb-1.5 text-xs font-semibold text-[var(--color-ink-soft)]">Red flags</p>
                <div className="flex flex-wrap gap-1.5">
                  {assessment.red_flags.map((f, i) => (
                    <Badge key={i} tone="critical">
                      {f}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {assessment.top_diagnoses.length > 0 && (
              <div className="mt-3">
                <p className="mb-1.5 text-xs font-semibold text-[var(--color-ink-soft)]">Possible diagnoses</p>
                <div className="flex flex-wrap gap-1.5">
                  {assessment.top_diagnoses.map((d, i) => (
                    <Badge key={i} tone="brand">
                      {d}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-4 flex items-center gap-2">
              <Badge tone={assessment.branches_agree ? "good" : "warn"} dot>
                {assessment.branches_agree ? "Models agree" : "Models disagree — treated conservatively"}
              </Badge>
            </div>

            {(assessment.rag_trajectory ||
              assessment.rag_urgency ||
              assessment.rag_evidence_strength != null ||
              assessment.rag_escalation_concern != null) && (
              <div className="mt-4 space-y-1.5 border-t border-[var(--color-border)] pt-3">
                <p className="text-xs font-semibold text-[var(--color-ink-soft)]">Contextual trajectory (RAG)</p>
                {assessment.rag_trajectory && (
                  <p className="text-xs leading-relaxed text-[var(--color-ink)]">{assessment.rag_trajectory}</p>
                )}
                <div className="flex flex-wrap items-center gap-1.5">
                  {assessment.rag_urgency && (
                    <Badge tone={urgencyTone(assessment.rag_urgency)}>{assessment.rag_urgency} urgency</Badge>
                  )}
                  {assessment.rag_evidence_strength != null && (
                    <Badge tone="neutral">Evidence strength {assessment.rag_evidence_strength}/5</Badge>
                  )}
                  {assessment.rag_escalation_concern != null && (
                    <Badge tone={assessment.rag_escalation_concern ? "warn" : "good"} dot>
                      {assessment.rag_escalation_concern ? "Escalation concern flagged" : "No escalation concern"}
                    </Badge>
                  )}
                </div>
              </div>
            )}

            {assessment.rag_narrative && (
              <div className="mt-4 border-t border-[var(--color-border)] pt-3">
                <button
                  onClick={() => setShowReasoning((s) => !s)}
                  className="text-xs font-medium text-[var(--color-brand-600)] hover:underline"
                >
                  {showReasoning ? "Hide clinical reasoning" : "Why? Show clinical reasoning"}
                </button>
                {showReasoning && (
                  <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-[var(--color-ink-soft)]">
                    {assessment.rag_narrative}
                  </p>
                )}
              </div>
            )}
          </Card>

          <Card title="Routing" subtitle="Clinically preferred vs. currently available">
            <RoutingPanel department={assessment.department} reasoning={assessment.department_reasoning} check={resourceCheck} />
          </Card>
        </div>
      )}

      {!assessment && !assessing && (
        <EmptyState
          title="No assessment yet"
          subtitle="Run an assessment to see clinical risk, red flags, and routing recommendations for this patient."
        />
      )}

      {showTriage && (
        <TriageModal
          patientId={summary.patient_id}
          age={simEntry ? simEntry.age : summary.age!}
          sex={simEntry ? simEntry.sex : summary.sex!}
          chiefComplaint={simEntry ? simEntry.chief_complaint : summary.chief_complaint}
          vitals={simEntry ? (simEntry.vitals ?? {}) : toSimVitals(summary.vitals)}
          hospitalId={hospitalId}
          departmentOptions={departmentOptions}
          activation={
            simEntry
              ? undefined
              : { chiefComplaint: summary.chief_complaint, age: summary.age!, sex: summary.sex!, acuity: summary.acuity }
          }
          onClose={() => setShowTriage(false)}
          onDone={() => { setShowTriage(false); refetchDash(); }}
        />
      )}
    </div>
  );
}

// TriageModal's vitals display expects SimVitals' short keys; the chart
// record carries PatientVitals' long keys — purely a display mapping.
function toSimVitals(v: PatientVitals): SimVitals {
  return {
    hr: v.heart_rate ?? undefined,
    rr: v.resp_rate ?? undefined,
    spo2: v.spo2 ?? undefined,
    sbp: v.sbp ?? undefined,
    dbp: v.dbp ?? undefined,
    temp: v.temperature ?? undefined,
    pain: v.pain_score ?? undefined,
  };
}

function RoutingPanel({
  department,
  reasoning,
  check,
}: {
  department: string;
  reasoning: string;
  check: ResourceCheck | null;
}) {
  const constrained = check?.resource_constrained;
  const allocated = check?.allocated_department ?? department;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="flex-1 rounded-xl border border-[var(--color-brand-100)] bg-[var(--color-brand-50)] p-3 text-center">
          <p className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-brand-600)]">
            Clinically preferred
          </p>
          <p className="mt-1 text-lg font-bold text-[var(--color-brand-700)]">{DEPT_LABELS[department] ?? department}</p>
        </div>
        {constrained && (
          <>
            <span className="text-[var(--color-ink-faint)]">→</span>
            <div className="flex-1 rounded-xl border border-[var(--color-warn-100)] bg-[var(--color-warn-50)] p-3 text-center">
              <p className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-warn-600)]">
                Current allocation
              </p>
              <p className="mt-1 text-lg font-bold text-[var(--color-warn-600)]">{DEPT_LABELS[allocated] ?? allocated}</p>
            </div>
          </>
        )}
      </div>

      <p className="text-xs text-[var(--color-ink-soft)]">{reasoning}</p>

      {check && (
        <div className="rounded-lg bg-[var(--color-surface-muted)] p-3 text-xs text-[var(--color-ink-soft)]">
          {check.available} of {check.capacity} beds open in {DEPT_LABELS[department] ?? department} right now.
        </div>
      )}

      {constrained && check?.note && (
        <div className="rounded-lg border border-[var(--color-warn-100)] bg-[var(--color-warn-50)] p-3 text-xs text-[var(--color-warn-600)]">
          <p className="font-semibold">Resource constraint — human review recommended</p>
          <p className="mt-1">{check.note}</p>
          <p className="mt-1 text-[var(--color-ink-faint)]">
            This does not mean the patient is less urgent — {DEPT_LABELS[department] ?? department} remains clinically indicated.
          </p>
        </div>
      )}
      {check?.tight && !constrained && (
        <div className="rounded-lg border border-[var(--color-warn-100)] bg-[var(--color-warn-50)] p-3 text-xs text-[var(--color-warn-600)]">
          {check.note}
        </div>
      )}
    </div>
  );
}

function ObservationTimeline({ observations }: { observations: Observation[] }) {
  if (observations.length === 0) {
    return <EmptyState title="No observations recorded yet." />;
  }
  const rows = [...observations].reverse();
  return (
    <ul className="space-y-2">
      {rows.map((o, i) => {
        const { field, value } = extractVital(o);
        return (
          <li
            key={i}
            className={`flex items-center justify-between rounded-lg border px-3 py-2 text-xs ${
              i === 0 ? "border-[var(--color-brand-100)] bg-[var(--color-brand-50)]" : "border-[var(--color-border)]"
            }`}
          >
            <div className="flex items-center gap-2">
              <span className="font-semibold text-[var(--color-ink)]">{field ?? o.type}</span>
              {value != null && <span className="text-[var(--color-ink-soft)]">{String(value)}</span>}
              {o.note && <span className="text-[var(--color-ink-faint)]">— {o.note}</span>}
            </div>
            <div className="flex items-center gap-2">
              {i === 0 && <Badge tone="brand">Latest</Badge>}
              <span className="text-[var(--color-ink-faint)]">{new Date(o.timestamp).toLocaleString()}</span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function extractVital(o: Observation): { field: string | null; value: unknown } {
  const knownFields = ["heart_rate", "spo2", "resp_rate", "sbp", "dbp", "temperature", "pain_score"];
  for (const f of knownFields) {
    if (o[f] !== undefined) return { field: f.replace("_", " "), value: o[f] };
  }
  return { field: null, value: null };
}

function urgencyTone(urgency: string): "good" | "brand" | "warn" | "critical" | "neutral" {
  switch (urgency) {
    case "low":
      return "good";
    case "moderate":
      return "brand";
    case "high":
      return "warn";
    case "critical":
      return "critical";
    default:
      return "neutral";
  }
}

function isDanger(kind: string, value: number | null): boolean {
  if (value == null) return false;
  switch (kind) {
    case "heart_rate":
      return value >= 120 || value <= 45;
    case "spo2":
      return value < 92;
    case "resp_rate":
      return value >= 24 || value <= 8;
    default:
      return false;
  }
}
