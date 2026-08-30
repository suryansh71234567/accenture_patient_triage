import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useSession } from "../state/SessionContext";
import { Badge, Button, Card, EmptyState, Spinner } from "../components/ui";
import type { CalibrationScenario, DepartmentConfigInput } from "../types";

type DeptRow = { name: string; capacity: number; occupied: number; included: boolean };

const DEFAULT_DEPARTMENTS: DeptRow[] = [
  { name: "ICU", capacity: 10, occupied: 4, included: true },
  { name: "CICU", capacity: 6, occupied: 2, included: true },
  { name: "ADMITTED_GEN", capacity: 40, occupied: 20, included: true },
  { name: "ED_OBS", capacity: 15, occupied: 5, included: true },
];

export function HospitalOnboarding() {
  const { hospitalId, setHospitalId, bumpMutationTick } = useSession();

  const [newId, setNewId] = useState("");
  const [newName, setNewName] = useState("");
  const [depts, setDepts] = useState<DeptRow[]>(DEFAULT_DEPARTMENTS);
  const [addBusy, setAddBusy] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [addedId, setAddedId] = useState<string | null>(null);

  const toggleDept = (name: string) =>
    setDepts((ds) => ds.map((d) => (d.name === name ? { ...d, included: !d.included } : d)));
  const updateDept = (name: string, field: "capacity" | "occupied", value: number) =>
    setDepts((ds) => ds.map((d) => (d.name === name ? { ...d, [field]: value } : d)));

  const submitNewHospital = async () => {
    setAddError(null);
    setAddedId(null);
    if (!newId.trim() || !newName.trim()) {
      setAddError("Hospital ID and name are required.");
      return;
    }
    const included = depts.filter((d) => d.included);
    if (included.length === 0) {
      setAddError("At least one department is required.");
      return;
    }
    setAddBusy(true);
    try {
      const departments: Record<string, DepartmentConfigInput> = {};
      for (const d of included) {
        departments[d.name] = { capacity: d.capacity, occupied: d.occupied, status: "OPEN" };
      }
      departments.DISCHARGE = { capacity: 999, occupied: 0, status: "OPEN" };

      const result = await api.registerHospital({
        hospital_id: newId.trim(),
        hospital_name: newName.trim(),
        departments,
      });
      bumpMutationTick(); // so HospitalSelector picks up the new hospital
      setHospitalId(result.hospital_id);
      setAddedId(result.hospital_id);
      setNewId("");
      setNewName("");
    } catch (err) {
      setAddError((err as Error).message);
    } finally {
      setAddBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="text-2xl font-bold text-[var(--color-ink)]">Hospital onboarding</h1>
        <p className="mt-1 text-sm text-[var(--color-ink-faint)]">
          Add a hospital, then calibrate its routing policy so it can be used for live triage.
        </p>
      </div>

      <Card title="1. Add hospital" subtitle="Register a new hospital and its department capacities">
        <div className="space-y-4">
          {addError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">{addError}</div>
          )}
          {addedId && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              "{addedId}" registered and selected. Calibrate it below.
            </div>
          )}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-[var(--color-ink-soft)] uppercase tracking-wide">
                Hospital ID
              </label>
              <input
                value={newId}
                onChange={(e) => setNewId(e.target.value)}
                placeholder="e.g. westside_clinic"
                className="w-full rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-ink)]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-semibold text-[var(--color-ink-soft)] uppercase tracking-wide">
                Hospital name
              </label>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. Westside Clinic"
                className="w-full rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-ink)]"
              />
            </div>
          </div>

          <div>
            <p className="mb-2 text-xs font-semibold text-[var(--color-ink-soft)] uppercase tracking-wide">
              Departments
            </p>
            <div className="space-y-2">
              {depts.map((d) => (
                <div
                  key={d.name}
                  className="flex flex-wrap items-center gap-3 rounded-lg border border-[var(--color-border)] px-3 py-2"
                >
                  <label className="flex w-40 items-center gap-2 text-sm font-medium text-[var(--color-ink)]">
                    <input type="checkbox" checked={d.included} onChange={() => toggleDept(d.name)} />
                    {d.name}
                  </label>
                  <label className="flex items-center gap-1.5 text-xs text-[var(--color-ink-faint)]">
                    Capacity
                    <input
                      type="number"
                      min={0}
                      disabled={!d.included}
                      value={d.capacity}
                      onChange={(e) => updateDept(d.name, "capacity", Number(e.target.value))}
                      className="w-20 rounded border border-[var(--color-border)] px-2 py-1 text-sm disabled:opacity-40"
                    />
                  </label>
                  <label className="flex items-center gap-1.5 text-xs text-[var(--color-ink-faint)]">
                    Occupied
                    <input
                      type="number"
                      min={0}
                      disabled={!d.included}
                      value={d.occupied}
                      onChange={(e) => updateDept(d.name, "occupied", Number(e.target.value))}
                      className="w-20 rounded border border-[var(--color-border)] px-2 py-1 text-sm disabled:opacity-40"
                    />
                  </label>
                </div>
              ))}
              <p className="text-[10px] text-[var(--color-ink-faint)]">DISCHARGE is always included automatically.</p>
            </div>
          </div>

          <Button disabled={addBusy} onClick={submitNewHospital}>
            {addBusy ? (
              <>
                <Spinner className="h-4 w-4" /> Registering…
              </>
            ) : (
              "Register hospital"
            )}
          </Button>
        </div>
      </Card>

      <CalibrationSection hospitalId={hospitalId} />
    </div>
  );
}

function CalibrationSection({ hospitalId }: { hospitalId: string }) {
  const [status, setStatus] = useState<{ calibrated: boolean } | null>(null);
  const [scenarios, setScenarios] = useState<CalibrationScenario[]>([]);
  const [loading, setLoading] = useState(true);
  const [responses, setResponses] = useState<Record<string, string>>({});
  const [submitBusy, setSubmitBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitResult, setSubmitResult] = useState<{ trained_scenarios: number | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setSubmitResult(null);
    setSubmitError(null);
    Promise.all([api.calibrationStatus(hospitalId), api.calibrationScenarios(hospitalId)])
      .then(([statusRes, scenariosRes]) => {
        if (cancelled) return;
        setStatus(statusRes);
        setScenarios(scenariosRes.scenarios);
        // Pre-fill from each scenario's own default so every scenario has a
        // real, submittable answer — the nurse only needs to change the
        // ones that don't match how this hospital actually operates.
        setResponses(
          Object.fromEntries(scenariosRes.scenarios.map((s) => [s.scenario_id, s.preferred_department]))
        );
      })
      .catch(() => {
        if (!cancelled) {
          setStatus(null);
          setScenarios([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [hospitalId]);

  const pick = (scenarioId: string, dept: string) => setResponses((r) => ({ ...r, [scenarioId]: dept }));

  const submit = async () => {
    setSubmitBusy(true);
    setSubmitError(null);
    try {
      const result = await api.submitCalibration(hospitalId, responses);
      setSubmitResult(result);
      setStatus({ calibrated: result.calibrated });
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitBusy(false);
    }
  };

  return (
    <Card
      title="2. Calibrate"
      subtitle={`Nurse-guided routing calibration for "${hospitalId}"`}
      right={
        status && (
          <Badge tone={status.calibrated ? "good" : "warn"} dot>
            {status.calibrated ? "Calibrated" : "Not calibrated"}
          </Badge>
        )
      }
    >
      {loading ? (
        <div className="flex items-center justify-center py-10">
          <Spinner className="h-5 w-5 text-[var(--color-brand-500)]" />
        </div>
      ) : scenarios.length === 0 ? (
        <EmptyState title="No calibration scenarios available for this hospital's departments." />
      ) : (
        <div className="space-y-4">
          <p className="text-xs text-[var(--color-ink-faint)]">
            Each scenario starts with a typical default department. Click a different department for any scenario
            that doesn't match how "{hospitalId}" actually operates, then train the policy below.
          </p>

          {submitError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
              {submitError}
            </div>
          )}
          {submitResult && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              Policy trained and saved for "{hospitalId}" ({submitResult.trained_scenarios ?? scenarios.length}{" "}
              scenarios). This hospital's live routing and simulation will now use it.
            </div>
          )}

          <ul className="max-h-[480px] space-y-3 overflow-y-auto pr-1">
            {scenarios.map((s) => (
              <li key={s.scenario_id} className="rounded-xl border border-[var(--color-border)] p-3">
                <p className="text-sm text-[var(--color-ink)]">{s.description}</p>
                <p className="mt-1 text-[10px] text-[var(--color-ink-faint)]">{s.reason}</p>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {s.candidate_departments.map((dept) => {
                    const chosen = responses[s.scenario_id] === dept;
                    return (
                      <button
                        key={dept}
                        onClick={() => pick(s.scenario_id, dept)}
                        className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                          chosen
                            ? "border-[var(--color-brand-500)] bg-[var(--color-brand-50)] text-[var(--color-brand-700)]"
                            : "border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-brand-300)]"
                        }`}
                      >
                        {dept}
                      </button>
                    );
                  })}
                </div>
              </li>
            ))}
          </ul>

          <Button disabled={submitBusy} onClick={submit}>
            {submitBusy ? (
              <>
                <Spinner className="h-4 w-4" /> Training…
              </>
            ) : (
              "3. Train & save policy"
            )}
          </Button>
        </div>
      )}
    </Card>
  );
}
