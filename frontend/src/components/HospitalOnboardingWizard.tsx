import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useModalA11y } from "../hooks/useModalA11y";
import { useSession } from "../state/SessionContext";
import type { CalibrationScenario, DepartmentConfigInput } from "../types";

type DeptRow = { name: string; capacity: number; occupied: number; included: boolean };
type Step = "details" | "departments" | "policy" | "review";

const DEFAULT_DEPARTMENTS: DeptRow[] = [
  { name: "ICU", capacity: 10, occupied: 4, included: true },
  { name: "CICU", capacity: 6, occupied: 2, included: true },
  { name: "ADMITTED_GEN", capacity: 40, occupied: 20, included: true },
  { name: "ED_OBS", capacity: 15, occupied: 5, included: true },
];

const STEP_LABELS: { key: Step; label: string }[] = [
  { key: "details", label: "1. Details" },
  { key: "departments", label: "2. Departments" },
  { key: "policy", label: "3. Policy Framing" },
  { key: "review", label: "4. Review" },
];

/**
 * 4-step modal wizard (mockup's Hospital Onboarding Modal), replacing the
 * old flat /hospitals page. Fixes the calibration-scope bug the flat page
 * had: Step 3 here is bound to `registeredId`, a value captured ONCE when
 * this wizard's own registration call succeeds — never to the reactive
 * app-wide `hospitalId` from SessionContext. The old page's calibration
 * section read `hospitalId` directly, so selecting a different hospital
 * anywhere else in the app while that section was open would silently
 * start showing/training scenarios for the wrong hospital.
 */
export function HospitalOnboardingWizard({
  onClose,
  existingHospital,
}: {
  onClose: () => void;
  /** Skip registration and jump straight to Policy Framing for an already-registered hospital (mockup's "Configure Policy"). */
  existingHospital?: { hospital_id: string; hospital_name: string };
}) {
  const { setHospitalId, bumpMutationTick } = useSession();
  const containerRef = useModalA11y(onClose);
  const [step, setStep] = useState<Step>(existingHospital ? "policy" : "details");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const [newId, setNewId] = useState("");
  const [newName, setNewName] = useState("");
  const [depts, setDepts] = useState<DeptRow[]>(DEFAULT_DEPARTMENTS);

  const [registeredId, setRegisteredId] = useState<string | null>(existingHospital?.hospital_id ?? null);
  const [registeredName, setRegisteredName] = useState<string>(existingHospital?.hospital_name ?? "");

  const [scenarios, setScenarios] = useState<CalibrationScenario[]>([]);
  const [responses, setResponses] = useState<Record<string, string>>({});
  const [scenariosLoading, setScenariosLoading] = useState(false);
  const [submitResult, setSubmitResult] = useState<{ trained_scenarios: number | null } | null>(null);

  const stepIndex = STEP_LABELS.findIndex((s) => s.key === step);

  const toggleDept = (idx: number) => setDepts((ds) => ds.map((d, i) => (i === idx ? { ...d, included: !d.included } : d)));
  const updateDept = (idx: number, field: "capacity" | "occupied", value: number) =>
    setDepts((ds) => ds.map((d, i) => (i === idx ? { ...d, [field]: value } : d)));
  const renameDept = (idx: number, value: string) => setDepts((ds) => ds.map((d, i) => (i === idx ? { ...d, name: value } : d)));
  const addDept = () => setDepts((ds) => [...ds, { name: "", capacity: 10, occupied: 0, included: true }]);
  const removeDept = (idx: number) => setDepts((ds) => ds.filter((_, i) => i !== idx));

  const registerHospital = async () => {
    setError(null);
    if (!newId.trim() || !newName.trim()) {
      setError("Hospital ID and name are required.");
      return;
    }
    const included = depts.filter((d) => d.included);
    if (included.length === 0) {
      setError("At least one department is required.");
      return;
    }
    if (included.some((d) => !d.name.trim())) {
      setError("Every included department needs a name.");
      return;
    }
    setSubmitting(true);
    try {
      const departments: Record<string, DepartmentConfigInput> = {};
      for (const d of included) departments[d.name.trim()] = { capacity: d.capacity, occupied: d.occupied, status: "OPEN" };
      departments.DISCHARGE = { capacity: 999, occupied: 0, status: "OPEN" };

      const result = await api.registerHospital({ hospital_id: newId.trim(), hospital_name: newName.trim(), departments });
      bumpMutationTick();
      setHospitalId(result.hospital_id);
      setRegisteredId(result.hospital_id);
      setRegisteredName(result.hospital_name);
      setStep("policy");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  // Fetches Step 3's scenarios against `registeredId` — the wizard's own
  // captured value, not the live session hospitalId.
  useEffect(() => {
    if (!registeredId || step !== "policy") return;
    let cancelled = false;
    setScenariosLoading(true);
    setSubmitResult(null);
    api
      .calibrationScenarios(registeredId)
      .then((res) => {
        if (cancelled) return;
        setScenarios(res.scenarios);
        setResponses(Object.fromEntries(res.scenarios.map((s) => [s.scenario_id, s.preferred_department])));
      })
      .catch(() => {
        if (!cancelled) setScenarios([]);
      })
      .finally(() => {
        if (!cancelled) setScenariosLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [registeredId, step]);

  const pick = (scenarioId: string, dept: string) => setResponses((r) => ({ ...r, [scenarioId]: dept }));

  const submitCalibration = async () => {
    if (!registeredId) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await api.submitCalibration(registeredId, responses);
      setSubmitResult(result);
      setStep("review");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-[65] flex items-center justify-center bg-black/40 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="max-h-[88vh] w-[640px] overflow-y-auto rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-7 shadow-2xl">
        <div className="mb-1 flex items-start justify-between">
          <p className="text-[15px] font-bold text-[var(--color-ink)]">Hospital Onboarding</p>
          <button onClick={onClose} className="text-lg leading-none text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]">×</button>
        </div>
        <p className="mb-4 text-[11.5px] text-[var(--color-ink-faint)]">
          Add a hospital, then calibrate its routing policy so it can be used for live triage.
        </p>

        <div className="mb-4 flex gap-1.5">
          {STEP_LABELS.map((s, i) => (
            <span
              key={s.key}
              className="rounded-full px-2.5 py-1 text-[10px] font-bold"
              style={
                i === stepIndex
                  ? { background: "var(--color-brand-500)", color: "#fff" }
                  : i < stepIndex
                    ? { background: "var(--color-good-50)", color: "var(--color-good-600)" }
                    : { background: "var(--color-surface-muted)", color: "var(--color-ink-faint)" }
              }
            >
              {s.label}
            </span>
          ))}
        </div>

        {error && <div className="mb-3 rounded-lg border border-[var(--color-critical-100)] bg-[var(--color-critical-50)] px-3 py-2 text-xs text-[var(--color-critical-600)]">{error}</div>}

        {step === "details" && (
          <>
            <p className="mb-2 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">1. Hospital Details</p>
            <div className="mb-4 grid grid-cols-2 gap-2.5">
              <label className="flex flex-col gap-1">
                <span className="text-[9.5px] uppercase text-[var(--color-ink-faint)]">Hospital ID</span>
                <input value={newId} onChange={(e) => setNewId(e.target.value)} placeholder="e.g. westside_clinic" className="rounded-lg border border-[var(--color-border)] px-2.5 py-2 font-mono text-[12.5px]" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-[9.5px] uppercase text-[var(--color-ink-faint)]">Hospital Name</span>
                <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. Westside Clinic" className="rounded-lg border border-[var(--color-border)] px-2.5 py-2 text-[12.5px]" />
              </label>
            </div>
            <button
              onClick={() => setStep("departments")}
              className="w-full rounded-lg bg-[var(--color-brand-500)] px-4 py-2.5 text-xs font-bold text-white"
            >
              Next: Departments &amp; Capacity
            </button>
          </>
        )}

        {step === "departments" && (
          <>
            <p className="mb-1 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">2. Departments &amp; Capacity</p>
            <p className="mb-2.5 text-[11px] text-[var(--color-ink-faint)]">
              Suggested defaults below — rename, remove, or add departments freely. Hospitals can define any set of departments.
            </p>
            <div className="mb-2.5 flex flex-col gap-2">
              {depts.map((d, i) => (
                <div key={i} className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--color-border)] px-2.5 py-2">
                  <input type="checkbox" checked={d.included} onChange={() => toggleDept(i)} aria-label="Include department" />
                  <input value={d.name} onChange={(e) => renameDept(i, e.target.value)} placeholder="Department name" className="min-w-0 flex-1 rounded border border-[var(--color-border)] px-2 py-1 text-xs font-semibold" />
                  <label className="flex items-center gap-1.5 text-[10.5px] text-[var(--color-ink-faint)]">
                    Capacity
                    <input type="number" value={d.capacity} onChange={(e) => updateDept(i, "capacity", Number(e.target.value))} className="w-16 rounded border border-[var(--color-border)] px-1.5 py-1 font-mono text-xs" />
                  </label>
                  <label className="flex items-center gap-1.5 text-[10.5px] text-[var(--color-ink-faint)]">
                    Occupied
                    <input type="number" value={d.occupied} onChange={(e) => updateDept(i, "occupied", Number(e.target.value))} className="w-16 rounded border border-[var(--color-border)] px-1.5 py-1 font-mono text-xs" />
                  </label>
                  <button onClick={() => removeDept(i)} aria-label="Remove department" className="px-1 text-base text-[var(--color-critical-500)]">×</button>
                </div>
              ))}
            </div>
            <button onClick={addDept} className="mb-2.5 rounded-lg border border-dashed px-3.5 py-2 text-[11.5px] font-bold" style={{ borderColor: "var(--color-brand-400)", background: "var(--color-brand-50)", color: "var(--color-brand-600)" }}>
              + Add Department
            </button>
            <p className="mb-3.5 text-[10px] text-[var(--color-ink-faint)]">DISCHARGE is always included automatically as a virtual destination.</p>
            <div className="flex gap-2">
              <button onClick={() => setStep("details")} disabled={submitting} className="rounded-lg border border-[var(--color-border)] px-4 py-2.5 text-xs font-semibold disabled:opacity-50">Back</button>
              <button onClick={registerHospital} disabled={submitting} className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2.5 text-xs font-bold text-white disabled:opacity-50">
                {submitting ? "Registering…" : "Register Hospital"}
              </button>
            </div>
          </>
        )}

        {step === "policy" && registeredId && (
          <>
            <div className="mb-3 rounded-lg px-3 py-2 text-[11.5px]" style={{ background: "var(--color-good-50)", color: "var(--color-good-600)" }}>
              "{registeredName}" registered and selected. Calibrate its routing policy below.
            </div>
            <p className="mb-1 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">3. Policy Framing</p>
            <p className="mb-2.5 text-[11px] text-[var(--color-ink-faint)]">
              Available policy scenarios for "{registeredName}" ({scenarios.length}). Pick a different department for any scenario that doesn't match how it actually operates.
            </p>
            {scenariosLoading ? (
              <div className="flex justify-center py-8"><Spinner /></div>
            ) : scenarios.length === 0 ? (
              <div className="mb-3.5 rounded-xl bg-[var(--color-surface-muted)] px-4 py-4 text-center text-xs text-[var(--color-ink-soft)]">
                No applicable policy scenarios are available for this hospital configuration.
              </div>
            ) : (
              <div className="mb-3.5 max-h-[340px] space-y-2 overflow-y-auto">
                {scenarios.map((s) => (
                  <div key={s.scenario_id} className="rounded-lg border border-[var(--color-border)] px-3 py-2.5">
                    <p className="text-xs text-[var(--color-ink)]">{s.description}</p>
                    <p className="mt-0.5 text-[10px] text-[var(--color-ink-faint)]">{s.reason}</p>
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {s.candidate_departments.map((dept) => {
                        const chosen = responses[s.scenario_id] === dept;
                        return (
                          <button
                            key={dept}
                            onClick={() => pick(s.scenario_id, dept)}
                            className="rounded-full border px-2.5 py-1 text-[11px] font-medium"
                            style={chosen ? { borderColor: "var(--color-brand-500)", background: "var(--color-brand-50)", color: "var(--color-brand-700)" } : { borderColor: "var(--color-border)" }}
                          >
                            {dept}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <button
              onClick={submitCalibration}
              disabled={submitting || scenarios.length === 0}
              className="w-full rounded-xl bg-[var(--color-brand-500)] px-4 py-2.5 text-[13px] font-bold text-white disabled:opacity-50"
            >
              {submitting ? "Training…" : "Train & Save Policy"}
            </button>
          </>
        )}

        {step === "review" && (
          <>
            <p className="mb-2 text-[10px] font-bold uppercase tracking-wide" style={{ color: "var(--color-good-600)" }}>4. Policy Trained</p>
            <div className="mb-4 rounded-xl border p-3.5" style={{ borderColor: "var(--color-good-100)", background: "var(--color-good-50)" }}>
              <p className="text-[13px] font-bold" style={{ color: "var(--color-good-600)" }}>Policy trained and saved for "{registeredName}"</p>
              <p className="mt-1 text-[11.5px]" style={{ color: "var(--color-good-600)" }}>
                {submitResult?.trained_scenarios ?? 0} scenarios trained. This hospital's live routing and simulation will now use it.
              </p>
            </div>
            <div className="flex gap-2">
              <button onClick={() => setStep("policy")} className="rounded-lg border border-[var(--color-border)] px-4 py-2.5 text-xs font-semibold">Revise Policy</button>
              <button onClick={onClose} className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2.5 text-xs font-bold text-white">Done</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="h-5 w-5 animate-spin text-[var(--color-brand-500)]" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}
