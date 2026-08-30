import { useState } from "react";
import { api } from "../api/client";
import { useSession } from "../state/SessionContext";

interface Props {
  onSuccess: () => void;
  onClose: () => void;
}

const ACUITY_OPTS = [
  { value: 1, label: "1 — Critical" },
  { value: 2, label: "2 — Emergent" },
  { value: 3, label: "3 — Urgent" },
  { value: 4, label: "4 — Less Urgent" },
  { value: 5, label: "5 — Non-Urgent" },
];

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold text-[var(--color-ink-soft)] uppercase tracking-wide">
        {label}
        {hint && <span className="ml-1 text-[10px] font-normal text-[var(--color-ink-faint)]">({hint})</span>}
      </label>
      {children}
    </div>
  );
}

const inputCls =
  "rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-ink)] placeholder:text-[var(--color-ink-faint)] focus:outline-none focus:ring-2 focus:ring-[var(--color-brand-400)] transition";

export function ManualIntakeForm({ onSuccess, onClose }: Props) {
  const { hospitalId } = useSession();
  const [form, setForm] = useState({
    patient_id: "",
    chief_complaint: "",
    age: "",
    sex: "M",
    acuity: 3,
    hr: "",
    rr: "",
    spo2: "",
    sbp: "",
    dbp: "",
    temperature: "",
    pain: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ has_history: boolean; history_text: string; patient_id: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const set = (k: string, v: string | number) => setForm((f) => ({ ...f, [k]: v }));

  const num = (v: string) => (v.trim() === "" ? null : Number(v));

  const submit = async () => {
    if (!form.patient_id.trim() || !form.chief_complaint.trim() || !form.age) {
      setError("Patient ID, complaint, and age are required.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await api.manualArrival({
        patient_id: form.patient_id.trim(),
        chief_complaint: form.chief_complaint.trim(),
        age: Number(form.age),
        sex: form.sex,
        acuity: form.acuity,
        hr: num(form.hr),
        rr: num(form.rr),
        spo2: num(form.spo2),
        sbp: num(form.sbp),
        dbp: num(form.dbp),
        temperature: num(form.temperature),
        pain: num(form.pain),
        hospital_id: hospitalId,
      });
      setResult({ has_history: res.has_history, history_text: res.history_text, patient_id: res.patient_id });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  if (result) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
        <div className="w-full max-w-sm rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl space-y-4">
          <h2 className="text-lg font-bold text-[var(--color-ink)]">Patient Added to Queue</h2>

          {result.has_history ? (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 space-y-1">
              <p className="text-sm font-semibold text-emerald-700">✓ Prior hospital record found</p>
              <p className="text-xs text-emerald-600">{result.history_text || "History loaded from records."}</p>
              <p className="text-xs text-emerald-500 mt-1">RAG will retrieve this patient's past visits. The LLM reasoning will reference actual history.</p>
            </div>
          ) : (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 space-y-1">
              <p className="text-sm font-semibold text-amber-700">⚠ No prior record found</p>
              <p className="text-xs text-amber-600">This patient has no history in the system. The LLM will reason purely from current vitals and similar cases. Evidence strength will be lower.</p>
            </div>
          )}

          <p className="text-xs text-[var(--color-ink-faint)]">Patient <strong>{result.patient_id}</strong> is now in the ED queue. Use the Triage button to run the full assessment.</p>

          <div className="flex gap-2">
            <button
              onClick={() => { onSuccess(); onClose(); }}
              className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--color-brand-600)] transition"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl space-y-5 max-h-[90vh] overflow-y-auto">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold text-[var(--color-ink)]">Manual Patient Intake</h2>
            <p className="text-xs text-[var(--color-ink-faint)] mt-0.5">
              Enter a known Patient ID to load history automatically, or type a new ID for a walk-in.
            </p>
          </div>
          <button onClick={onClose} className="text-[var(--color-ink-faint)] hover:text-[var(--color-ink)] text-lg leading-none">✕</button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">{error}</div>
        )}

        {/* Identity */}
        <div className="grid grid-cols-2 gap-4">
          <Field label="Patient ID" hint="use known ID to load history">
            <input
              className={inputCls}
              placeholder="e.g. 52 or WALK-001"
              value={form.patient_id}
              onChange={(e) => set("patient_id", e.target.value)}
            />
          </Field>
          <Field label="Acuity">
            <select
              className={inputCls}
              value={form.acuity}
              onChange={(e) => set("acuity", Number(e.target.value))}
            >
              {ACUITY_OPTS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </Field>
        </div>

        <Field label="Chief Complaint">
          <input
            className={inputCls}
            placeholder="e.g. chest pain and shortness of breath"
            value={form.chief_complaint}
            onChange={(e) => set("chief_complaint", e.target.value)}
          />
        </Field>

        <div className="grid grid-cols-3 gap-4">
          <Field label="Age" hint="years">
            <input type="number" className={inputCls} placeholder="e.g. 67" value={form.age} onChange={(e) => set("age", e.target.value)} />
          </Field>
          <Field label="Sex">
            <select className={inputCls} value={form.sex} onChange={(e) => set("sex", e.target.value)}>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </Field>
          <Field label="Pain Score" hint="0–10">
            <input type="number" min={0} max={10} className={inputCls} placeholder="0–10" value={form.pain} onChange={(e) => set("pain", e.target.value)} />
          </Field>
        </div>

        {/* Vitals */}
        <div>
          <p className="text-xs font-semibold text-[var(--color-ink-faint)] uppercase tracking-wide mb-3">Vitals <span className="font-normal">(leave blank if not measured)</span></p>
          <div className="grid grid-cols-3 gap-4">
            <Field label="Heart Rate" hint="bpm">
              <input type="number" className={inputCls} placeholder="e.g. 112" value={form.hr} onChange={(e) => set("hr", e.target.value)} />
            </Field>
            <Field label="Resp Rate" hint="/min">
              <input type="number" className={inputCls} placeholder="e.g. 22" value={form.rr} onChange={(e) => set("rr", e.target.value)} />
            </Field>
            <Field label="SpO₂" hint="%">
              <input type="number" className={inputCls} placeholder="e.g. 94" value={form.spo2} onChange={(e) => set("spo2", e.target.value)} />
            </Field>
            <Field label="SBP" hint="mmHg">
              <input type="number" className={inputCls} placeholder="e.g. 138" value={form.sbp} onChange={(e) => set("sbp", e.target.value)} />
            </Field>
            <Field label="DBP" hint="mmHg">
              <input type="number" className={inputCls} placeholder="e.g. 88" value={form.dbp} onChange={(e) => set("dbp", e.target.value)} />
            </Field>
            <Field label="Temperature" hint="°F">
              <input type="number" step="0.1" className={inputCls} placeholder="e.g. 98.7" value={form.temperature} onChange={(e) => set("temperature", e.target.value)} />
            </Field>
          </div>
        </div>

        <div className="flex gap-3 pt-2">
          <button
            onClick={onClose}
            className="flex-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-raised)] transition"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={loading}
            className="flex-1 rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--color-brand-600)] disabled:opacity-50 transition"
          >
            {loading ? "Adding…" : "Add Patient to Queue"}
          </button>
        </div>
      </div>
    </div>
  );
}
