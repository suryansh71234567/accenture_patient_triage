import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { EmptyState, Spinner } from "../components/ui";
import type { AssessmentResult } from "../types";

type QueueEntry = {
  patient_id: string;
  chief_complaint: string;
  status: string;
  clinical_assessment?: AssessmentResult | null;
};

/**
 * Rather than letting a dropdown pick ANY patient and silently trigger a
 * fresh (potentially slow) XGBoost+RAG assessment just from browsing, this
 * screen only lists patients this hospital's simulation has already
 * assessed — clinical_assessment already rides along on every TRIAGED
 * full_queue entry, so nothing extra is fetched to populate it.
 */
export function ClinicalIntelligence() {
  const { hospitalId } = useSession();
  const { data: dash, loading } = usePoll(() => api.dashboard(hospitalId), 8000, [hospitalId]);
  const [selected, setSelected] = useState<string | null>(null);

  if (loading && !dash) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
      </div>
    );
  }
  if (!dash) return <EmptyState title="Could not reach the hospital simulation service." />;

  const assessed = ((dash.full_queue as unknown as QueueEntry[]) ?? []).filter(
    (p) => p.status === "TRIAGED" && p.clinical_assessment
  );
  const current = assessed.find((p) => p.patient_id === selected) ?? assessed[0] ?? null;
  const a = current?.clinical_assessment;

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-8">
      <h1 className="text-sm font-bold text-[var(--color-ink)]">Clinical Intelligence</h1>

      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
        <div className="flex flex-wrap items-center justify-center gap-2.5">
          <PipeNode label="PATIENT" />
          <Arrow />
          <div className="flex flex-col gap-1.5">
            <SubNode label="XGBoost" sub="quantitative risk" hue={250} />
            <SubNode label="RAG + LLM" sub="contextual reasoning" hue={300} />
          </div>
          <Arrow />
          <PipeNode label="RECONCILIATION" />
          <Arrow />
          <PipeNode label="ACUITY / RISK" tone="good" />
          <Arrow />
          <PipeNode label="CLINICAL DEPARTMENT" />
        </div>
      </div>

      {assessed.length === 0 ? (
        <p className="text-xs text-[var(--color-ink-faint)]">
          No patients with detailed clinical intelligence available in this hospital right now.
        </p>
      ) : current && a ? (
        <>
          <div className="flex items-center gap-2.5">
            <span className="text-[11px] text-[var(--color-ink-faint)]">Patient</span>
            <select
              value={current.patient_id}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-lg border border-[var(--color-border)] px-2.5 py-1.5 font-mono text-xs font-semibold"
            >
              {assessed.map((p) => (
                <option key={p.patient_id} value={p.patient_id}>{p.patient_id}</option>
              ))}
            </select>
            <span className="text-[11.5px] text-[var(--color-ink-soft)]">{current.chief_complaint}</span>
          </div>

          <div className="grid grid-cols-1 gap-3.5 md:grid-cols-2">
            <div className="flex flex-col gap-2.5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
              <p className="text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Acuity Tier</p>
              <p className="text-[15px] font-bold text-[var(--color-ink)]">{a.acuity_tier}</p>
              <div className="mt-1 grid grid-cols-2 gap-2">
                <Stat label="Admission Risk" value={`${Math.round(a.reconciled_admission_risk * 100)}%`} />
                <Stat label="ICU Risk" value={`${Math.round(a.reconciled_icu_risk * 100)}%`} />
              </div>
              <p className="text-[11px] text-[var(--color-ink-soft)]">{a.confidence_note}</p>
              <p className="text-[10px] font-semibold" style={{ color: a.branches_agree ? "var(--color-good-600)" : "var(--color-warn-600)" }}>
                {a.branches_agree ? "Models agree" : "Models disagree — treated conservatively"}
              </p>
            </div>
            <div className="flex flex-col gap-2.5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
              <div>
                <p className="mb-1 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Top Diagnoses</p>
                {a.top_diagnoses.map((dx, i) => <p key={i} className="text-xs">• {dx}</p>)}
              </div>
              <div>
                <p className="mb-1 text-[10px] font-bold uppercase tracking-wide text-[var(--color-critical-600)]">Red Flags</p>
                {a.red_flags.length === 0
                  ? <p className="text-xs text-[var(--color-ink-faint)]">None</p>
                  : a.red_flags.map((rf, i) => <p key={i} className="text-xs text-[var(--color-critical-600)]">⚠ {rf}</p>)}
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
            <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Department Reasoning</p>
            <p className="mb-3 text-xs leading-relaxed text-[var(--color-ink)]">{a.department_reasoning}</p>
            {a.rag_narrative && (
              <>
                <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide" style={{ color: "var(--color-retriage-600)" }}>RAG Narrative</p>
                <p className="rounded-lg bg-[var(--color-surface-muted)] px-3 py-2.5 text-xs leading-relaxed text-[var(--color-ink-soft)]">{a.rag_narrative}</p>
              </>
            )}
          </div>
        </>
      ) : null}
    </div>
  );
}

function PipeNode({ label, tone }: { label: string; tone?: "good" }) {
  return (
    <div
      className="rounded-lg px-4 py-2.5 text-center text-[11px] font-bold"
      style={tone === "good" ? { background: "var(--color-good-50)", color: "var(--color-good-600)" } : { background: "var(--color-surface-muted)" }}
    >
      {label}
    </div>
  );
}

function SubNode({ label, sub, hue }: { label: string; sub: string; hue: number }) {
  return (
    <div className="rounded-lg px-3.5 py-2 text-center" style={{ background: `oklch(94% 0.03 ${hue})`, color: `oklch(45% 0.15 ${hue})` }}>
      <p className="text-[10.5px] font-bold">{label}</p>
      <p className="text-[9px] font-medium" style={{ color: "var(--color-ink-faint)" }}>{sub}</p>
    </div>
  );
}

function Arrow() {
  return <span className="text-[var(--color-ink-faint)]">→</span>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-[var(--color-surface-muted)] px-2.5 py-2">
      <p className="text-[9px] text-[var(--color-ink-faint)]">{label}</p>
      <p className="font-mono text-[15px] font-semibold text-[var(--color-ink)]">{value}</p>
    </div>
  );
}
