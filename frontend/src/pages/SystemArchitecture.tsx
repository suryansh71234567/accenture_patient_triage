const STEPS = [
  { label: "RECONCILIATION", hue: null },
  { label: "ACUITY / RISK", tone: "good" as const },
  { label: "CLINICAL DEPARTMENT", hue: null },
];

export function SystemArchitecture() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center gap-1.5 p-8">
      <h1 className="mb-2 self-start text-sm font-bold text-[var(--color-ink)]">System Architecture</h1>

      <Node>PATIENT</Node>
      <Down />
      <div className="grid w-full grid-cols-2 gap-2.5">
        <SubNode label="XGBoost" sub="quantitative risk" hue={250} />
        <SubNode label="RAG + LLM" sub="historical / contextual" hue={300} />
      </div>
      <Down />
      {STEPS.map((s) => (
        <div key={s.label} className="flex w-full flex-col items-center gap-1.5">
          <Node tone={s.tone}>{s.label}</Node>
          <Down />
        </div>
      ))}
      <Node tone="good">FINAL PATIENT FLOW</Node>

      <div className="mt-5 w-full self-stretch rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3.5">
        <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Multi-Hospital Isolation</p>
        <p className="text-[11.5px] leading-relaxed text-[var(--color-ink)]">
          Each hospital maintains its own configuration, capacity, and routing policy. Clinical assessment is shared;
          operational placement is hospital-specific. Simulation state runs independently per hospital.
        </p>
      </div>
    </div>
  );
}

function Node({ children, tone }: { children: React.ReactNode; tone?: "good" }) {
  return (
    <div
      className="w-full rounded-lg px-4 py-3 text-center text-xs font-bold"
      style={tone === "good" ? { background: "var(--color-good-50)", color: "var(--color-good-600)" } : { background: "var(--color-surface-muted)" }}
    >
      {children}
    </div>
  );
}

function SubNode({ label, sub, hue }: { label: string; sub: string; hue: number }) {
  return (
    <div className="rounded-lg px-3 py-3 text-center" style={{ background: `oklch(94% 0.03 ${hue})`, color: `oklch(45% 0.15 ${hue})` }}>
      <p className="text-[11.5px] font-bold">{label}</p>
      <p className="mt-0.5 text-[9.5px] font-medium" style={{ color: "var(--color-ink-faint)" }}>{sub}</p>
    </div>
  );
}

function Down() {
  return <span className="text-[var(--color-ink-faint)]">↓</span>;
}
