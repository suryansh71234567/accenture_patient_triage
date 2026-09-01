import { useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { Button, EmptyState, Spinner, formatOperatingMode } from "../components/ui";

export function SimulationControlCenter() {
  const { hospitalId } = useSession();
  const { data: dash, loading, refetch } = usePoll(() => api.dashboard(hospitalId), 4000, [hospitalId]);
  const { data: scenarios } = usePoll(() => api.scenarios(hospitalId), 60000, [hospitalId]);
  const [busy, setBusy] = useState(false);

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

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-8">
      <h1 className="text-sm font-bold text-[var(--color-ink)]">
        Simulation Control Center — {dash.scenario.title}
      </h1>

      <div className="flex flex-wrap items-center gap-5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-5 py-4">
        <div>
          <p className="whitespace-nowrap text-[9.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">Simulation Time</p>
          <p className="font-mono text-[26px] font-semibold text-[var(--color-ink)]">{dash.time}</p>
        </div>
        <div className="flex gap-2">
          <Button disabled={busy} onClick={() => run(() => api.step(5, true, hospitalId))}>Step +5 min</Button>
          <Button variant="secondary" disabled={busy} onClick={() => run(() => api.step(15, true, hospitalId))}>Step +15 min</Button>
          <Button variant="secondary" disabled={busy} onClick={() => run(() => api.triggerArrival(undefined, hospitalId))}>Manual Arrival</Button>
        </div>
      </div>
      <p className="text-[10px] text-[var(--color-ink-faint)]">
        Simulation advances only when stepped from here — there is no continuous backend clock.
      </p>

      <div className="flex flex-wrap items-center gap-3.5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3.5">
        <div className="min-w-[300px] flex-1">
          <p className="mb-1.5 whitespace-nowrap text-[9.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">1. Scenario</p>
          <div className="grid grid-cols-2 gap-2">
            {scenarios?.map((s) => (
              <button
                key={s.name}
                disabled={busy}
                onClick={() => run(() => api.loadScenario(s.name, hospitalId))}
                className="flex flex-col gap-0.5 rounded-lg border px-2.5 py-2 text-left"
                style={
                  s.name === dash.scenario.name
                    ? { borderColor: "var(--color-brand-500)", background: "var(--color-brand-50)", color: "var(--color-brand-700)" }
                    : { borderColor: "var(--color-border)" }
                }
              >
                <span className="text-[11.5px] font-bold">{s.title}</span>
                <span className="text-[10px] leading-snug text-[var(--color-ink-faint)]">{s.description}</span>
              </button>
            ))}
          </div>
        </div>
        <span className="text-[var(--color-ink-faint)]">→</span>
        <div>
          <p className="mb-1.5 whitespace-nowrap text-[9.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">2. Load Ratio (system-calculated)</p>
          <p className="font-mono text-lg font-semibold text-[var(--color-ink)]">{dash.load.lambda.toFixed(2)}×</p>
        </div>
        <span className="text-[var(--color-ink-faint)]">→</span>
        <div>
          <p className="mb-1.5 whitespace-nowrap text-[9.5px] font-bold uppercase tracking-wide text-[var(--color-ink-faint)]">3. Operating Mode (derived)</p>
          <span className="inline-block rounded-full px-3 py-1 text-xs font-bold" style={{ background: "var(--color-brand-50)", color: "var(--color-brand-700)" }}>
            {formatOperatingMode(dash.load.operating_mode)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <SimStat label="Untriaged" value={dash.untriaged_count} colorVar="--color-warn-500" />
        <SimStat label="Triaged" value={dash.triaged_count} colorVar="--color-brand-500" />
        <SimStat label="Admitted" value={dash.admitted_count} colorVar="--color-good-500" />
      </div>

      <div>
        <p className="mb-2 text-xs font-bold text-[var(--color-ink)]">Recent Events</p>
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
          {[...dash.recent_events].reverse().map((e, i) => (
            <div key={i} className="border-b border-[var(--color-border)] px-3.5 py-2.5 text-[11.5px] text-[var(--color-ink)] last:border-0">
              {e}
            </div>
          ))}
          {dash.recent_events.length === 0 && (
            <div className="px-3.5 py-2.5 text-[11.5px] text-[var(--color-ink-faint)]">No events yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function SimStat({ label, value, colorVar }: { label: string; value: number; colorVar: string }) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3.5">
      <p className="text-[9.5px] uppercase text-[var(--color-ink-faint)]">{label}</p>
      <p className="font-mono text-2xl font-semibold" style={{ color: `var(${colorVar})` }}>{value}</p>
    </div>
  );
}
