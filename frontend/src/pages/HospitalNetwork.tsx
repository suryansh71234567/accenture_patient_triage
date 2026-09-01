import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useSession } from "../state/SessionContext";
import { HospitalOnboardingWizard } from "../components/HospitalOnboardingWizard";
import { EmptyState, Spinner, deptStatusTone, formatOperatingMode } from "../components/ui";
import type { HospitalInfo, SimulationDashboard } from "../types";

const MODE_DOT: Record<string, string> = {
  NORMAL: "bg-[var(--color-good-500)]",
  HIGH_LOAD: "bg-[var(--color-warn-500)]",
  CRITICAL: "bg-[var(--color-critical-500)]",
};

type NetworkRow = { info: HospitalInfo; dash: SimulationDashboard | null; calibrated: boolean | null };

/**
 * Builds as designed (approved): one dashboard() call per registered
 * hospital, sequentially. /api/hospitals itself carries no capacity/mode/
 * patient-count data, so there is no batch alternative today — this is an
 * accepted N-calls-per-render cost, fine for a handful of hospitals in a
 * demo, not something to scale past without a real batch endpoint.
 */
export function HospitalNetwork() {
  const { hospitalId, setHospitalId } = useSession();
  const [rows, setRows] = useState<NetworkRow[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [configureTarget, setConfigureTarget] = useState<{ hospital_id: string; hospital_name: string } | null>(null);
  const [reloadTick, setReloadTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      const list = await api.listHospitals().catch(() => [] as HospitalInfo[]);
      const built: NetworkRow[] = [];
      for (const info of list) {
        if (cancelled) return;
        const [dash, status] = await Promise.all([
          api.dashboard(info.hospital_id).catch(() => null),
          api.calibrationStatus(info.hospital_id).catch(() => null),
        ]);
        built.push({ info, dash, calibrated: status?.calibrated ?? null });
      }
      if (!cancelled) {
        setRows(built);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [reloadTick]);

  if (loading && !rows) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-[var(--color-brand-500)]" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-sm font-bold text-[var(--color-ink)]">Hospital Network</h1>
        <button
          onClick={() => setShowAdd(true)}
          className="rounded-lg bg-[var(--color-brand-500)] px-4 py-2 text-sm font-bold text-white shadow-sm hover:bg-[var(--color-brand-600)]"
        >
          + Add Hospital
        </button>
      </div>

      {!rows || rows.length === 0 ? (
        <EmptyState title="No hospitals registered yet." subtitle='Use "+ Add Hospital" to register one.' />
      ) : (
        <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
          {rows.map((r) => {
            const active = r.dash ? r.dash.untriaged_count + r.dash.triaged_count + r.dash.admitted_count : null;
            const isCurrent = r.info.hospital_id === hospitalId;
            const warn = r.dash && r.dash.load.operating_mode !== "NORMAL";
            return (
              <div
                key={r.info.hospital_id}
                className="flex flex-col gap-2.5 rounded-xl border bg-[var(--color-surface)] p-4"
                style={{ borderColor: isCurrent ? "var(--color-brand-500)" : "var(--color-border)" }}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <p className="truncate text-sm font-bold text-[var(--color-ink)]">{r.info.hospital_name}</p>
                  {r.dash && <span className={`h-2 w-2 shrink-0 rounded-full ${MODE_DOT[r.dash.load.operating_mode] ?? "bg-slate-300"}`} />}
                </div>
                <p className="font-mono text-[10.5px] text-[var(--color-ink-faint)]">{r.info.hospital_id}</p>
                <p className="text-[11px] text-[var(--color-ink-faint)]">
                  {r.dash ? `${formatOperatingMode(r.dash.load.operating_mode)} · ${active} active patients` : "Status unavailable"}
                </p>
                {warn && (
                  <p className="rounded-md px-2 py-1 text-[10.5px]" style={{ background: "var(--color-warn-50)", color: "var(--color-warn-600)" }}>
                    ⚠ Elevated load — escalation sensitivity increased
                  </p>
                )}
                <span
                  className="w-fit rounded-full px-2 py-0.5 text-[9px] font-bold"
                  style={
                    r.calibrated
                      ? { background: "var(--color-good-50)", color: "var(--color-good-600)" }
                      : { background: "var(--color-warn-50)", color: "var(--color-warn-600)" }
                  }
                >
                  {r.calibrated ? "CALIBRATED" : "NOT CALIBRATED"}
                </span>

                {r.dash && (
                  <div className="mt-1 flex flex-col gap-1.5">
                    {r.dash.departments.map((d) => {
                      const s = deptStatusTone(d.occupied, d.capacity, d.status);
                      return (
                        <div key={d.name}>
                          <div className="mb-0.5 flex justify-between text-[10.5px] text-[var(--color-ink-faint)]">
                            <span className="truncate">{d.name}</span>
                            <span className="font-mono">{d.occupied}/{d.capacity}</span>
                          </div>
                          <div className="h-[5px] w-full overflow-hidden rounded-full bg-slate-200">
                            <div
                              className={`h-full rounded-full ${s.tone === "critical" ? "bg-[var(--color-critical-500)]" : s.tone === "warn" ? "bg-[var(--color-warn-500)]" : "bg-[var(--color-good-500)]"}`}
                              style={{ width: `${s.closed ? 100 : s.pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                <div className="mt-1.5 flex gap-2">
                  <button
                    onClick={() => setHospitalId(r.info.hospital_id)}
                    className="flex-1 rounded-lg border px-3 py-2 text-xs font-bold"
                    style={{ borderColor: "var(--color-brand-500)", color: "var(--color-brand-600)", background: isCurrent ? "var(--color-brand-50)" : "transparent" }}
                  >
                    {isCurrent ? "Selected" : "Select"}
                  </button>
                  <button
                    onClick={() => setConfigureTarget({ hospital_id: r.info.hospital_id, hospital_name: r.info.hospital_name })}
                    className="flex-1 rounded-lg border border-[var(--color-border)] px-3 py-2 text-xs font-semibold text-[var(--color-ink-soft)]"
                  >
                    Configure Policy
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {showAdd && (
        <HospitalOnboardingWizard
          onClose={() => {
            setShowAdd(false);
            setReloadTick((t) => t + 1);
          }}
        />
      )}

      {configureTarget && (
        <HospitalOnboardingWizard
          existingHospital={configureTarget}
          onClose={() => {
            setConfigureTarget(null);
            setReloadTick((t) => t + 1);
          }}
        />
      )}
    </div>
  );
}
