import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { useSession } from "../state/SessionContext";
import { formatOperatingMode } from "./ui";
import type { HospitalInfo } from "../types";

const MODE_DOT: Record<string, string> = {
  NORMAL: "bg-[var(--color-good-500)]",
  HIGH_LOAD: "bg-[var(--color-warn-500)]",
  CRITICAL: "bg-[var(--color-critical-500)]",
};

/**
 * Header dropdown for the hospital the app is scoped to. Shows a live
 * operating-mode dot + active-patient count for the *currently selected*
 * hospital only (via the same api.dashboard() poll Dashboard/LiveHospital
 * already use) — other hospitals in the list show name only. Fetching each
 * hospital's own status just to decorate this dropdown would mean N extra
 * dashboard() calls on every single page load app-wide; that N-calls cost is
 * accepted for the dedicated Hospital Network screen, not for a header
 * control rendered on every route.
 */
export function HospitalSelector() {
  const { hospitalId, setHospitalId, mutationTick } = useSession();
  const [hospitals, setHospitals] = useState<HospitalInfo[]>([]);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const { data: dash } = usePoll(() => api.dashboard(hospitalId), 10000, [hospitalId, mutationTick]);

  useEffect(() => {
    let cancelled = false;
    api
      .listHospitals()
      .then((list) => {
        if (!cancelled) setHospitals(list);
      })
      .catch(() => {
        // Non-fatal — selector just falls back to showing the current hospitalId alone.
      });
    return () => {
      cancelled = true;
    };
  }, [mutationTick]);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const knownIds = new Set(hospitals.map((h) => h.hospital_id));
  const currentName = hospitals.find((h) => h.hospital_id === hospitalId)?.hospital_name ?? hospitalId;
  const dotClass = dash ? (MODE_DOT[dash.load.operating_mode] ?? "bg-[var(--color-brand-500)]") : "bg-slate-300";
  const activeCount = dash ? dash.untriaged_count + dash.triaged_count + dash.admitted_count : null;

  const rows: HospitalInfo[] = knownIds.has(hospitalId)
    ? hospitals
    : [{ hospital_id: hospitalId, hospital_name: hospitalId, config_path: "" }, ...hospitals];

  return (
    <div className="relative" ref={rootRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Active hospital"
        className="flex w-full items-center gap-2.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-left"
      >
        <span className={`h-2 w-2 shrink-0 rounded-full ${dotClass}`} />
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-[12.5px] font-semibold text-[var(--color-ink)]">{currentName}</span>
          <span className="truncate text-[10px] text-[var(--color-ink-faint)]">
            {dash ? `${formatOperatingMode(dash.load.operating_mode)} · ${activeCount} active` : "…"}
          </span>
        </span>
        <span className="shrink-0 text-[9px] text-[var(--color-ink-faint)]">▾</span>
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute left-0 top-[calc(100%+6px)] z-30 w-[280px] overflow-hidden rounded-[10px] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[0_12px_32px_rgba(20,25,40,.14)]"
        >
          {rows.map((h) => (
            <button
              key={h.hospital_id}
              role="option"
              aria-selected={h.hospital_id === hospitalId}
              onClick={() => {
                setHospitalId(h.hospital_id);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2.5 border-b border-[var(--color-border)] px-3.5 py-2.5 text-left last:border-0"
              style={{ background: h.hospital_id === hospitalId ? "var(--color-surface-muted)" : "transparent" }}
            >
              <span
                className={`h-2 w-2 shrink-0 rounded-full ${h.hospital_id === hospitalId ? dotClass : "bg-slate-300"}`}
              />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[12.5px] font-semibold text-[var(--color-ink)]">
                  {h.hospital_name}
                </span>
                <span className="block truncate font-mono text-[10px] text-[var(--color-ink-faint)]">
                  {h.hospital_id}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
