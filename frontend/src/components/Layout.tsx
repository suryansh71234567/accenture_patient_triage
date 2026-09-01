import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import { usePoll } from "../hooks/usePoll";
import { ChatDock } from "./ChatDock";
import { HospitalSelector } from "./HospitalSelector";
import { PendingActionModal } from "./PendingActionModal";
import { formatOperatingMode } from "./ui";
import { useSession } from "../state/SessionContext";

// Primary nav is scoped to the four screens that carry real demo/nurse
// value (Phase 5 product-surface cleanup). Simulation, System Architecture,
// and standalone Clinical Intelligence remain fully functional at their
// existing routes (see App.tsx) — only their top-nav entry point is gone.
// Simulation's unique capability (time-stepping) moved into Live Hospital;
// Clinical Intelligence's content already lives in the patient drawer/modals.
const NAV = [
  { to: "/", label: "Overview" },
  { to: "/network", label: "Hospital Network" },
  { to: "/live", label: "Live Operations" },
  { to: "/patients", label: "Patients" },
];

export function Layout() {
  const [chatOpen, setChatOpen] = useState(true);
  const { ready, hospitalId } = useSession();
  // Header-level capacity-warning pill: reuses the same api.dashboard() poll
  // pattern every page already uses, rather than inventing a second data
  // source for "is this hospital under load."
  const { data: dash } = usePoll(() => api.dashboard(hospitalId), 10000, [hospitalId]);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[var(--color-canvas)]">
      {/* Header */}
      <header className="flex h-[60px] shrink-0 items-center gap-5 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-5">
        <div className="flex items-center gap-2.5">
          <span className="flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-md bg-[var(--color-brand-500)]">
            <span className="h-2.5 w-2.5 bg-white" style={{ clipPath: "polygon(50% 0%, 100% 100%, 0% 100%)" }} />
          </span>
          <div>
            <p className="text-[15px] font-bold leading-tight tracking-tight text-[var(--color-ink)]">TriageGuard</p>
            <p className="text-[9.5px] font-medium uppercase leading-tight tracking-wide text-[var(--color-ink-faint)]">
              ED Operations Platform
            </p>
          </div>
        </div>

        <div className="h-7 w-px bg-[var(--color-border)]" />

        <div className="w-[250px] shrink-0">
          <HospitalSelector />
        </div>

        <div className="flex-1" />

        {dash && dash.load.operating_mode !== "NORMAL" && (
          <span
            className="flex items-center gap-[7px] rounded-lg border px-3 py-[6px] text-[11.5px] font-semibold"
            style={{ borderColor: "oklch(80% 0.06 65)", background: "oklch(95% 0.035 65)", color: "oklch(40% 0.1 60)" }}
          >
            ⚠ {formatOperatingMode(dash.load.operating_mode)} load
          </span>
        )}

        <div className="flex items-center gap-1.5" aria-live="polite">
          <span className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-[var(--color-good-500)]" : "bg-slate-300"}`} />
          <span className="text-[10.5px] font-semibold text-[var(--color-ink-faint)]">
            {ready ? "Connected" : "Connecting…"}
          </span>
        </div>

        <button
          onClick={() => setChatOpen((o) => !o)}
          className="flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] px-3.5 py-2 text-xs font-bold transition"
          style={
            chatOpen
              ? { borderColor: "var(--color-brand-500)", background: "var(--color-brand-50)", color: "var(--color-brand-700)" }
              : { background: "var(--color-surface)", color: "var(--color-ink-soft)" }
          }
        >
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-brand-500)]" />
          Ops Assistant
        </button>
      </header>

      {/* Secondary nav */}
      <nav className="flex h-11 shrink-0 items-center gap-1 overflow-x-auto border-b border-[var(--color-border)] bg-[var(--color-surface)] px-5">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `shrink-0 border-b-2 px-3 py-1.5 text-[12px] font-semibold transition ${
                isActive
                  ? "border-[var(--color-brand-500)] text-[var(--color-ink)]"
                  : "border-transparent text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-muted)]"
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Main content */}
      <main className="min-h-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>

      {/* Assistant overlay panel */}
      {chatOpen && (
        <div className="fixed bottom-0 right-0 top-[104px] z-30 w-[380px] border-l border-[var(--color-border)] shadow-[-12px_0_30px_rgba(0,0,0,.08)]">
          <ChatDock />
        </div>
      )}

      <PendingActionModal />
    </div>
  );
}
