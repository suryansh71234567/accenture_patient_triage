import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { ChatDock } from "./ChatDock";
import { HospitalSelector } from "./HospitalSelector";
import { PendingActionModal } from "./PendingActionModal";
import { useSession } from "../state/SessionContext";

const NAV = [
  { to: "/", label: "Dashboard", icon: "◧" },
  { to: "/live", label: "Live Hospital", icon: "▣" },
  { to: "/patients", label: "Patients", icon: "☰" },
  { to: "/hospitals", label: "Hospitals", icon: "⛨" },
];

export function Layout() {
  const [chatOpen, setChatOpen] = useState(true);
  const { ready } = useSession();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--color-canvas)]">
      {/* Sidebar */}
      <nav className="flex w-56 shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-2 px-5 py-5">
          <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-[var(--color-brand-500)] text-base text-white">
            +
          </span>
          <div>
            <p className="text-sm font-bold leading-tight text-[var(--color-ink)]">TriageGuard</p>
            <p className="text-[10px] leading-tight text-[var(--color-ink-faint)]">Clinical triage assistant</p>
          </div>
        </div>
        <div className="px-5 pb-4">
          <label className="mb-1 block text-[10px] font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">
            Hospital
          </label>
          <HospitalSelector />
        </div>
        <div className="flex flex-col gap-0.5 px-3">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-[var(--color-brand-50)] text-[var(--color-brand-700)]"
                    : "text-[var(--color-ink-soft)] hover:bg-slate-50"
                }`
              }
            >
              <span className="text-base">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </div>
        <div className="mt-auto flex items-center gap-2 border-t border-[var(--color-border)] px-5 py-4">
          <span className={`h-2 w-2 rounded-full ${ready ? "bg-[var(--color-good-500)]" : "bg-slate-300"}`} />
          <p className="text-[11px] text-[var(--color-ink-faint)]">{ready ? "Connected" : "Connecting…"}</p>
        </div>
      </nav>

      {/* Main content */}
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>

      {/* Chat dock */}
      <div
        className={`shrink-0 border-l border-[var(--color-border)] transition-all ${chatOpen ? "w-[380px]" : "w-0"}`}
      >
        {chatOpen && <ChatDock />}
      </div>

      <button
        onClick={() => setChatOpen((o) => !o)}
        className="fixed bottom-5 right-5 z-10 flex h-11 w-11 items-center justify-center rounded-full bg-[var(--color-brand-500)] text-white shadow-lg hover:bg-[var(--color-brand-600)]"
        style={{ right: chatOpen ? 396 : 20 }}
        title={chatOpen ? "Hide assistant" : "Show assistant"}
      >
        {chatOpen ? "›" : "✦"}
      </button>

      <PendingActionModal />
    </div>
  );
}
