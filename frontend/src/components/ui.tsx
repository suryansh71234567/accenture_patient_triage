import type { CSSProperties, ReactNode } from "react";

export function Card({
  children,
  className = "",
  style,
  title,
  subtitle,
  right,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  title?: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div style={style} className={`rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-sm ${className}`}>
      {(title || right) && (
        <div className="flex items-start justify-between gap-3 border-b border-[var(--color-border)] px-5 py-3.5">
          <div>
            {title && <h3 className="text-[15px] font-semibold text-[var(--color-ink)]">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-[var(--color-ink-faint)]">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

type Tone = "neutral" | "brand" | "critical" | "warn" | "good" | "teal";

const toneClasses: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-700 ring-1 ring-slate-200",
  brand: "bg-[var(--color-brand-50)] text-[var(--color-brand-700)] ring-1 ring-[var(--color-brand-100)]",
  critical: "bg-[var(--color-critical-50)] text-[var(--color-critical-600)] ring-1 ring-[var(--color-critical-100)]",
  warn: "bg-[var(--color-warn-50)] text-[var(--color-warn-600)] ring-1 ring-[var(--color-warn-100)]",
  good: "bg-[var(--color-good-50)] text-[var(--color-good-600)] ring-1 ring-[var(--color-good-100)]",
  teal: "bg-[var(--color-teal-50)] text-[var(--color-teal-600)] ring-1 ring-teal-100",
};

export function Badge({
  children,
  tone = "neutral",
  dot = false,
  className = "",
}: {
  children: ReactNode;
  tone?: Tone;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${toneClasses[tone]} ${className}`}
    >
      {dot && <span className={`h-1.5 w-1.5 rounded-full ${dotColor(tone)}`} />}
      {children}
    </span>
  );
}

function dotColor(tone: Tone): string {
  switch (tone) {
    case "critical":
      return "bg-[var(--color-critical-500)]";
    case "warn":
      return "bg-[var(--color-warn-500)]";
    case "good":
      return "bg-[var(--color-good-500)]";
    case "brand":
      return "bg-[var(--color-brand-500)]";
    case "teal":
      return "bg-[var(--color-teal-500)]";
    default:
      return "bg-slate-400";
  }
}

export function acuityTone(acuity: number | null | undefined): Tone {
  if (acuity == null) return "neutral";
  if (acuity <= 1) return "critical";
  if (acuity === 2) return "warn";
  if (acuity === 3) return "brand";
  return "good";
}

/** Left-border accent color for a patient card, keyed off the same acuity Tone used for its badge. */
export function acuityBorderColor(tone: Tone): string {
  switch (tone) {
    case "critical":
      return "var(--color-critical-500)";
    case "warn":
      return "var(--color-warn-500)";
    case "brand":
      return "var(--color-brand-500)";
    case "good":
      return "var(--color-good-500)";
    default:
      return "var(--color-border)";
  }
}

/** "42m" under an hour, "1h 12m" at/above — used for sim-clock wait times (sim_time_minutes - arrival_time_min). */
export function fmtWaitMinutes(min: number): string {
  if (min < 60) return `${min}m`;
  return `${Math.floor(min / 60)}h ${min % 60}m`;
}

/** "HIGH_LOAD" -> "High Load" — the backend's raw operating_mode enum, humanized for display. */
export function formatOperatingMode(mode: string): string {
  return mode
    .toLowerCase()
    .split("_")
    .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}

export function acuityLabel(acuity: number | null | undefined): string {
  switch (acuity) {
    case 1:
      return "Critical";
    case 2:
      return "High Priority";
    case 3:
      return "Moderate";
    case 4:
      return "Low-Moderate";
    case 5:
      return "Minor";
    default:
      return "Unassessed";
  }
}

/** Design-exact 5-way ESI-style acuity palette (distinct color per level, unlike the 4-bucket
 * acuityTone/acuityLabel above which PatientList/PatientWorkspace still use unchanged). Scoped to
 * Dashboard/Live Hospital and the shared modals they open. */
export const ACUITY_META: Record<number, { label: string; color: string; bg: string }> = {
  1: { label: "Resuscitation", color: "oklch(50% 0.19 25)", bg: "oklch(94% 0.045 25)" },
  2: { label: "Emergent", color: "oklch(56% 0.17 45)", bg: "oklch(94% 0.04 45)" },
  3: { label: "Urgent", color: "oklch(58% 0.13 85)", bg: "oklch(94% 0.035 85)" },
  4: { label: "Less Urgent", color: "oklch(55% 0.1 205)", bg: "oklch(93% 0.02 205)" },
  5: { label: "Non-Urgent", color: "oklch(52% 0.12 150)", bg: "oklch(93% 0.03 150)" },
};
const ACUITY_META_UNASSESSED = { label: "Unassessed", color: "oklch(60% 0.01 255)", bg: "oklch(95% 0.004 250)" };

export function acuityMeta(acuity: number | null | undefined): { label: string; color: string; bg: string } {
  if (acuity == null) return ACUITY_META_UNASSESSED;
  return ACUITY_META[acuity] ?? ACUITY_META_UNASSESSED;
}

/** Small acuity pill using the design's exact 5-color palette — "A{n} · {label}" or just "A{n}". */
export function AcuityPill({ acuity, withLabel = true, className = "" }: { acuity: number | null | undefined; withLabel?: boolean; className?: string }) {
  const meta = acuityMeta(acuity);
  return (
    <span
      className={`inline-flex items-center rounded-[5px] px-[7px] py-[2px] text-[9.5px] font-bold ${className}`}
      style={{ background: meta.bg, color: meta.color }}
    >
      {acuity != null ? `A${acuity}` : "—"}
      {withLabel ? ` · ${meta.label}` : ""}
    </span>
  );
}

export function RiskBar({
  label,
  value,
  emphasized = false,
}: {
  label: string;
  value: number | null | undefined;
  emphasized?: boolean;
}) {
  const pct = value == null ? 0 : Math.round(value * 100);
  const tone = pct >= 65 ? "critical" : pct >= 35 ? "warn" : "good";
  const barColor =
    tone === "critical"
      ? "bg-[var(--color-critical-500)]"
      : tone === "warn"
        ? "bg-[var(--color-warn-500)]"
        : "bg-[var(--color-good-500)]";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className={`text-xs ${emphasized ? "font-semibold text-[var(--color-ink)]" : "text-[var(--color-ink-soft)]"}`}>
          {label}
        </span>
        <span className={`font-mono text-xs font-semibold ${emphasized ? "text-base" : ""}`} style={{ color: "var(--color-ink)" }}>
          {value == null ? "—" : `${pct}%`}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${barColor} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export type DeptTone = "good" | "warn" | "critical";

/** Single source of truth for department occupancy → status/color, shared by
 * DeptGauge and DepartmentQueueBoard's column headers so both read the same
 * OPEN/RESTRICTED/CLOSED-style thresholds off the same real capacity data. */
export function deptStatusTone(
  occupied: number,
  capacity: number,
  status?: string
): { tone: DeptTone; label: string; pct: number; closed: boolean } {
  const closed = status === "CLOSED";
  const pct = capacity > 0 ? Math.min(100, Math.round((occupied / capacity) * 100)) : 0;
  if (closed) return { tone: "critical", label: "Closed", pct: 100, closed: true };
  if (pct >= 95) return { tone: "critical", label: "Full", pct, closed: false };
  if (pct >= 80) return { tone: "warn", label: "High load", pct, closed: false };
  return { tone: "good", label: "Available", pct, closed: false };
}

/** Design-exact OPEN/RESTRICTED/CLOSED status (100%/85% thresholds), additive alongside
 * deptStatusTone above (which HospitalNetwork.tsx — out of scope — still relies on unchanged).
 * Used by DeptGauge and DepartmentQueueBoard's column headers. */
export function deptStatus(
  occupied: number,
  capacity: number
): { label: "OPEN" | "RESTRICTED" | "CLOSED"; color: string; bg: string; pct: number } {
  const pct = capacity > 0 ? occupied / capacity : 0;
  if (pct >= 1) return { label: "CLOSED", color: "oklch(50% 0.18 25)", bg: "oklch(94% 0.04 25)", pct: 100 };
  if (pct >= 0.85) return { label: "RESTRICTED", color: "oklch(56% 0.15 65)", bg: "oklch(94% 0.035 65)", pct: Math.round(pct * 100) };
  return { label: "OPEN", color: "oklch(48% 0.13 150)", bg: "oklch(94% 0.03 150)", pct: Math.round(pct * 100) };
}

export function DeptGauge({
  name,
  occupied,
  capacity,
  available,
}: {
  name: string;
  occupied: number;
  capacity: number;
  available: number;
  status?: string;
}) {
  const st = deptStatus(occupied, capacity);

  return (
    <div className="rounded-xl border border-[var(--color-border)] px-[18px] py-4">
      <div className="mb-2.5 flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-bold text-[var(--color-ink)]">{DEPT_LABELS[name] ?? name}</span>
        <span
          className="rounded-[5px] px-[7px] py-[2px] text-[9.5px] font-bold tracking-[.05em]"
          style={{ background: st.bg, color: st.color }}
        >
          {st.label}
        </span>
      </div>
      <div className="mb-1.5 h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full" style={{ width: `${st.pct}%`, background: st.color }} />
      </div>
      <div className="flex items-center justify-between text-[11.5px] text-[var(--color-ink-faint)]">
        <span className="font-mono">
          {occupied}/{capacity} beds
        </span>
        <span>{available} open</span>
      </div>
    </div>
  );
}

/** Some presimulated/demo patient records never populate the newer
 * ai_operational_department field (only the live triage path in
 * hospital_simulator.py does) — clinical_department is always present and is
 * the real AI-preferred department before operational/capacity adjustment,
 * so it's the correct real-data fallback rather than showing a blank. */
export function aiDeptOf(decision: { ai_operational_department?: string | null; clinical_department: string }): string {
  return decision.ai_operational_department ?? decision.clinical_department;
}

export const DEPT_LABELS: Record<string, string> = {
  ICU: "ICU",
  CICU: "Cardiac ICU",
  ADMITTED_GEN: "General Ward",
  ED_OBS: "ED Observation",
  DISCHARGE: "Discharge",
};

export function VitalTile({
  icon,
  label,
  value,
  unit,
  danger,
}: {
  icon: string;
  label: string;
  value: number | string | null | undefined;
  unit?: string;
  danger?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-3 ${danger ? "border-[var(--color-critical-100)] bg-[var(--color-critical-50)]" : "border-[var(--color-border)] bg-[var(--color-surface-muted)]"}`}
    >
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-[var(--color-ink-faint)]">
        <span>{icon}</span>
        <span>{label}</span>
      </div>
      <div className={`mt-1 font-mono text-xl font-semibold ${danger ? "text-[var(--color-critical-600)]" : "text-[var(--color-ink)]"}`}>
        {value ?? "—"}
        {value != null && unit && <span className="ml-1 text-sm font-normal text-[var(--color-ink-faint)]">{unit}</span>}
      </div>
    </div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}

export function EmptyState({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-14 text-center">
      <p className="text-sm font-medium text-[var(--color-ink-soft)]">{title}</p>
      {subtitle && <p className="mt-1 max-w-sm text-xs text-[var(--color-ink-faint)]">{subtitle}</p>}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  size = "md",
  type = "button",
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "danger" | "ghost";
  disabled?: boolean;
  size?: "sm" | "md";
  type?: "button" | "submit";
  className?: string;
}) {
  const base = "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition disabled:cursor-not-allowed disabled:opacity-50";
  const sizeCls = size === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2 text-sm";
  const variantCls: Record<string, string> = {
    primary: "bg-[var(--color-brand-500)] text-white hover:bg-[var(--color-brand-600)]",
    secondary: "bg-white text-[var(--color-ink)] ring-1 ring-[var(--color-border)] hover:bg-slate-50",
    danger: "bg-[var(--color-critical-500)] text-white hover:bg-[var(--color-critical-600)]",
    ghost: "text-[var(--color-ink-soft)] hover:bg-slate-100",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${sizeCls} ${variantCls[variant]} ${className}`}
    >
      {children}
    </button>
  );
}
