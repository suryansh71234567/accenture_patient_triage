import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
  title,
  subtitle,
  right,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className={`rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-sm ${className}`}>
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
        <span className={`text-xs font-semibold ${emphasized ? "text-base" : ""}`} style={{ color: "var(--color-ink)" }}>
          {value == null ? "—" : `${pct}%`}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${barColor} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function DeptGauge({
  name,
  occupied,
  capacity,
  available,
  status,
}: {
  name: string;
  occupied: number;
  capacity: number;
  available: number;
  status?: string;
}) {
  const pct = capacity > 0 ? Math.min(100, Math.round((occupied / capacity) * 100)) : 0;
  const tone = pct >= 95 ? "critical" : pct >= 80 ? "warn" : "good";
  const barColor =
    tone === "critical"
      ? "bg-[var(--color-critical-500)]"
      : tone === "warn"
        ? "bg-[var(--color-warn-500)]"
        : "bg-[var(--color-good-500)]";
  const closed = status === "CLOSED";

  return (
    <div className="rounded-xl border border-[var(--color-border)] p-3.5">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-semibold text-[var(--color-ink)]">{DEPT_LABELS[name] ?? name}</span>
        {closed ? (
          <Badge tone="neutral">Closed</Badge>
        ) : tone === "critical" ? (
          <Badge tone="critical" dot>
            Full
          </Badge>
        ) : tone === "warn" ? (
          <Badge tone="warn" dot>
            High load
          </Badge>
        ) : (
          <Badge tone="good" dot>
            Available
          </Badge>
        )}
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${closed ? "bg-slate-300" : barColor}`} style={{ width: `${closed ? 100 : pct}%` }} />
      </div>
      <div className="mt-1.5 flex items-center justify-between text-xs text-[var(--color-ink-faint)]">
        <span>
          {occupied}/{capacity} occupied
        </span>
        <span>{available} open</span>
      </div>
    </div>
  );
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
      <div className={`mt-1 text-xl font-semibold ${danger ? "text-[var(--color-critical-600)]" : "text-[var(--color-ink)]"}`}>
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
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-[var(--color-border)] py-14 text-center">
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
