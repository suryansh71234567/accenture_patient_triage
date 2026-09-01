import { useModalA11y } from "../hooks/useModalA11y";
import { useSession } from "../state/SessionContext";
import { Button, Spinner } from "./ui";

export function PendingActionModal() {
  const { pendingDirectAction, resolvingDirectAction, resolveDirectAction } = useSession();
  if (!pendingDirectAction) return null;

  // Extracted so useModalA11y's mount-once effect only runs while the
  // dialog is actually present — this component itself stays mounted for
  // the app's whole lifetime (rendered unconditionally in Layout.tsx), so
  // calling the hook here directly would either violate the Rules of Hooks
  // (it'd have to be skipped whenever pendingDirectAction is null) or leave
  // its Escape listener attached globally even with no dialog to close.
  return (
    <PendingActionDialog
      description={pendingDirectAction.description}
      resolving={resolvingDirectAction}
      onResolve={resolveDirectAction}
    />
  );
}

function PendingActionDialog({
  description,
  resolving,
  onResolve,
}: {
  description: string;
  resolving: boolean;
  onResolve: (approve: boolean) => void;
}) {
  // Cancel is disabled while resolving; Escape mirrors that same gating.
  const containerRef = useModalA11y(resolving ? () => {} : () => onResolve(false));

  return (
    <div ref={containerRef} tabIndex={-1} className="fixed inset-0 z-40 flex items-center justify-center bg-black/30 backdrop-blur-[2px]" role="dialog" aria-modal="true">
      <div className="w-[420px] rounded-2xl bg-white p-5 shadow-xl">
        <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-warn-600)]">
          Confirmation required
        </p>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-ink)]">{description}</p>

        {resolving && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-[var(--color-surface-muted)] px-3 py-2 text-xs text-[var(--color-ink-soft)]">
            <Spinner className="h-3.5 w-3.5 text-[var(--color-brand-500)]" />
            Applying and re-running the clinical assessment — this can take up to a minute…
          </div>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={() => onResolve(false)} disabled={resolving}>
            Cancel
          </Button>
          <Button onClick={() => onResolve(true)} disabled={resolving}>
            {resolving ? "Working…" : "Confirm"}
          </Button>
        </div>
      </div>
    </div>
  );
}
