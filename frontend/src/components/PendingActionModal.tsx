import { useSession } from "../state/SessionContext";
import { Button, Spinner } from "./ui";

export function PendingActionModal() {
  const { pendingDirectAction, resolvingDirectAction, resolveDirectAction } = useSession();
  if (!pendingDirectAction) return null;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/30 backdrop-blur-[2px]">
      <div className="w-[420px] rounded-2xl bg-white p-5 shadow-xl">
        <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-warn-600)]">
          Confirmation required
        </p>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-ink)]">{pendingDirectAction.description}</p>

        {resolvingDirectAction && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-[var(--color-surface-muted)] px-3 py-2 text-xs text-[var(--color-ink-soft)]">
            <Spinner className="h-3.5 w-3.5 text-[var(--color-brand-500)]" />
            Applying and re-running the clinical assessment — this can take up to a minute…
          </div>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={() => resolveDirectAction(false)} disabled={resolvingDirectAction}>
            Cancel
          </Button>
          <Button onClick={() => resolveDirectAction(true)} disabled={resolvingDirectAction}>
            {resolvingDirectAction ? "Working…" : "Confirm"}
          </Button>
        </div>
      </div>
    </div>
  );
}
