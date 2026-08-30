import { useEffect, useRef, useState } from "react";
import { useSession } from "../state/SessionContext";
import { Badge, Button, DEPT_LABELS, Spinner } from "./ui";
import type { AgentAction, AgentResponse, ChatEntry } from "../types";

const SUGGESTIONS = [
  "What's the current hospital status?",
  "Which patients need attention first?",
  "Show me the current queues",
  "Which department is most constrained?",
  "Are any patients flagged for review?",
];

export function ChatDock() {
  const { history, chatBusy, chatAwaitingConfirmation, awaitingEntryId, sendChat, resolveChatConfirmation, ready, clearChat } =
    useSession();
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, chatBusy]);

  const submit = () => {
    const text = draft.trim();
    if (!text || chatBusy) return;
    setDraft("");
    sendChat(text);
  };

  return (
    <aside className="flex h-full w-full flex-col bg-[var(--color-surface)]">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3.5">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--color-brand-500)] text-sm text-white">
            ✦
          </span>
          <div>
            <p className="text-sm font-semibold text-[var(--color-ink)]">TriageGuard Assistant</p>
            <p className="text-[11px] text-[var(--color-ink-faint)]">Hospital operations & triage copilot — in plain language</p>
          </div>
        </div>
        <button
          onClick={clearChat}
          className="text-[11px] text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]"
          title="Clear conversation"
        >
          Clear
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {history.length === 0 && (
          <div className="space-y-3">
            <p className="text-xs leading-relaxed text-[var(--color-ink-faint)]">
              I can help with hospital-wide status, queues, routing explanations, and individual patients. I explain
              recommendations — you make the call. Try:
            </p>
            <div className="flex flex-col gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => sendChat(s)}
                  className="rounded-lg border border-[var(--color-border)] px-3 py-2 text-left text-xs text-[var(--color-ink-soft)] hover:border-[var(--color-brand-500)] hover:text-[var(--color-brand-600)]"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {history.map((entry) => (
          <ChatBubble
            key={entry.id}
            entry={entry}
            onConfirm={resolveChatConfirmation}
            busy={chatBusy}
            isLiveConfirmation={entry.id === awaitingEntryId}
          />
        ))}

        {chatBusy && (
          <div className="flex items-center gap-2 text-xs text-[var(--color-ink-faint)]">
            <Spinner className="h-3.5 w-3.5" />
            Thinking…
          </div>
        )}
      </div>

      <div className="border-t border-[var(--color-border)] p-3">
        {chatAwaitingConfirmation && (
          <p className="mb-2 text-[11px] font-medium text-[var(--color-warn-600)]">
            Awaiting your confirmation above — reply below or use the buttons.
          </p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={ready ? "Message the agent…" : "Connecting…"}
            disabled={!ready}
            rows={2}
            className="min-h-[42px] flex-1 resize-none rounded-xl border border-[var(--color-border)] px-3 py-2 text-sm outline-none focus:border-[var(--color-brand-500)]"
          />
          <Button onClick={submit} disabled={!ready || chatBusy || !draft.trim()}>
            Send
          </Button>
        </div>
      </div>
    </aside>
  );
}

function ChatBubble({
  entry,
  onConfirm,
  busy,
  isLiveConfirmation,
}: {
  entry: ChatEntry;
  onConfirm: (approve: boolean) => void;
  busy: boolean;
  isLiveConfirmation: boolean;
}) {
  if (entry.kind === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-[var(--color-brand-500)] px-3.5 py-2 text-sm text-white">
          {entry.text}
        </div>
      </div>
    );
  }

  const r = entry.response;
  const isError = r.response_type === "error";
  const isApproval = r.response_type === "approval_required";

  return (
    <div className="flex flex-col items-start gap-2">
      <div
        className={`max-w-[92%] rounded-2xl rounded-bl-sm px-3.5 py-2.5 text-sm ${
          isError
            ? "bg-[var(--color-critical-50)] text-[var(--color-critical-600)]"
            : "bg-[var(--color-surface-muted)] text-[var(--color-ink)]"
        }`}
      >
        <p className="whitespace-pre-wrap leading-relaxed">{displayMessage(r)}</p>
      </div>

      {isApproval && !isLiveConfirmation && (
        <div className="w-[92%] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-muted)] p-3">
          <p className="text-xs text-[var(--color-ink-faint)]">This confirmation has already been resolved.</p>
        </div>
      )}

      {isApproval && isLiveConfirmation && (
        <div className="w-[92%] rounded-xl border border-[var(--color-warn-100)] bg-[var(--color-warn-50)] p-3">
          <p className="mb-2 text-xs font-semibold text-[var(--color-warn-600)]">Confirmation needed</p>
          <div className="flex gap-2">
            <Button size="sm" onClick={() => onConfirm(true)} disabled={busy}>
              Confirm
            </Button>
            <Button size="sm" variant="secondary" onClick={() => onConfirm(false)} disabled={busy}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {!isApproval && r.actions.length > 0 && <ActionSummaries actions={r.actions} />}
    </div>
  );
}

function ActionSummaries({ actions }: { actions: AgentAction[] }) {
  const meaningful = actions.filter((a) => a.status === "executed" && a.data);
  if (meaningful.length === 0) return null;
  return (
    <div className="w-[92%] space-y-2">
      {meaningful.map((a, i) => (
        <ActionCard key={i} action={a} />
      ))}
    </div>
  );
}

function ActionCard({ action }: { action: AgentAction }) {
  const d = action.data as Record<string, any>;

  if (action.tool === "add_patient_observation") {
    if (d.duplicate) {
      return (
        <div className="rounded-xl border border-[var(--color-border)] bg-white p-3 text-xs">
          <Badge tone="neutral">No change</Badge>
          <p className="mt-1.5 text-[var(--color-ink-soft)]">
            {d.observation_type?.replace("_", " ")} is already {d.new_value} — nothing new recorded.
          </p>
        </div>
      );
    }
    return (
      <div className="rounded-xl border border-[var(--color-border)] bg-white p-3 text-xs">
        <Badge tone="teal">Observation recorded</Badge>
        <p className="mt-1.5 text-[var(--color-ink-soft)]">
          {d.observation_type?.replace("_", " ")}: <span className="font-semibold text-[var(--color-ink)]">{d.previous_value ?? "—"}</span> →{" "}
          <span className="font-semibold text-[var(--color-ink)]">{d.new_value}</span> {d.unit}
        </p>
        {d.timestamp && (
          <p className="mt-0.5 text-[10px] text-[var(--color-ink-faint)]">{new Date(d.timestamp).toLocaleTimeString()}</p>
        )}
      </div>
    );
  }

  if (action.tool === "run_triage_assessment" || action.tool === "triage_simulated_patient") {
    return (
      <div className="rounded-xl border border-[var(--color-border)] bg-white p-3 text-xs">
        <Badge tone="brand">Updated assessment</Badge>
        <div className="mt-1.5 grid grid-cols-2 gap-1.5 text-[var(--color-ink-soft)]">
          <span>
            Department: <span className="font-semibold text-[var(--color-ink)]">{DEPT_LABELS[d.department] ?? d.department}</span>
          </span>
          <span>
            Admission risk: <span className="font-semibold text-[var(--color-ink)]">{pct(d.reconciled_admission_risk)}</span>
          </span>
          <span>
            ICU risk: <span className="font-semibold text-[var(--color-ink)]">{pct(d.reconciled_icu_risk)}</span>
          </span>
          <span>
            Branches agree: <span className="font-semibold text-[var(--color-ink)]">{d.branches_agree ? "Yes" : "No"}</span>
          </span>
        </div>
      </div>
    );
  }

  if (action.tool === "get_hospital_state") {
    return (
      <div className="rounded-xl border border-[var(--color-border)] bg-white p-3 text-xs text-[var(--color-ink-soft)]">
        <Badge tone="neutral">Hospital state fetched</Badge>
      </div>
    );
  }

  return null;
}

function pct(v: unknown): string {
  if (typeof v !== "number") return "—";
  return `${Math.round(v * 100)}%`;
}

/**
 * The runtime's own post-confirmation message is built with an f-string that
 * appends `{result.data}` / `{reassessment.data}` — a raw Python dict repr —
 * after "Result:" / "Updated assessment:" (see
 * AgentRuntime._handle_pending_confirmation). The real structured data is
 * already rendered separately by ActionSummaries below the bubble, so here
 * we only need to not show the dict dump a second time as prose. This is
 * display-only truncation — it never changes what data the UI shows, only
 * where duplicate raw text is hidden.
 */
function displayMessage(r: AgentResponse): string {
  if (r.response_type !== "confirmation") return r.message;
  const cut = r.message.search(/\b(Result|Updated assessment):/);
  return cut === -1 ? r.message : r.message.slice(0, cut).trim();
}
