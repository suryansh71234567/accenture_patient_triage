import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "../api/client";
import type { AgentResponse, ChatEntry, ToolExecuteResult } from "../types";

const SESSION_STORAGE_KEY = "triageguard.session_id";

interface PendingDirectAction {
  tool_name: string;
  kwargs: Record<string, unknown>;
  description: string;
}

interface SessionContextValue {
  sessionId: string | null;
  ready: boolean;
  role: string;

  // Conversational chat (agent as primary interface)
  history: ChatEntry[];
  chatBusy: boolean;
  chatAwaitingConfirmation: boolean;
  /** id of the single chat entry whose Confirm/Cancel buttons are still live. */
  awaitingEntryId: string | null;
  sendChat: (text: string) => Promise<void>;
  resolveChatConfirmation: (approve: boolean) => Promise<void>;
  clearChat: () => void;

  // Direct structured UI actions (forms/buttons), still going through the
  // same ToolExecutor + ConfirmationProtocol approval gate as chat.
  pendingDirectAction: PendingDirectAction | null;
  resolvingDirectAction: boolean;
  proposeAction: (
    tool_name: string,
    kwargs: Record<string, unknown>
  ) => Promise<ToolExecuteResult>;
  resolveDirectAction: (approve: boolean) => Promise<ToolExecuteResult>;

  anyPending: boolean;

  // Bumped every time a write action commits, so pages can refetch.
  mutationTick: number;
}

const SessionContext = createContext<SessionContextValue | null>(null);

let idCounter = 0;
const nextId = () => `entry-${++idCounter}-${Date.now()}`;

// Module-scoped so React StrictMode's dev-only double effect invocation
// can't race two POST /api/session calls into two different session ids.
let sessionBootstrap: Promise<string | null> | null = null;

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [role] = useState("nurse");
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatAwaitingConfirmation, setChatAwaitingConfirmation] = useState(false);
  const [awaitingEntryId, setAwaitingEntryId] = useState<string | null>(null);
  const [pendingDirectAction, setPendingDirectAction] = useState<PendingDirectAction | null>(null);
  const [resolvingDirectAction, setResolvingDirectAction] = useState(false);
  const [mutationTick, setMutationTick] = useState(0);
  const directResolver = useRef<((r: ToolExecuteResult) => void) | null>(null);

  useEffect(() => {
    let cancelled = false;

    if (!sessionBootstrap) {
      sessionBootstrap = (async () => {
        const existing = localStorage.getItem(SESSION_STORAGE_KEY);
        if (existing) {
          try {
            await api.getSessionState(existing);
            return existing;
          } catch {
            // Stale session_id (e.g. backend restarted and lost its
            // in-memory SESSIONS dict) — fall through and mint a new one.
            localStorage.removeItem(SESSION_STORAGE_KEY);
          }
        }
        try {
          const created = await api.createSession(role);
          localStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
          return created.session_id;
        } catch {
          return null;
        }
      })();
    }

    sessionBootstrap.then((id) => {
      if (cancelled) return;
      setSessionId(id);
      setReady(true);
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendChat = useCallback(
    async (text: string) => {
      if (!sessionId || !text.trim()) return;
      setHistory((h) => [...h, { kind: "user", text, id: nextId() }]);
      setChatBusy(true);
      // Whatever the nurse just sent (including "yes"/"no") resolves or
      // supersedes any previously live confirmation card — never leave a
      // stale Confirm/Cancel pair clickable once the turn has moved on.
      setAwaitingEntryId(null);
      try {
        const response: AgentResponse = await api.chat(sessionId, text);
        const entryId = nextId();
        setHistory((h) => [...h, { kind: "agent", response, id: entryId }]);
        setChatAwaitingConfirmation(response.human_approval_required);
        setAwaitingEntryId(response.human_approval_required ? entryId : null);
        if (
          response.response_type === "confirmation" ||
          response.actions.some((a) => a.status === "executed")
        ) {
          setMutationTick((t) => t + 1);
        }
      } catch (err) {
        setHistory((h) => [
          ...h,
          {
            kind: "agent",
            response: {
              message: `Connection error: ${(err as Error).message}`,
              response_type: "error",
              patient_id: null,
              actions: [],
              evidence: [],
              human_approval_required: false,
            },
            id: nextId(),
          },
        ]);
      } finally {
        setChatBusy(false);
      }
    },
    [sessionId]
  );

  const resolveChatConfirmation = useCallback(
    async (approve: boolean) => {
      await sendChat(approve ? "yes" : "no");
    },
    [sendChat]
  );

  const clearChat = useCallback(() => {
    setHistory([]);
    setChatAwaitingConfirmation(false);
    setAwaitingEntryId(null);
  }, []);

  const proposeAction = useCallback(
    async (tool_name: string, kwargs: Record<string, unknown>): Promise<ToolExecuteResult> => {
      if (!sessionId) throw new Error("Session not ready");
      const result = await api.executeTool(sessionId, tool_name, kwargs);
      if (result.status === "awaiting_confirmation") {
        setPendingDirectAction({
          tool_name,
          kwargs,
          description: result.description ?? `${tool_name}(${JSON.stringify(kwargs)})`,
        });
        return new Promise<ToolExecuteResult>((resolve) => {
          directResolver.current = resolve;
        });
      }
      if (result.status === "executed") setMutationTick((t) => t + 1);
      return result;
    },
    [sessionId]
  );

  const resolveDirectAction = useCallback(
    async (approve: boolean): Promise<ToolExecuteResult> => {
      if (!sessionId) throw new Error("Session not ready");
      setResolvingDirectAction(true);
      try {
        // Confirming a write (e.g. add_patient_observation) runs synchronously
        // through to a fresh triage reassessment server-side, which can take
        // tens of seconds for a real XGBoost+RAG call — the modal stays open
        // with a busy state for that whole window rather than looking stuck.
        const response: AgentResponse = await api.confirmTool(sessionId, approve);
        setPendingDirectAction(null);
        const outcome: ToolExecuteResult = {
          status: response.response_type === "error" ? "failed" : "executed",
          data: { message: response.message, actions: response.actions },
          error:
            response.response_type === "error"
              ? { code: "CONFIRM_FAILED", message: response.message }
              : null,
        };
        if (approve) setMutationTick((t) => t + 1);
        directResolver.current?.(outcome);
        directResolver.current = null;
        return outcome;
      } finally {
        setResolvingDirectAction(false);
      }
    },
    [sessionId]
  );

  const value: SessionContextValue = {
    sessionId,
    ready,
    role,
    history,
    chatBusy,
    chatAwaitingConfirmation,
    awaitingEntryId,
    sendChat,
    resolveChatConfirmation,
    clearChat,
    pendingDirectAction,
    resolvingDirectAction,
    proposeAction,
    resolveDirectAction,
    anyPending: chatAwaitingConfirmation || pendingDirectAction !== null,
    mutationTick,
  };

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used within SessionProvider");
  return ctx;
}
