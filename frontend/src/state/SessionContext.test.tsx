import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  api: {
    createSession: vi.fn(),
    getSessionState: vi.fn(),
    chat: vi.fn(),
    executeTool: vi.fn(),
    confirmTool: vi.fn(),
  },
}));

// SessionContext.tsx deliberately keeps `sessionBootstrap` as MODULE-scoped
// state (not component state) specifically to survive React StrictMode's
// double-mount — but that same design means it persists across tests within
// one file too. vi.resetModules() + a fresh dynamic import per test is the
// correct way to test a module-scoped singleton in isolation; without it,
// only the FIRST test's bootstrap would ever actually run.
async function freshSession() {
  vi.resetModules();
  const apiModule = await import("../api/client");
  const sessionModule = await import("./SessionContext");
  return { api: apiModule.api, SessionProvider: sessionModule.SessionProvider, useSession: sessionModule.useSession };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

describe("SessionContext — bootstrap", () => {
  it("mints a new session when localStorage has none, and persists it", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "new-1", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });

    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.sessionId).toBe("new-1");
    expect(api.getSessionState).not.toHaveBeenCalled();
    expect(localStorage.getItem("triageguard.session_id")).toBe("new-1");
  });

  it("restores an existing session_id from localStorage via getSessionState, without minting a new one", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    localStorage.setItem("triageguard.session_id", "existing-1");
    (api.getSessionState as ReturnType<typeof vi.fn>).mockResolvedValue({});
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });

    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.sessionId).toBe("existing-1");
    expect(api.createSession).not.toHaveBeenCalled();
  });

  it("a stale session (backend lost it) falls through to minting a fresh one, clearing the stale id first", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    localStorage.setItem("triageguard.session_id", "stale-1");
    (api.getSessionState as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("404 Unknown session_id"));
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "fresh-2", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });

    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.sessionId).toBe("fresh-2");
    expect(localStorage.getItem("triageguard.session_id")).toBe("fresh-2");
  });

  it("backend fully unavailable: still becomes ready, with a null sessionId rather than hanging forever", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("Network error"));
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });

    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.sessionId).toBeNull();
  });
});

describe("SessionContext — StrictMode double-mount protection", () => {
  it("only issues ONE createSession call even when effects run twice under StrictMode", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "strict-1", role: "nurse" });

    function Probe() {
      const { ready, sessionId } = useSession();
      return <div>{ready ? `ready:${sessionId}` : "loading"}</div>;
    }

    render(
      <StrictMode>
        <SessionProvider>
          <Probe />
        </SessionProvider>
      </StrictMode>
    );

    await waitFor(() => expect(screen.getByText("ready:strict-1")).toBeInTheDocument());
    expect(api.createSession).toHaveBeenCalledTimes(1);
  });
});

describe("SessionContext — hospital selection", () => {
  it("defaults to 'default' and persists a switch to localStorage", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "s1", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.hospitalId).toBe("default");
    act(() => result.current.setHospitalId("hosp-2"));
    expect(result.current.hospitalId).toBe("hosp-2");
    expect(localStorage.getItem("triageguard.hospital_id")).toBe("hosp-2");
  });

  it("restores a previously-selected hospital from localStorage on next mount", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    localStorage.setItem("triageguard.hospital_id", "hosp-9");
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "s1", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.hospitalId).toBe("hosp-9");
  });
});

describe("SessionContext — proposeAction / executeTool / confirmTool", () => {
  async function readySession() {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "s1", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });
    await waitFor(() => expect(result.current.ready).toBe(true));
    return { api, result };
  }

  it("an action that executes immediately (no confirmation needed) bumps mutationTick and resolves right away", async () => {
    const { api, result } = await readySession();
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "executed", data: { ok: true }, error: null });

    const tickBefore = result.current.mutationTick;
    let outcome: unknown;
    await act(async () => {
      outcome = await result.current.proposeAction("some_tool", { a: 1 });
    });
    expect(api.executeTool).toHaveBeenCalledWith("s1", "some_tool", { a: 1 });
    expect(outcome).toEqual({ status: "executed", data: { ok: true }, error: null });
    expect(result.current.mutationTick).toBe(tickBefore + 1);
    expect(result.current.pendingDirectAction).toBeNull();
  });

  it("an action requiring confirmation sets pendingDirectAction and does NOT resolve until confirmTool completes", async () => {
    const { api, result } = await readySession();
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "awaiting_confirmation", data: null, error: null, description: "Admit P-1 to ICU?",
    });
    (api.confirmTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "Admitted.", response_type: "confirmation", patient_id: "P-1", actions: [], evidence: [], human_approval_required: false,
    });

    let resolved = false;
    let outcomePromise!: Promise<unknown>;
    act(() => {
      outcomePromise = result.current.proposeAction("admit_simulated_patient", { patient_id: "P-1" });
      outcomePromise.then(() => { resolved = true; });
    });

    await waitFor(() => expect(result.current.pendingDirectAction).toEqual({
      tool_name: "admit_simulated_patient", kwargs: { patient_id: "P-1" }, description: "Admit P-1 to ICU?",
    }));
    expect(resolved).toBe(false); // still awaiting the nurse's confirm/cancel

    const tickBefore = result.current.mutationTick;
    await act(async () => {
      await result.current.resolveDirectAction(true);
    });

    expect(api.confirmTool).toHaveBeenCalledWith("s1", true);
    expect(result.current.pendingDirectAction).toBeNull();
    expect(result.current.mutationTick).toBe(tickBefore + 1);
    await expect(outcomePromise).resolves.toEqual({
      status: "executed",
      data: { message: "Admitted.", actions: [] },
      error: null,
    });
  });

  it("rejecting (Cancel) does not bump mutationTick and resolves the original caller with a failed-shaped outcome", async () => {
    const { api, result } = await readySession();
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "awaiting_confirmation", data: null, error: null, description: "Admit P-1 to ICU?",
    });
    (api.confirmTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "Cancelled by nurse.", response_type: "error", patient_id: "P-1", actions: [], evidence: [], human_approval_required: false,
    });

    let outcomePromise!: Promise<{ status: string }>;
    act(() => {
      outcomePromise = result.current.proposeAction("admit_simulated_patient", { patient_id: "P-1" });
    });
    await waitFor(() => expect(result.current.pendingDirectAction).not.toBeNull());

    const tickBefore = result.current.mutationTick;
    await act(async () => {
      await result.current.resolveDirectAction(false);
    });

    expect(api.confirmTool).toHaveBeenCalledWith("s1", false);
    expect(result.current.mutationTick).toBe(tickBefore); // no mutation on rejection
    const outcome = await outcomePromise;
    expect(outcome.status).toBe("failed");
  });

  it("resolvingDirectAction is true only while confirmTool is in flight", async () => {
    const { api, result } = await readySession();
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "awaiting_confirmation", data: null, error: null, description: "d",
    });
    let resolveConfirm!: (v: unknown) => void;
    (api.confirmTool as ReturnType<typeof vi.fn>).mockReturnValue(new Promise((r) => { resolveConfirm = r; }));

    act(() => { void result.current.proposeAction("t", {}); });
    await waitFor(() => expect(result.current.pendingDirectAction).not.toBeNull());

    let resolvePromise!: Promise<unknown>;
    act(() => {
      resolvePromise = result.current.resolveDirectAction(true);
    });
    await waitFor(() => expect(result.current.resolvingDirectAction).toBe(true));

    await act(async () => {
      resolveConfirm({ message: "ok", response_type: "confirmation", patient_id: null, actions: [], evidence: [], human_approval_required: false });
      await resolvePromise;
    });
    expect(result.current.resolvingDirectAction).toBe(false);
  });

  it("anyPending reflects a pending direct action", async () => {
    const { api, result } = await readySession();
    expect(result.current.anyPending).toBe(false);
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "awaiting_confirmation", data: null, error: null, description: "d",
    });
    act(() => { void result.current.proposeAction("t", {}); });
    await waitFor(() => expect(result.current.anyPending).toBe(true));
  });

  it("DOCUMENTED BEHAVIOR: a second proposeAction while one is already awaiting confirmation orphans the first caller's promise (last-writer-wins on the resolver)", async () => {
    // Not fixed — the real UI already prevents this via PendingActionModal's
    // blocking overlay once pendingDirectAction is set, so no known real
    // trigger path exists today; this test exists so a future change that
    // removes that UI-level protection doesn't silently reintroduce a hang
    // without anyone noticing the underlying architecture allows it.
    const { api, result } = await readySession();
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "awaiting_confirmation", data: null, error: null, description: "d",
    });
    (api.confirmTool as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "second wins", response_type: "confirmation", patient_id: null, actions: [], evidence: [], human_approval_required: false,
    });

    let firstResolved = false;
    let secondResolved = false;
    act(() => {
      result.current.proposeAction("tool_a", {}).then(() => { firstResolved = true; });
    });
    await waitFor(() => expect(result.current.pendingDirectAction?.tool_name).toBe("tool_a"));

    act(() => {
      result.current.proposeAction("tool_b", {}).then(() => { secondResolved = true; });
    });
    await waitFor(() => expect(result.current.pendingDirectAction?.tool_name).toBe("tool_b"));

    await act(async () => {
      await result.current.resolveDirectAction(true);
    });

    expect(secondResolved).toBe(true);
    expect(firstResolved).toBe(false); // orphaned — documented, not asserted as desirable
  });
});

describe("SessionContext — chat", () => {
  async function readySession() {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "s1", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });
    await waitFor(() => expect(result.current.ready).toBe(true));
    return { api, result };
  }

  it("sendChat appends the user entry immediately and the agent's reply after it resolves", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "Hospital is at 62% load.", response_type: "text", patient_id: null, actions: [], evidence: [], human_approval_required: false,
    });
    await act(async () => {
      await result.current.sendChat("What's the hospital status?");
    });
    expect(api.chat).toHaveBeenCalledWith("s1", "What's the hospital status?", "default");
    expect(result.current.history).toHaveLength(2);
    expect(result.current.history[0]).toMatchObject({ kind: "user", text: "What's the hospital status?" });
    expect(result.current.history[1]).toMatchObject({ kind: "agent" });
    expect(result.current.chatBusy).toBe(false);
  });

  it("a thrown network error is caught and surfaced as a synthesized error entry, not an unhandled rejection", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("fetch failed"));
    await act(async () => {
      await result.current.sendChat("hello");
    });
    expect(result.current.history[1].kind).toBe("agent");
    if (result.current.history[1].kind === "agent") {
      expect(result.current.history[1].response.response_type).toBe("error");
      expect(result.current.history[1].response.message).toContain("fetch failed");
    }
    expect(result.current.chatBusy).toBe(false);
  });

  it("a response requiring approval sets chatAwaitingConfirmation and marks that entry as the live one", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "Confirm admitting P-1?", response_type: "approval_required", patient_id: "P-1", actions: [], evidence: [], human_approval_required: true,
    });
    await act(async () => {
      await result.current.sendChat("admit P-1");
    });
    expect(result.current.chatAwaitingConfirmation).toBe(true);
    expect(result.current.awaitingEntryId).toBe(result.current.history[1].id);
  });

  it("sending the next message (including yes/no) retires a previous live confirmation card", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      message: "Confirm?", response_type: "approval_required", patient_id: null, actions: [], evidence: [], human_approval_required: true,
    });
    await act(async () => { await result.current.sendChat("do something"); });
    expect(result.current.awaitingEntryId).not.toBeNull();

    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      message: "ok", response_type: "text", patient_id: null, actions: [], evidence: [], human_approval_required: false,
    });
    await act(async () => { await result.current.sendChat("yes"); });
    expect(result.current.awaitingEntryId).toBeNull();
  });

  it("resolveChatConfirmation(true) sends 'yes'; resolveChatConfirmation(false) sends 'no'", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "ok", response_type: "text", patient_id: null, actions: [], evidence: [], human_approval_required: false,
    });
    await act(async () => { await result.current.resolveChatConfirmation(true); });
    expect(api.chat).toHaveBeenCalledWith("s1", "yes", "default");
    await act(async () => { await result.current.resolveChatConfirmation(false); });
    expect(api.chat).toHaveBeenCalledWith("s1", "no", "default");
  });

  it("clearChat empties history and resets confirmation state", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
      message: "Confirm?", response_type: "approval_required", patient_id: null, actions: [], evidence: [], human_approval_required: true,
    });
    await act(async () => { await result.current.sendChat("hi"); });
    expect(result.current.history.length).toBeGreaterThan(0);

    act(() => result.current.clearChat());
    expect(result.current.history).toEqual([]);
    expect(result.current.chatAwaitingConfirmation).toBe(false);
    expect(result.current.awaitingEntryId).toBeNull();
  });

  it("bumps mutationTick when an action actually executed, but not for a plain text response", async () => {
    const { api, result } = await readySession();
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      message: "just chatting", response_type: "text", patient_id: null, actions: [], evidence: [], human_approval_required: false,
    });
    const tickBefore = result.current.mutationTick;
    await act(async () => { await result.current.sendChat("hi"); });
    expect(result.current.mutationTick).toBe(tickBefore);

    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      message: "done", response_type: "text", patient_id: null,
      actions: [{ tool: "admit_simulated_patient", status: "executed", data: { ok: true } }],
      evidence: [], human_approval_required: false,
    });
    await act(async () => { await result.current.sendChat("admit them"); });
    expect(result.current.mutationTick).toBe(tickBefore + 1);
  });
});

describe("SessionContext — bumpMutationTick", () => {
  it("increments mutationTick for callers outside proposeAction/sendChat (e.g. plain REST writes)", async () => {
    const { api, SessionProvider, useSession } = await freshSession();
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "s1", role: "nurse" });
    const { result } = renderHook(() => useSession(), {
      wrapper: ({ children }) => <SessionProvider>{children}</SessionProvider>,
    });
    await waitFor(() => expect(result.current.ready).toBe(true));
    const before = result.current.mutationTick;
    act(() => result.current.bumpMutationTick());
    expect(result.current.mutationTick).toBe(before + 1);
  });
});
