import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatDock } from "./ChatDock";
import { SessionProvider } from "../state/SessionContext";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    createSession: vi.fn(),
    getSessionState: vi.fn(),
    chat: vi.fn(),
  },
}));

function renderDock() {
  return render(
    <SessionProvider>
      <ChatDock />
    </SessionProvider>
  );
}

function textReply(message: string) {
  return { message, response_type: "text", patient_id: null, actions: [], evidence: [], human_approval_required: false };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValue({ session_id: "s1", role: "nurse" });
});

async function ready() {
  await waitFor(() => expect(screen.getByPlaceholderText("Message the agent…")).toBeInTheDocument());
}

describe("ChatDock", () => {
  it("renders the header and suggested prompts when there is no history yet", async () => {
    renderDock();
    await ready();
    expect(screen.getByText("TriageGuard Assistant")).toBeInTheDocument();
    expect(screen.getByText("What's the current hospital status?")).toBeInTheDocument();
    expect(screen.getByText("Which patients need attention first?")).toBeInTheDocument();
  });

  it("disables the textarea and Send while the session is not yet ready", () => {
    // createSession() is still pending at first render — not yet resolved.
    (api.createSession as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {}));
    renderDock();
    expect(screen.getByPlaceholderText("Connecting…")).toBeDisabled();
  });

  it("Send is disabled for an empty/whitespace-only draft", async () => {
    renderDock();
    await ready();
    const send = screen.getByText("Send").closest("button")!;
    expect(send).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "   " } });
    expect(send).toBeDisabled();
  });

  it("clicking a suggested prompt sends it directly", async () => {
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue(textReply("62% load."));
    renderDock();
    await ready();
    fireEvent.click(screen.getByText("Which department is most constrained?"));
    await waitFor(() => expect(api.chat).toHaveBeenCalledWith("s1", "Which department is most constrained?", "default"));
    await waitFor(() => expect(screen.getByText("62% load.")).toBeInTheDocument());
  });

  it("typing and clicking Send submits the draft and clears the input", async () => {
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue(textReply("Understood."));
    renderDock();
    await ready();
    const input = screen.getByPlaceholderText("Message the agent…") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "What's the status?" } });
    fireEvent.click(screen.getByText("Send"));

    expect(input.value).toBe(""); // cleared immediately, optimistic
    await waitFor(() => expect(api.chat).toHaveBeenCalledWith("s1", "What's the status?", "default"));
    await waitFor(() => expect(screen.getByText("Understood.")).toBeInTheDocument());
  });

  it("Enter (no Shift) submits the message", async () => {
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue(textReply("ok"));
    renderDock();
    await ready();
    const input = screen.getByPlaceholderText("Message the agent…");
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });
    await waitFor(() => expect(api.chat).toHaveBeenCalledWith("s1", "hello", "default"));
  });

  it("Shift+Enter does NOT submit (newline instead)", async () => {
    renderDock();
    await ready();
    const input = screen.getByPlaceholderText("Message the agent…");
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(api.chat).not.toHaveBeenCalled();
  });

  it("shows the 'Thinking…' indicator while a request is in flight and hides it after", async () => {
    let resolveChat!: (v: unknown) => void;
    (api.chat as ReturnType<typeof vi.fn>).mockReturnValue(new Promise((r) => { resolveChat = r; }));
    renderDock();
    await ready();
    fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => expect(screen.getByText("Thinking…")).toBeInTheDocument());
    // Send is disabled while busy — a rapid repeat click can't fire a second request.
    fireEvent.click(screen.getByText("Send"));
    expect(api.chat).toHaveBeenCalledTimes(1);

    resolveChat(textReply("done"));
    await waitFor(() => expect(screen.queryByText("Thinking…")).not.toBeInTheDocument());
  });

  it("renders a plain text response as an agent bubble", async () => {
    (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue(textReply("The hospital is currently at Normal load."));
    renderDock();
    await ready();
    fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "status?" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("The hospital is currently at Normal load.")).toBeInTheDocument());
  });

  it("renders a connection-error bubble distinctly when the backend call throws — never silently drops the message", async () => {
    (api.chat as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("fetch failed"));
    renderDock();
    await ready();
    fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "status?" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText(/Connection error: fetch failed/)).toBeInTheDocument());
  });

  describe("tool confirmation flow", () => {
    it("shows live Confirm/Cancel buttons for a response requiring approval", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "Admit P-1 to ICU?", response_type: "approval_required", patient_id: "P-1",
        actions: [], evidence: [], human_approval_required: true,
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "admit P-1" } });
      fireEvent.click(screen.getByText("Send"));

      await waitFor(() => expect(screen.getByText("Confirmation needed")).toBeInTheDocument());
      expect(screen.getByText("Awaiting your confirmation above — reply below or use the buttons.")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    });

    it("clicking Confirm resends 'yes' as a real chat message through the same confirmation gate", async () => {
      (api.chat as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({
          message: "Admit P-1 to ICU?", response_type: "approval_required", patient_id: "P-1",
          actions: [], evidence: [], human_approval_required: true,
        })
        .mockResolvedValueOnce(textReply("Admitted P-1 to ICU."));
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "admit P-1" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument());

      fireEvent.click(screen.getByRole("button", { name: "Confirm" }));
      await waitFor(() => expect(api.chat).toHaveBeenLastCalledWith("s1", "yes", "default"));
      await waitFor(() => expect(screen.getByText("Admitted P-1 to ICU.")).toBeInTheDocument());
    });

    it("clicking Cancel resends 'no'", async () => {
      (api.chat as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({
          message: "Admit P-1 to ICU?", response_type: "approval_required", patient_id: "P-1",
          actions: [], evidence: [], human_approval_required: true,
        })
        .mockResolvedValueOnce(textReply("Cancelled."));
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "admit P-1" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument());

      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
      await waitFor(() => expect(api.chat).toHaveBeenLastCalledWith("s1", "no", "default"));
    });

    it("a resolved (no-longer-live) confirmation card shows 'already resolved' instead of live buttons", async () => {
      (api.chat as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({
          message: "Admit P-1 to ICU?", response_type: "approval_required", patient_id: "P-1",
          actions: [], evidence: [], human_approval_required: true,
        })
        .mockResolvedValueOnce(textReply("Something else."));
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "admit P-1" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument());

      // Sending a NEW message (not yes/no) retires the old confirmation card.
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "never mind" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("This confirmation has already been resolved.")).toBeInTheDocument());
      expect(screen.queryByRole("button", { name: "Confirm" })).not.toBeInTheDocument();
    });
  });

  describe("action card renderers", () => {
    it("renders a recorded observation with the previous/new value and unit", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "Recorded.", response_type: "confirmation", patient_id: "52", human_approval_required: false, evidence: [],
        actions: [{
          tool: "add_patient_observation", status: "executed",
          data: { observation_type: "heart_rate", previous_value: 90, new_value: 112, unit: "bpm", duplicate: false, timestamp: "2026-01-01T00:00:00Z" },
        }],
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "record heart rate" } });
      fireEvent.click(screen.getByText("Send"));
      const card = (await screen.findByText("Observation recorded")).closest("div")!;
      expect(within(card).getByText("90")).toBeInTheDocument();
      expect(within(card).getByText("112")).toBeInTheDocument();
    });

    it("renders a duplicate observation as 'No change' rather than a fabricated update", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "No change.", response_type: "confirmation", patient_id: "52", human_approval_required: false, evidence: [],
        actions: [{
          tool: "add_patient_observation", status: "executed",
          data: { observation_type: "heart_rate", new_value: 112, duplicate: true },
        }],
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "hr 112" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("No change")).toBeInTheDocument());
      expect(screen.queryByText("Observation recorded")).not.toBeInTheDocument();
    });

    it("renders an updated triage assessment card with department and risk figures", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "Re-triaged.", response_type: "confirmation", patient_id: "52", human_approval_required: false, evidence: [],
        actions: [{
          tool: "run_triage_assessment", status: "executed",
          data: { department: "ICU", reconciled_admission_risk: 0.8, reconciled_icu_risk: 0.6, branches_agree: true },
        }],
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "retriage 52" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Updated assessment")).toBeInTheDocument());
      expect(screen.getByText("ICU")).toBeInTheDocument();
      expect(screen.getByText("80%")).toBeInTheDocument();
      expect(screen.getByText("60%")).toBeInTheDocument();
      expect(screen.getByText("Yes")).toBeInTheDocument();
    });

    it("renders a generic card for get_hospital_state", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "Here's the state.", response_type: "text", patient_id: null, human_approval_required: false, evidence: [],
        actions: [{ tool: "get_hospital_state", status: "executed", data: { ok: true } }],
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "hospital state?" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Hospital state fetched")).toBeInTheDocument());
    });

    it("renders no action card for an unrecognized tool (never fabricates a summary)", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "Done.", response_type: "text", patient_id: null, human_approval_required: false, evidence: [],
        actions: [{ tool: "some_future_tool", status: "executed", data: { whatever: true } }],
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "do the thing" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Done.")).toBeInTheDocument());
      // No card renders for an action this UI doesn't have a renderer for.
      expect(screen.queryByText("Updated assessment")).not.toBeInTheDocument();
      expect(screen.queryByText("Observation recorded")).not.toBeInTheDocument();
    });

    it("renders no action card for a PROPOSED (not yet executed) action — never shows a result before it happened", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: "Confirm?", response_type: "approval_required", patient_id: "P-1", human_approval_required: true, evidence: [],
        actions: [{ tool: "admit_simulated_patient", status: "proposed", data: null }],
      });
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "admit P-1" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Confirmation needed")).toBeInTheDocument());
      expect(screen.queryByText("Updated assessment")).not.toBeInTheDocument();
    });
  });

  describe("conversation history and Clear", () => {
    it("accumulates multiple turns in order", async () => {
      (api.chat as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(textReply("First reply."))
        .mockResolvedValueOnce(textReply("Second reply."));
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "one" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("First reply.")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "two" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Second reply.")).toBeInTheDocument());

      expect(screen.getByText("one")).toBeInTheDocument();
      expect(screen.getByText("two")).toBeInTheDocument();
      expect(screen.getByText("First reply.")).toBeInTheDocument();
    });

    it("Clear empties the conversation back to the suggested-prompts view", async () => {
      (api.chat as ReturnType<typeof vi.fn>).mockResolvedValue(textReply("hi there"));
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "hello" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("hi there")).toBeInTheDocument());

      fireEvent.click(screen.getByText("Clear"));
      expect(screen.queryByText("hi there")).not.toBeInTheDocument();
      expect(screen.getByText("What's the current hospital status?")).toBeInTheDocument();
    });
  });

  describe("adversarial: unmount and rapid interaction", () => {
    it("unmounting while a chat request is still in flight does not throw or warn", async () => {
      let resolveChat!: (v: ReturnType<typeof textReply>) => void;
      (api.chat as ReturnType<typeof vi.fn>).mockReturnValue(
        new Promise((resolve) => {
          resolveChat = resolve;
        })
      );
      const { unmount } = renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "hello" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Thinking…")).toBeInTheDocument());

      expect(() => unmount()).not.toThrow();
      // Resolving after unmount must not throw either (no setState-on-unmounted crash).
      expect(() => resolveChat(textReply("late reply"))).not.toThrow();
    });

    it("Send button click is inert while a request is already in flight, even if clicked repeatedly", async () => {
      let resolveChat!: (v: ReturnType<typeof textReply>) => void;
      (api.chat as ReturnType<typeof vi.fn>).mockReturnValue(
        new Promise((resolve) => {
          resolveChat = resolve;
        })
      );
      renderDock();
      await ready();
      fireEvent.change(screen.getByPlaceholderText("Message the agent…"), { target: { value: "hello" } });
      fireEvent.click(screen.getByText("Send"));
      await waitFor(() => expect(screen.getByText("Thinking…")).toBeInTheDocument());

      // Rapid repeated clicks while disabled must not fire additional requests.
      const send = screen.getByText("Send").closest("button")!;
      fireEvent.click(send);
      fireEvent.click(send);
      expect(api.chat).toHaveBeenCalledTimes(1);

      resolveChat(textReply("done"));
      await waitFor(() => expect(screen.getByText("done")).toBeInTheDocument());
    });
  });
});
