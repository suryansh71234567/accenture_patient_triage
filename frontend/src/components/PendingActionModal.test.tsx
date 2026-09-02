import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PendingActionModal } from "./PendingActionModal";

const mockSession = vi.hoisted(() => ({
  pendingDirectAction: null as { tool_name: string; kwargs: Record<string, unknown>; description: string } | null,
  resolvingDirectAction: false,
  resolveDirectAction: vi.fn(),
}));

vi.mock("../state/SessionContext", () => ({
  useSession: () => mockSession,
}));

function setPending(description = "Admit patient P-1 to ICU?") {
  mockSession.pendingDirectAction = { tool_name: "admit_simulated_patient", kwargs: { patient_id: "P-1" }, description };
}

beforeEach(() => {
  mockSession.pendingDirectAction = null;
  mockSession.resolvingDirectAction = false;
  mockSession.resolveDirectAction = vi.fn();
});

describe("PendingActionModal", () => {
  it("renders nothing when there is no pending action", () => {
    const { container } = render(<PendingActionModal />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the action description and dialog semantics when a direct action is pending", () => {
    setPending("Admit patient P-1 to ICU?");
    render(<PendingActionModal />);
    expect(screen.getByText("Confirmation required")).toBeInTheDocument();
    expect(screen.getByText("Admit patient P-1 to ICU?")).toBeInTheDocument();
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
  });

  it("Confirm calls resolveDirectAction(true)", () => {
    setPending();
    render(<PendingActionModal />);
    fireEvent.click(screen.getByText("Confirm"));
    expect(mockSession.resolveDirectAction).toHaveBeenCalledWith(true);
  });

  it("Cancel calls resolveDirectAction(false)", () => {
    setPending();
    render(<PendingActionModal />);
    fireEvent.click(screen.getByText("Cancel"));
    expect(mockSession.resolveDirectAction).toHaveBeenCalledWith(false);
  });

  it("shows the resolving/busy state with the long-running-assessment hint, and disables both buttons", () => {
    setPending();
    mockSession.resolvingDirectAction = true;
    render(<PendingActionModal />);
    expect(screen.getByText(/Applying and re-running the clinical assessment/)).toBeInTheDocument();
    expect(screen.getByText("Working…")).toBeInTheDocument();
    expect(screen.getByText("Working…").closest("button")).toBeDisabled();
    expect(screen.getByText("Cancel").closest("button")).toBeDisabled();
  });

  it("a disabled Confirm cannot be clicked again while resolving (no duplicate resolve call)", () => {
    setPending();
    mockSession.resolvingDirectAction = true;
    render(<PendingActionModal />);
    fireEvent.click(screen.getByText("Working…"));
    fireEvent.click(screen.getByText("Working…"));
    expect(mockSession.resolveDirectAction).not.toHaveBeenCalled();
  });

  it("Escape resolves as Cancel (false) when not resolving", () => {
    setPending();
    render(<PendingActionModal />);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(mockSession.resolveDirectAction).toHaveBeenCalledWith(false);
  });

  it("Escape is a no-op while resolving (mirrors Cancel being disabled)", () => {
    setPending();
    mockSession.resolvingDirectAction = true;
    render(<PendingActionModal />);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(mockSession.resolveDirectAction).not.toHaveBeenCalled();
  });

  it("focuses a focusable element on mount (focus trap wiring via useModalA11y)", () => {
    setPending();
    render(<PendingActionModal />);
    // useModalA11y focuses the first focusable element (Cancel) on mount.
    expect(document.activeElement).toBe(screen.getByText("Cancel"));
  });

  it("restores focus to the trigger element after the pending action clears (unmount)", () => {
    const trigger = document.createElement("button");
    trigger.textContent = "Admit";
    document.body.appendChild(trigger);
    trigger.focus();
    expect(document.activeElement).toBe(trigger);

    setPending();
    const { rerender } = render(<PendingActionModal />);
    expect(document.activeElement).not.toBe(trigger);

    mockSession.pendingDirectAction = null;
    rerender(<PendingActionModal />);
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });
});
