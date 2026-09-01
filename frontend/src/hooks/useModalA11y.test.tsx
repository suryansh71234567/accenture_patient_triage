import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useModalA11y } from "./useModalA11y";

function TestDialog({ onEscape }: { onEscape: () => void }) {
  const containerRef = useModalA11y(onEscape);
  return (
    <div ref={containerRef} tabIndex={-1} role="dialog" aria-modal="true">
      <button>First</button>
      <button>Last</button>
    </div>
  );
}

describe("useModalA11y", () => {
  it("focuses the first focusable element on mount", () => {
    render(<TestDialog onEscape={vi.fn()} />);
    expect(screen.getByText("First")).toHaveFocus();
  });

  it("calls onEscape when Escape is pressed", () => {
    const onEscape = vi.fn();
    render(<TestDialog onEscape={onEscape} />);
    fireEvent.keyDown(screen.getByText("First"), { key: "Escape" });
    expect(onEscape).toHaveBeenCalled();
  });

  it("wraps focus from the last element back to the first on Tab", () => {
    render(<TestDialog onEscape={vi.fn()} />);
    screen.getByText("Last").focus();
    fireEvent.keyDown(screen.getByText("Last"), { key: "Tab" });
    expect(screen.getByText("First")).toHaveFocus();
  });

  it("wraps focus from the first element to the last on Shift+Tab", () => {
    render(<TestDialog onEscape={vi.fn()} />);
    screen.getByText("First").focus();
    fireEvent.keyDown(screen.getByText("First"), { key: "Tab", shiftKey: true });
    expect(screen.getByText("Last")).toHaveFocus();
  });

  it("returns focus to the triggering element once the dialog unmounts", () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger);
    trigger.focus();

    const { unmount } = render(<TestDialog onEscape={vi.fn()} />);
    expect(screen.getByText("First")).toHaveFocus();
    unmount();
    expect(trigger).toHaveFocus();
    trigger.remove();
  });

  it("always calls the latest onEscape callback, not a stale one from an earlier render", () => {
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = render(<TestDialog onEscape={first} />);
    rerender(<TestDialog onEscape={second} />);
    fireEvent.keyDown(screen.getByText("First"), { key: "Escape" });
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalled();
  });
});
