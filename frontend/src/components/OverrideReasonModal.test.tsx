import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OverrideReasonModal } from "./OverrideReasonModal";

describe("OverrideReasonModal", () => {
  it("shows the from/to departments, humanized via DEPT_LABELS", () => {
    render(
      <OverrideReasonModal patientId="P-1" from="ICU" to="ED_OBS" busy={false} onConfirm={vi.fn()} onCancel={vi.fn()} />
    );
    expect(screen.getByText("ICU")).toBeInTheDocument();
    expect(screen.getByText("ED Observation")).toBeInTheDocument();
  });

  it("fills the textarea when a quick-reason chip is clicked", () => {
    render(
      <OverrideReasonModal patientId="P-1" from="ICU" to="ED_OBS" busy={false} onConfirm={vi.fn()} onCancel={vi.fn()} />
    );
    fireEvent.click(screen.getByText("Capacity constraint"));
    expect(screen.getByPlaceholderText("Reason for override…")).toHaveValue("Capacity constraint");
  });

  it("calls onConfirm with the typed reason", () => {
    const onConfirm = vi.fn();
    render(
      <OverrideReasonModal patientId="P-1" from="ICU" to="ED_OBS" busy={false} onConfirm={onConfirm} onCancel={vi.fn()} />
    );
    fireEvent.change(screen.getByPlaceholderText("Reason for override…"), { target: { value: "Family request" } });
    fireEvent.click(screen.getByText("Confirm Move"));
    expect(onConfirm).toHaveBeenCalledWith("Family request");
  });

  it("calls onCancel when Cancel is clicked", () => {
    const onCancel = vi.fn();
    render(
      <OverrideReasonModal patientId="P-1" from="ICU" to="ED_OBS" busy={false} onConfirm={vi.fn()} onCancel={onCancel} />
    );
    fireEvent.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalled();
  });

  it("disables the confirm button while busy", () => {
    render(
      <OverrideReasonModal patientId="P-1" from="ICU" to="ED_OBS" busy onConfirm={vi.fn()} onCancel={vi.fn()} />
    );
    expect(screen.getByText("Applying…")).toBeDisabled();
  });
});
