import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { VitalsEditModal } from "./VitalsEditModal";

describe("VitalsEditModal", () => {
  it("only includes filled-in fields in the saved vitals object", () => {
    const onSave = vi.fn();
    render(<VitalsEditModal patientId="P-1" busy={false} onSave={onSave} onClose={vi.fn()} />);

    const hrInput = screen.getAllByRole("spinbutton")[0];
    fireEvent.change(hrInput, { target: { value: "110" } });
    fireEvent.click(screen.getByText("Save & Re-triage"));

    expect(onSave).toHaveBeenCalledWith({ hr: 110 });
  });

  it("saves an empty object when nothing is filled in", () => {
    const onSave = vi.fn();
    render(<VitalsEditModal patientId="P-1" busy={false} onSave={onSave} onClose={vi.fn()} />);
    fireEvent.click(screen.getByText("Save & Re-triage"));
    expect(onSave).toHaveBeenCalledWith({});
  });

  it("calls onClose from Cancel", () => {
    const onClose = vi.fn();
    render(<VitalsEditModal patientId="P-1" busy={false} onSave={vi.fn()} onClose={onClose} />);
    fireEvent.click(screen.getByText("Cancel"));
    expect(onClose).toHaveBeenCalled();
  });
});
