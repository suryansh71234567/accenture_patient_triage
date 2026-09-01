import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { PatientDrawer } from "./PatientDrawer";
import type { OperationalDecision } from "../types";

function renderDrawer(props: Partial<React.ComponentProps<typeof PatientDrawer>>) {
  return render(
    <MemoryRouter>
      <PatientDrawer
        patientId="P-1"
        age={60}
        sex="M"
        chiefComplaint="chest pain"
        vitals={[{ label: "HR", value: 100 }]}
        acuity={2}
        onClose={vi.fn()}
        {...props}
      />
    </MemoryRouter>
  );
}

const decision: OperationalDecision = {
  clinical_department: "ICU",
  operational_department: "ICU",
  ai_operational_department: "ICU",
  nurse_override: false,
  override_reason: null,
  available_beds_in_clinical_dept: 2,
  operating_mode: "NORMAL",
  lambda: 1,
  capacity_warning: false,
  confirmation_required: false,
  recommendation_summary: "ICU-level need.",
};

describe("PatientDrawer", () => {
  it("shows a chart-only message when there is no simulation record", () => {
    renderDrawer({ status: undefined });
    expect(screen.getByText(/No simulation record for this patient/)).toBeInTheDocument();
  });

  it("shows the Triage button for an ARRIVED (waiting) patient", () => {
    const onTriage = vi.fn();
    renderDrawer({ status: "ARRIVED", onTriage });
    expect(screen.getByText("Triage Patient")).toBeInTheDocument();
  });

  it("shows the AI recommendation for a TRIAGED patient", () => {
    renderDrawer({ status: "TRIAGED", decision });
    expect(screen.getByText("ICU-level need.")).toBeInTheDocument();
  });

  it("shows the admitted message for an IN_TREATMENT patient", () => {
    renderDrawer({ status: "IN_TREATMENT", decision });
    expect(screen.getByText(/No longer in active queue/)).toBeInTheDocument();
  });

  it("switches to the Record tab and links to the full workspace", () => {
    renderDrawer({ status: "TRIAGED", decision });
    fireEvent.click(screen.getByText("Record"));
    expect(screen.getByText("Open full patient workspace →")).toBeInTheDocument();
  });
});
