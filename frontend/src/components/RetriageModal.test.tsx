import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RetriageModal } from "./RetriageModal";
import type { OperationalDecision } from "../types";

const baseDecision: OperationalDecision = {
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
  recommendation_summary: "Deterioration since last check — escalated to ICU.",
  retriage: true,
  previous_operational_department: "ADMITTED_GEN",
};

describe("RetriageModal", () => {
  it("shows the new recommendation and previous department, but no fabricated previous vitals", () => {
    render(
      <RetriageModal patientId="P-1" vitals={{ hr: 120, spo2: 90 }} acuity={1} decision={baseDecision} onAcknowledge={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.getByText(/Deterioration since last check/)).toBeInTheDocument();
    expect(screen.getByText(/Previously: General Ward/)).toBeInTheDocument();
    // No "previous vitals" heading — the real API doesn't carry that data.
    expect(screen.queryByText(/previous vitals/i)).not.toBeInTheDocument();
  });

  it("shows the prior-override warning banner when previous_nurse_override is true", () => {
    render(
      <RetriageModal
        patientId="P-1"
        vitals={{}}
        acuity={1}
        decision={{ ...baseDecision, previous_nurse_override: true, previous_override_reason: "family request" }}
        onAcknowledge={vi.fn()}
        onClose={vi.fn()}
      />
    );
    expect(screen.getByText(/RE-TRIAGE REQUIRES REVIEW/)).toBeInTheDocument();
    expect(screen.getByText(/family request/)).toBeInTheDocument();
  });

  it("calls onAcknowledge from the Acknowledge button", () => {
    const onAcknowledge = vi.fn();
    render(
      <RetriageModal patientId="P-1" vitals={{}} acuity={1} decision={baseDecision} onAcknowledge={onAcknowledge} onClose={vi.fn()} />
    );
    fireEvent.click(screen.getByText(/Acknowledge & Requeue/));
    expect(onAcknowledge).toHaveBeenCalled();
  });
});
