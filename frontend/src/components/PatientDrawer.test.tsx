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
  it("shows the same not-yet-triaged treatment for a chart-only patient as an ARRIVED one (Fix 2)", () => {
    renderDrawer({ status: undefined, onTriage: vi.fn() });
    expect(screen.getByText("Not yet triaged. Clinical and operational assessment has not been performed.")).toBeInTheDocument();
    expect(screen.getByText("Triage Patient")).toBeInTheDocument();
  });

  it("hides the Triage Patient button for a chart-only patient when no real activation is possible (no onTriage passed)", () => {
    renderDrawer({ status: undefined });
    expect(screen.getByText(/Not yet triaged/)).toBeInTheDocument();
    expect(screen.queryByText("Triage Patient")).not.toBeInTheDocument();
  });

  it("shows the Triage button for an ARRIVED (waiting) patient and calls onTriage on click", () => {
    const onTriage = vi.fn();
    renderDrawer({ status: "ARRIVED", onTriage });
    const btn = screen.getByText("Triage Patient");
    expect(btn).toBeInTheDocument();
    fireEvent.click(btn);
    expect(onTriage).toHaveBeenCalled();
  });

  it("shows the AI recommendation for a TRIAGED patient", () => {
    renderDrawer({ status: "TRIAGED", decision });
    expect(screen.getByText("ICU-level need.")).toBeInTheDocument();
  });

  it("shows the admitted message for an IN_TREATMENT patient", () => {
    renderDrawer({ status: "IN_TREATMENT", decision });
    expect(screen.getByText(/No longer in active queue/)).toBeInTheDocument();
  });

  describe("DISCHARGED status", () => {
    it("shows the real discharge department from the decision, not a fabricated one", () => {
      renderDrawer({ status: "DISCHARGED", decision });
      expect(screen.getByText("Discharged from ICU. No longer an active patient.")).toBeInTheDocument();
    });

    it("falls back to a generic discharged message when no decision is available (never invents a department)", () => {
      renderDrawer({ status: "DISCHARGED" });
      expect(screen.getByText("Discharged. No longer an active patient.")).toBeInTheDocument();
    });

    it("offers no triage controls for a discharged patient", () => {
      const onTriage = vi.fn();
      renderDrawer({ status: "DISCHARGED", decision, onTriage });
      expect(screen.queryByText("Triage Patient")).not.toBeInTheDocument();
    });

    it("offers no Update Vitals or Admit action for a discharged patient — no fabricated active queue", () => {
      const onEditVitals = vi.fn();
      const onAdmit = vi.fn();
      renderDrawer({ status: "DISCHARGED", decision, onEditVitals, onAdmit });
      expect(screen.queryByText("Update Vitals")).not.toBeInTheDocument();
      expect(screen.queryByText(/^Admit to/)).not.toBeInTheDocument();
    });

    it("does not show the AI Recommendation / Current Queue cards — those only apply while TRIAGED", () => {
      renderDrawer({ status: "DISCHARGED", decision });
      expect(screen.queryByText("AI Recommendation")).not.toBeInTheDocument();
      expect(screen.queryByText("Current Queue")).not.toBeInTheDocument();
    });

    it("still shows the real chart vitals for a discharged patient", () => {
      renderDrawer({ status: "DISCHARGED", decision, vitals: [{ label: "HR", value: 88 }] });
      expect(screen.getByText("HR")).toBeInTheDocument();
      expect(screen.getByText("88")).toBeInTheDocument();
    });
  });

  it("switches to the Record tab and links to the full workspace", () => {
    renderDrawer({ status: "TRIAGED", decision });
    fireEvent.click(screen.getByText("Record"));
    expect(screen.getByText("Open full patient workspace →")).toBeInTheDocument();
  });
});
