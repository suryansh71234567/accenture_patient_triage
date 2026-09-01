import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TriageModal } from "./TriageModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    triageSimulated: vi.fn(),
    overrideDepartment: vi.fn(),
  },
}));

const bumpMutationTick = vi.fn();
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ bumpMutationTick }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

const triageResult = {
  patient_id: "P-1",
  clinical_assessment: { acuity_tier: 2, department_reasoning: "", top_diagnoses: [], red_flags: [] },
  operational_decision: {
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
    recommendation_summary: "ICU-level risk, beds available.",
  },
  patient: { patient_id: "P-1", age: 60, sex: "M", chief_complaint: "chest pain", vitals: {}, acuity: 2, status: "TRIAGED" },
};

describe("TriageModal", () => {
  it("closes on Escape (shared useModalA11y wiring)", () => {
    const onClose = vi.fn();
    render(
      <TriageModal
        patientId="P-1" age={60} sex="M" chiefComplaint="chest pain" vitals={{ hr: 100 }}
        hospitalId="h1" departmentOptions={["ICU", "ED_OBS"]} onClose={onClose} onDone={vi.fn()}
      />
    );
    fireEvent.keyDown(screen.getByText("Run AI Clinical Assessment"), { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("shows the Run Assessment button before any assessment has run", () => {
    render(
      <TriageModal
        patientId="P-1" age={60} sex="M" chiefComplaint="chest pain" vitals={{ hr: 100 }}
        hospitalId="h1" departmentOptions={["ICU", "ED_OBS"]} onClose={vi.fn()} onDone={vi.fn()}
      />
    );
    expect(screen.getByText("Run AI Clinical Assessment")).toBeInTheDocument();
  });

  it("runs the assessment and shows the AI recommendation", async () => {
    (api.triageSimulated as ReturnType<typeof vi.fn>).mockResolvedValue(triageResult);
    render(
      <TriageModal
        patientId="P-1" age={60} sex="M" chiefComplaint="chest pain" vitals={{ hr: 100 }}
        hospitalId="h1" departmentOptions={["ICU", "ED_OBS"]} onClose={vi.fn()} onDone={vi.fn()}
      />
    );
    fireEvent.click(screen.getByText("Run AI Clinical Assessment"));
    await waitFor(() => expect(screen.getByText("ICU-level risk, beds available.")).toBeInTheDocument());
    expect(api.triageSimulated).toHaveBeenCalledWith("P-1", "h1");
    expect(screen.getByText(/Confirm —/)).toBeInTheDocument();
  });

  it("Confirm calls onDone without a second API call (assessment already committed)", async () => {
    (api.triageSimulated as ReturnType<typeof vi.fn>).mockResolvedValue(triageResult);
    const onDone = vi.fn();
    render(
      <TriageModal
        patientId="P-1" age={60} sex="M" chiefComplaint="chest pain" vitals={{}}
        hospitalId="h1" departmentOptions={["ICU"]} onClose={vi.fn()} onDone={onDone}
      />
    );
    fireEvent.click(screen.getByText("Run AI Clinical Assessment"));
    await waitFor(() => screen.getByText(/Confirm —/));
    fireEvent.click(screen.getByText(/Confirm —/));
    expect(onDone).toHaveBeenCalled();
    expect(api.overrideDepartment).not.toHaveBeenCalled();
  });

  it("Override Placement reveals a department picker and calls overrideDepartment on confirm", async () => {
    (api.triageSimulated as ReturnType<typeof vi.fn>).mockResolvedValue(triageResult);
    (api.overrideDepartment as ReturnType<typeof vi.fn>).mockResolvedValue({});
    const onDone = vi.fn();
    render(
      <TriageModal
        patientId="P-1" age={60} sex="M" chiefComplaint="chest pain" vitals={{}}
        hospitalId="h1" departmentOptions={["ICU", "ED_OBS"]} onClose={vi.fn()} onDone={onDone}
      />
    );
    fireEvent.click(screen.getByText("Run AI Clinical Assessment"));
    await waitFor(() => screen.getByText("Override Placement"));
    fireEvent.click(screen.getByText("Override Placement"));
    fireEvent.click(screen.getByText(/Confirm Override/));
    await waitFor(() => expect(api.overrideDepartment).toHaveBeenCalledWith("P-1", "ICU", "", "h1"));
    expect(onDone).toHaveBeenCalled();
    // A direct override (not routed through proposeAction) must still bump
    // mutationTick, or other pages' pollers won't learn the override happened.
    expect(bumpMutationTick).toHaveBeenCalled();
  });
});
