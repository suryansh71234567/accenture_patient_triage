import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { PatientWorkspace } from "./PatientWorkspace";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getPatient: vi.fn(),
    dashboard: vi.fn(),
    assessPatient: vi.fn(),
    manualArrival: vi.fn(),
    triageSimulated: vi.fn(),
  },
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ sessionId: "s1", proposeAction: vi.fn(), mutationTick: 0, hospitalId: "default" }),
}));

function renderWorkspace(id = "CHART-ONLY") {
  return render(
    <MemoryRouter initialEntries={[`/patients/${id}`]}>
      <Routes>
        <Route path="/patients/:id" element={<PatientWorkspace />} />
      </Routes>
    </MemoryRouter>
  );
}

const emptyDash = {
  time: "10:00", sim_time_minutes: 0,
  scenario: { name: "n", title: "n", description: "", arrival_rate_per_hour: 1 },
  load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
  departments: [{ name: "ICU", capacity: 10, occupied: 5, available: 5, occupancy_pct: 50, status: "OPEN" }],
  waiting_queue: [], full_queue: [],
  waiting_count: 0, triaged_count: 0, untriaged_count: 0, admitted_count: 0, recent_events: [],
};

function chartDetail(id = "CHART-ONLY") {
  return {
    summary: {
      patient_id: id, age: 40, sex: "F", chief_complaint: "headache", acuity: 4,
      time_elapsed_minutes: 15,
      vitals: { heart_rate: 80, resp_rate: 16, spo2: 98, sbp: 120, dbp: 80, temperature: 37, pain_score: 3 },
      last_updated: "2026-01-01T00:00:00Z",
    },
    observations: [],
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue(emptyDash);
});

describe("PatientWorkspace", () => {
  it("renders the patient header and Run assessment button", async () => {
    (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
    renderWorkspace();
    await waitFor(() => expect(screen.getByText("Patient CHART-ONLY")).toBeInTheDocument());
    expect(screen.getByText("Run assessment")).toBeInTheDocument();
  });

  it("offers Triage Patient for a chart-only patient with no live simulation record", async () => {
    (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
    renderWorkspace();
    await waitFor(() => expect(screen.getByText("Triage Patient")).toBeInTheDocument());
  });

  it("hides Triage Patient once the patient is already triaged in the live simulation", async () => {
    (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
    (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...emptyDash,
      full_queue: [{
        patient_id: "CHART-ONLY", age: 40, sex: "F", chief_complaint: "headache", acuity: 4, status: "TRIAGED",
        operational_decision: {
          clinical_department: "ADMITTED_GEN", operational_department: "ADMITTED_GEN", ai_operational_department: "ADMITTED_GEN",
          nurse_override: false, override_reason: null, available_beds_in_clinical_dept: 5,
          operating_mode: "NORMAL", lambda: 1, capacity_warning: false, confirmation_required: false,
          recommendation_summary: "General ward.",
        },
      }],
    });
    renderWorkspace();
    await waitFor(() => expect(screen.getByText("Patient CHART-ONLY")).toBeInTheDocument());
    expect(screen.queryByText("Triage Patient")).not.toBeInTheDocument();
  });

  it("clicking Triage Patient activates (manualArrival) then triages the chart-only patient in one flow", async () => {
    (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
    (api.manualArrival as ReturnType<typeof vi.fn>).mockResolvedValue({ patient_id: "CHART-ONLY", status: "ARRIVED" });
    (api.triageSimulated as ReturnType<typeof vi.fn>).mockResolvedValue({
      patient_id: "CHART-ONLY",
      clinical_assessment: { acuity_tier: 4, department_reasoning: "", top_diagnoses: [], red_flags: [] },
      operational_decision: {
        clinical_department: "ADMITTED_GEN", operational_department: "ADMITTED_GEN", ai_operational_department: "ADMITTED_GEN",
        nurse_override: false, override_reason: null, available_beds_in_clinical_dept: 5,
        operating_mode: "NORMAL", lambda: 1, capacity_warning: false, confirmation_required: false,
        recommendation_summary: "General ward appropriate.",
      },
      patient: { patient_id: "CHART-ONLY", age: 40, sex: "F", chief_complaint: "headache", vitals: {}, acuity: 4, status: "TRIAGED" },
    });

    renderWorkspace();
    await waitFor(() => expect(screen.getByText("Triage Patient")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Triage Patient"));

    await waitFor(() => expect(api.manualArrival).toHaveBeenCalledWith({
      patient_id: "CHART-ONLY", chief_complaint: "headache", age: 40, sex: "F", acuity: 4, hospital_id: "default",
    }));
    await waitFor(() => expect(api.triageSimulated).toHaveBeenCalledWith("CHART-ONLY", "default"));
    await waitFor(() => expect(screen.getByText("General ward appropriate.")).toBeInTheDocument());
    // "Run assessment" (the separate, read-only preview action) is untouched by this flow.
    expect(api.assessPatient).not.toHaveBeenCalled();
  });

  it("Run assessment still calls the real, unchanged read-only assess endpoint", async () => {
    (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
    (api.assessPatient as ReturnType<typeof vi.fn>).mockResolvedValue({
      assessment: {
        department: "ICU", department_reasoning: "reasoning", acuity_tier: 2,
        reconciled_admission_risk: 0.5, reconciled_icu_risk: 0.3, branches_agree: true,
        confidence_note: "confident", top_diagnoses: [], red_flags: [],
      },
      resource_check: null,
    });
    renderWorkspace();
    await waitFor(() => expect(screen.getByText("Run assessment")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run assessment"));
    await waitFor(() => expect(api.assessPatient).toHaveBeenCalledWith("CHART-ONLY", "s1", "default"));
    expect(api.manualArrival).not.toHaveBeenCalled();
    expect(api.triageSimulated).not.toHaveBeenCalled();
  });
});
