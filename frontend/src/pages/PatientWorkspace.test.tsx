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

const mockSession = vi.hoisted(() => ({
  sessionId: "s1" as string | null,
  proposeAction: vi.fn(),
  mutationTick: 0,
  hospitalId: "default",
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => mockSession,
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
  mockSession.sessionId = "s1";
  mockSession.proposeAction = vi.fn();
  mockSession.mutationTick = 0;
  mockSession.hospitalId = "default";
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

  describe("observation flow", () => {
    it("renders the observation controls, with Record disabled until a value is entered", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());
      expect(screen.getByText("Record").closest("button")).toBeDisabled();
    });

    it("submitObservation proposes the correct real action for the real patient in context", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail("52"));
      mockSession.proposeAction = vi.fn().mockResolvedValue({ status: "failed", data: null, error: { code: "REJECTED", message: "cancelled" } });
      renderWorkspace("52");
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      // Default observation type is "Heart rate" (first OBS_TYPES entry).
      const valueInput = screen.getByPlaceholderText("bpm");
      fireEvent.change(valueInput, { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));

      await waitFor(() => expect(mockSession.proposeAction).toHaveBeenCalledWith("add_patient_observation", {
        patient_id: "52", observation_type: "heart_rate", value: 118,
      }));
    });

    it("submits the currently-selected observation type, not always the default", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      mockSession.proposeAction = vi.fn().mockResolvedValue({ status: "failed", data: null, error: { code: "REJECTED", message: "cancelled" } });
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      fireEvent.change(screen.getByDisplayValue("Heart rate"), { target: { value: "spo2" } });
      fireEvent.change(screen.getByPlaceholderText("%"), { target: { value: "91" } });
      fireEvent.click(screen.getByText("Record"));

      await waitFor(() => expect(mockSession.proposeAction).toHaveBeenCalledWith("add_patient_observation", {
        patient_id: "CHART-ONLY", observation_type: "spo2", value: 91,
      }));
    });

    it("on a real human-in-the-loop confirmation (status executed), clears the input and refreshes both the chart and the assessment", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      (api.assessPatient as ReturnType<typeof vi.fn>).mockResolvedValue({
        assessment: {
          department: "ICU", department_reasoning: "r", acuity_tier: 2,
          reconciled_admission_risk: 0.5, reconciled_icu_risk: 0.3, branches_agree: true,
          confidence_note: "c", top_diagnoses: [], red_flags: [],
        },
        resource_check: null,
      });
      mockSession.proposeAction = vi.fn().mockResolvedValue({ status: "executed", data: { message: "ok" }, error: null });
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      const valueInput = screen.getByPlaceholderText("bpm") as HTMLInputElement;
      fireEvent.change(valueInput, { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));

      await waitFor(() => expect(mockSession.proposeAction).toHaveBeenCalled());
      // Real refetch of the chart record (not fabricated) after a confirmed write.
      await waitFor(() => expect(api.getPatient).toHaveBeenCalledTimes(2));
      // Re-runs the SAME hospital-scoped assess call "Run assessment" uses —
      // never trusts the backend's own piggybacked reassessment payload
      // (that one has no hospital_id, see the code's own comment).
      await waitFor(() => expect(api.assessPatient).toHaveBeenCalledWith("CHART-ONLY", "s1", "default"));
      await waitFor(() => expect((screen.getByPlaceholderText("bpm") as HTMLInputElement).value).toBe(""));
    });

    it("on cancellation/rejection (status failed), does NOT clear the input and does NOT refetch or reassess", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      mockSession.proposeAction = vi.fn().mockResolvedValue({
        status: "failed", data: null, error: { code: "CONFIRM_FAILED", message: "Nurse cancelled." },
      });
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      const valueInput = screen.getByPlaceholderText("bpm") as HTMLInputElement;
      fireEvent.change(valueInput, { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));

      await waitFor(() => expect(mockSession.proposeAction).toHaveBeenCalled());
      expect(api.getPatient).toHaveBeenCalledTimes(1); // no refetch
      expect(api.assessPatient).not.toHaveBeenCalled(); // no reassessment
      expect(valueInput.value).toBe("118"); // not silently cleared, so the nurse can see/retry it
      // A deliberate cancellation is not an error — by design, no error banner for this path.
      expect(screen.queryByText("Nurse cancelled.")).not.toBeInTheDocument();
    });

    it("a retry clears the previous error banner", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      mockSession.proposeAction = vi.fn()
        .mockRejectedValueOnce(new Error("Network error"))
        .mockResolvedValueOnce({ status: "executed", data: { message: "ok" }, error: null });
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("bpm"), { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));
      await waitFor(() => expect(screen.getByText("Network error")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("bpm"), { target: { value: "120" } });
      fireEvent.click(screen.getByText("Record"));
      await waitFor(() => expect(screen.queryByText("Network error")).not.toBeInTheDocument());
    });

    it("stays busy (Record disabled) while a confirmation is genuinely pending, matching PendingActionModal's resolving state", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      let resolveProposal!: (v: unknown) => void;
      mockSession.proposeAction = vi.fn().mockReturnValue(new Promise((r) => { resolveProposal = r; }));
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("bpm"), { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));

      await waitFor(() => expect(screen.getByText("Record").closest("button")).toBeDisabled());
      resolveProposal({ status: "failed", data: null, error: { code: "REJECTED", message: "no" } });
      await waitFor(() => expect(screen.getByText("Record").closest("button")).not.toBeDisabled());
    });

    it("re-enables Record and shows a visible error after a network/backend failure (proposeAction throws), not just a silent unhandled rejection", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      mockSession.proposeAction = vi.fn().mockRejectedValue(new Error("Network error"));
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("bpm"), { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));

      await waitFor(() => expect(screen.getByText("Record").closest("button")).not.toBeDisabled());
      await waitFor(() => expect(screen.getByText("Network error")).toBeInTheDocument());
      expect(api.getPatient).toHaveBeenCalledTimes(1);
    });

    it("unmounting while an observation submission is in flight does not throw", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      let resolveProposal!: (v: unknown) => void;
      mockSession.proposeAction = vi.fn().mockReturnValue(new Promise((r) => { resolveProposal = r; }));
      const { unmount } = renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("bpm"), { target: { value: "118" } });
      fireEvent.click(screen.getByText("Record"));
      await waitFor(() => expect(screen.getByText("Record").closest("button")).toBeDisabled());

      expect(() => unmount()).not.toThrow();
      expect(() => resolveProposal({ status: "executed", data: { message: "ok" }, error: null })).not.toThrow();
    });

    it("rapid repeated clicks on Record while busy only propose the action once", async () => {
      (api.getPatient as ReturnType<typeof vi.fn>).mockResolvedValue(chartDetail());
      let resolveProposal!: (v: unknown) => void;
      mockSession.proposeAction = vi.fn().mockReturnValue(new Promise((r) => { resolveProposal = r; }));
      renderWorkspace();
      await waitFor(() => expect(screen.getByText("Record a new observation")).toBeInTheDocument());

      fireEvent.change(screen.getByPlaceholderText("bpm"), { target: { value: "118" } });
      const recordButton = screen.getByText("Record").closest("button")!;
      fireEvent.click(recordButton);
      await waitFor(() => expect(recordButton).toBeDisabled());
      fireEvent.click(recordButton);
      fireEvent.click(recordButton);

      expect(mockSession.proposeAction).toHaveBeenCalledTimes(1);
      resolveProposal({ status: "executed", data: { message: "ok" }, error: null });
    });
  });
});
