import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PatientList } from "./PatientList";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    listPatients: vi.fn(),
    dashboard: vi.fn(),
    listHospitals: vi.fn(),
    updateSimulatedVitals: vi.fn(),
    triageSimulated: vi.fn(),
    manualArrival: vi.fn(),
  },
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ hospitalId: "default" }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  (api.listHospitals as ReturnType<typeof vi.fn>).mockResolvedValue([]);
  (api.listPatients as ReturnType<typeof vi.fn>).mockResolvedValue([
    { patient_id: "52", age: 62, sex: "M", chief_complaint: "chest pain", acuity: 2, time_elapsed_minutes: 15, vitals: {}, last_updated: "" },
    { patient_id: "CHART-ONLY", age: 40, sex: "F", chief_complaint: "headache", acuity: 4, time_elapsed_minutes: 15, vitals: {}, last_updated: "" },
  ]);
  (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
    time: "10:00", sim_time_minutes: 40,
    scenario: { name: "n", title: "n", description: "", arrival_rate_per_hour: 1 },
    load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
    departments: [{ name: "ICU", capacity: 10, occupied: 5, available: 5, occupancy_pct: 50, status: "OPEN" }],
    waiting_queue: [],
    full_queue: [
      {
        patient_id: "52", age: 62, sex: "M", chief_complaint: "chest pain", acuity: 2, status: "TRIAGED",
        arrival_time_min: 10, vitals: { hr: 100 },
        operational_decision: {
          clinical_department: "ICU", operational_department: "ICU", ai_operational_department: "ICU",
          nurse_override: false, override_reason: null, available_beds_in_clinical_dept: 5,
          operating_mode: "NORMAL", lambda: 1, capacity_warning: false, confirmation_required: false,
          recommendation_summary: "ICU need.",
        },
      },
      // No chart record for this one — created purely by Random Arrival/Register Patient/a scenario.
      {
        patient_id: "PAT-101", age: 34, sex: "F", chief_complaint: "fall", acuity: 3, status: "ARRIVED",
        arrival_time_min: 5, vitals: { hr: 88 },
      },
    ],
    waiting_count: 0, triaged_count: 1, untriaged_count: 0, admitted_count: 0,
    recent_events: [],
  });
});

describe("PatientList table", () => {
  it("shows real AI Dept / Current Dept / Waiting for a simulation-linked patient", async () => {
    render(<PatientList />);
    await waitFor(() => expect(screen.getByText("52")).toBeInTheDocument());
    const row = screen.getByText("52").closest("button")!;
    expect(row.textContent).toContain("ICU");
    expect(row.textContent).toContain("30m"); // 40 - 10
  });

  it('shows "—" placeholders for a chart-only patient with no simulation record', async () => {
    render(<PatientList />);
    await waitFor(() => expect(screen.getByText("CHART-ONLY")).toBeInTheDocument());
    const row = screen.getByText("CHART-ONLY").closest("button")!;
    // Status, AI Dept, Current Dept, Waiting all render as "—"
    expect(row.textContent!.match(/—/g)?.length).toBeGreaterThanOrEqual(4);
  });

  it("filters by search text", async () => {
    render(<PatientList />);
    await waitFor(() => expect(screen.getByText("52")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Search by ID or complaint…"), { target: { value: "headache" } });
    expect(screen.queryByText("52")).not.toBeInTheDocument();
    expect(screen.getByText("CHART-ONLY")).toBeInTheDocument();
  });

  it("opens the patient drawer on row click", async () => {
    render(<PatientList />);
    await waitFor(() => expect(screen.getByText("52")).toBeInTheDocument());
    fireEvent.click(screen.getByText("52"));
    expect(screen.getByText("ICU need.")).toBeInTheDocument();
  });

  it("shows a simulation-only patient with no chart record (e.g. created via Random Arrival)", async () => {
    render(<PatientList />);
    await waitFor(() => expect(screen.getByText("PAT-101")).toBeInTheDocument());
    const row = screen.getByText("PAT-101").closest("button")!;
    expect(row.textContent).toContain("34F");
    expect(row.textContent).toContain("fall");
    expect(row.textContent).toContain("arrived");
  });

  it("opens the drawer for a chart-less simulation-only patient using its live record", async () => {
    render(<PatientList />);
    await waitFor(() => expect(screen.getByText("PAT-101")).toBeInTheDocument());
    fireEvent.click(screen.getByText("PAT-101"));
    expect(screen.getByText("Triage Patient")).toBeInTheDocument();
    // Header combines age/sex/complaint from the live sim record (no chart record exists for PAT-101).
    expect(screen.getByText(/34 F · fall/)).toBeInTheDocument();
    // Vitals come from the sim record too (hr: 88), not fabricated/blank.
    expect(screen.getByText("88")).toBeInTheDocument();
  });

  describe("Triage Patient for a chart-only patient (Fix 2)", () => {
    it("offers Triage Patient in the drawer, and clicking it activates + triages the patient in one flow", async () => {
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

      render(<PatientList />);
      await waitFor(() => expect(screen.getByText("CHART-ONLY")).toBeInTheDocument());
      fireEvent.click(screen.getByText("CHART-ONLY"));

      const triageBtn = screen.getByText("Triage Patient");
      fireEvent.click(triageBtn);

      // manualArrival brings the chart-only patient into the live queue using its
      // real chart data — no vitals re-entry, no fabricated fields.
      await waitFor(() => expect(api.manualArrival).toHaveBeenCalledWith({
        patient_id: "CHART-ONLY", chief_complaint: "headache", age: 40, sex: "F", acuity: 4, hospital_id: "default",
      }));
      // Then triaged exactly like any other queue patient — same endpoint, same result UI.
      await waitFor(() => expect(api.triageSimulated).toHaveBeenCalledWith("CHART-ONLY", "default"));
      await waitFor(() => expect(screen.getByText("General ward appropriate.")).toBeInTheDocument());
    });
  });
});
