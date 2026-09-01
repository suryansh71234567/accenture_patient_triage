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
});
