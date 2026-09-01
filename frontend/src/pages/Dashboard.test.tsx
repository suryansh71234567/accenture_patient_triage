import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "./Dashboard";
import { api } from "../api/client";

function renderDashboard() {
  return render(
    <MemoryRouter>
      <Dashboard />
    </MemoryRouter>
  );
}

vi.mock("../api/client", () => ({
  api: {
    dashboard: vi.fn(),
  },
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ mutationTick: 0, hospitalId: "default", proposeAction: vi.fn() }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
    time: "10:00",
    sim_time_minutes: 60,
    scenario: { name: "normal", title: "Normal Day", description: "", arrival_rate_per_hour: 5 },
    load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
    departments: [{ name: "ICU", capacity: 10, occupied: 5, available: 5, occupancy_pct: 50, status: "OPEN" }],
    waiting_queue: [],
    full_queue: [
      {
        patient_id: "P-1", age: 60, sex: "M", chief_complaint: "chest pain", acuity: 2, status: "TRIAGED",
        vitals: { hr: 100 }, arrival_time_min: 30,
        operational_decision: {
          clinical_department: "ICU", operational_department: "ICU", ai_operational_department: "ICU",
          nurse_override: false, override_reason: null, available_beds_in_clinical_dept: 5,
          operating_mode: "NORMAL", lambda: 1, capacity_warning: false,
          confirmation_required: false, recommendation_summary: "ICU-level need.",
        },
      },
    ],
    waiting_count: 0, triaged_count: 1, untriaged_count: 2, admitted_count: 1,
    recent_events: ["Patient arrived"],
  });
});

describe("Dashboard", () => {
  it("renders pipeline stats, department capacity, and the department queue preview", async () => {
    renderDashboard();
    await waitFor(() => expect(screen.getByText("Waiting for Triage")).toBeInTheDocument());
    expect(screen.getByText("2")).toBeInTheDocument(); // untriaged_count
    expect(screen.getByText("Hospital Capacity")).toBeInTheDocument();
    expect(screen.getByText("Live Queues by Department")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("P-1")).toBeInTheDocument());
  });

  it("opens the patient drawer on a queue-preview patient click", async () => {
    renderDashboard();
    await waitFor(() => expect(screen.getByText("P-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("P-1"));
    await waitFor(() => expect(screen.getByText("ICU-level need.")).toBeInTheDocument());
  });
});
