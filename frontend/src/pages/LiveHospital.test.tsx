import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LiveHospital } from "./LiveHospital";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    dashboard: vi.fn(),
    scenarios: vi.fn(),
    listHospitals: vi.fn(),
    triggerArrival: vi.fn(),
    reorderQueue: vi.fn(),
    triageSimulated: vi.fn(),
    overrideDepartment: vi.fn(),
    step: vi.fn(),
  },
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ hospitalId: "default", proposeAction: vi.fn(), sessionId: "s1" }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  (api.scenarios as ReturnType<typeof vi.fn>).mockResolvedValue([]);
  (api.listHospitals as ReturnType<typeof vi.fn>).mockResolvedValue([]);
  (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
    time: "10:00",
    sim_time_minutes: 30,
    scenario: { name: "normal", title: "Normal Day", description: "baseline", arrival_rate_per_hour: 5 },
    load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
    departments: [{ name: "ICU", capacity: 10, occupied: 5, available: 5, occupancy_pct: 50, status: "OPEN" }],
    waiting_queue: [],
    full_queue: [
      { patient_id: "PAT-1", age: 40, sex: "F", chief_complaint: "fall", acuity: 3, status: "ARRIVED", vitals: { hr: 90 }, arrival_time_min: 0 },
      { patient_id: "PAT-2", age: 25, sex: "M", chief_complaint: "sprain", acuity: 4, status: "ARRIVED", vitals: { hr: 80 }, arrival_time_min: 5 },
    ],
    waiting_count: 2, triaged_count: 0, untriaged_count: 2, admitted_count: 0,
    recent_events: [],
  });
});

describe("LiveHospital", () => {
  it("renders the waiting-for-triage rail with a Triage button per card", async () => {
    render(<LiveHospital />);
    await waitFor(() => expect(screen.getByText("PAT-1")).toBeInTheDocument());
    expect(screen.getByText("Waiting for Triage · 2")).toBeInTheDocument();
    expect(screen.getAllByText("Triage")).toHaveLength(2);
  });

  it("opens the TriageModal from the rail card's Triage button", async () => {
    render(<LiveHospital />);
    await waitFor(() => expect(screen.getByText("PAT-1")).toBeInTheDocument());
    fireEvent.click(screen.getAllByText("Triage")[0]);
    expect(screen.getByText("Run AI Clinical Assessment")).toBeInTheDocument();
  });

  it("reorders the waiting rail via the ◀/▶ controls", async () => {
    (api.reorderQueue as ReturnType<typeof vi.fn>).mockResolvedValue({ moved: true, queue_length: 1 });
    render(<LiveHospital />);
    await waitFor(() => expect(screen.getByText("PAT-1")).toBeInTheDocument());
    fireEvent.click(screen.getAllByTitle("Move later in queue")[0]);
    await waitFor(() => expect(api.reorderQueue).toHaveBeenCalledWith("PAT-1", 1, "", "default"));
  });

  // Phase 5: time-stepping folded in from the retired standalone Simulation
  // screen — same api.step handler, now reachable from Live Hospital.
  it("Step +5 min and Step +15 min call the existing api.step handler", async () => {
    (api.step as ReturnType<typeof vi.fn>).mockResolvedValue({});
    render(<LiveHospital />);
    await waitFor(() => expect(screen.getByText("PAT-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Step +5 min"));
    await waitFor(() => expect(api.step).toHaveBeenCalledWith(5, true, "default"));
    fireEvent.click(screen.getByText("Step +15 min"));
    await waitFor(() => expect(api.step).toHaveBeenCalledWith(15, true, "default"));
  });

  it("shows the sim clock alongside load ratio and operating mode", async () => {
    render(<LiveHospital />);
    await waitFor(() => expect(screen.getByText("10:00")).toBeInTheDocument());
  });
});
