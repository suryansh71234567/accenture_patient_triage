import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HospitalNetwork } from "./HospitalNetwork";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    listHospitals: vi.fn(),
    dashboard: vi.fn(),
    calibrationStatus: vi.fn(),
  },
}));
const setHospitalId = vi.fn();
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ hospitalId: "a", setHospitalId }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  (api.listHospitals as ReturnType<typeof vi.fn>).mockResolvedValue([
    { hospital_id: "a", hospital_name: "Hospital A", config_path: "" },
    { hospital_id: "b", hospital_name: "Hospital B", config_path: "" },
  ]);
  (api.dashboard as ReturnType<typeof vi.fn>).mockImplementation((id: string) =>
    Promise.resolve({
      time: "10:00", sim_time_minutes: 0,
      scenario: { name: "n", title: "n", description: "", arrival_rate_per_hour: 1 },
      load: { load_ratio: 0.5, operating_mode: id === "b" ? "HIGH_LOAD" : "NORMAL", lambda: 1 },
      departments: [{ name: "ICU", capacity: 10, occupied: 5, available: 5, occupancy_pct: 50, status: "OPEN" }],
      waiting_queue: [], full_queue: [],
      waiting_count: 0, triaged_count: 0, untriaged_count: 0, admitted_count: 0,
      recent_events: [],
    })
  );
  (api.calibrationStatus as ReturnType<typeof vi.fn>).mockResolvedValue({ hospital_id: "x", calibrated: true });
});

describe("HospitalNetwork", () => {
  it("calls dashboard() once per registered hospital (N sequential calls)", async () => {
    render(<HospitalNetwork />);
    await waitFor(() => expect(screen.getByText("Hospital A")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("Hospital B")).toBeInTheDocument());
    expect(api.dashboard).toHaveBeenCalledTimes(2);
    expect(api.dashboard).toHaveBeenCalledWith("a");
    expect(api.dashboard).toHaveBeenCalledWith("b");
  });

  it("flags elevated load for a HIGH_LOAD hospital", async () => {
    render(<HospitalNetwork />);
    await waitFor(() => expect(screen.getByText("Hospital B")).toBeInTheDocument());
    expect(screen.getByText(/Elevated load/)).toBeInTheDocument();
  });

  it("selecting a hospital calls setHospitalId", async () => {
    render(<HospitalNetwork />);
    await waitFor(() => expect(screen.getByText("Hospital B")).toBeInTheDocument());
    fireEvent.click(screen.getAllByText("Select")[0]);
    expect(setHospitalId).toHaveBeenCalledWith("b");
  });

  it("opens the onboarding wizard from + Add Hospital", async () => {
    render(<HospitalNetwork />);
    await waitFor(() => expect(screen.getByText("Hospital A")).toBeInTheDocument());
    fireEvent.click(screen.getByText("+ Add Hospital"));
    expect(screen.getByText("Hospital Onboarding")).toBeInTheDocument();
  });
});
