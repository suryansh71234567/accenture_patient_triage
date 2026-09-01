import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SimulationControlCenter } from "./SimulationControlCenter";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { dashboard: vi.fn(), scenarios: vi.fn(), step: vi.fn(), triggerArrival: vi.fn(), loadScenario: vi.fn() },
}));
vi.mock("../state/SessionContext", () => ({ useSession: () => ({ hospitalId: "default" }) }));

beforeEach(() => {
  vi.clearAllMocks();
  (api.scenarios as ReturnType<typeof vi.fn>).mockResolvedValue([
    { name: "normal_day", title: "Normal Day", description: "baseline", arrival_rate_per_hour: 5 },
  ]);
  (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
    time: "10:00", sim_time_minutes: 30,
    scenario: { name: "normal_day", title: "Normal Day", description: "baseline", arrival_rate_per_hour: 5 },
    load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 0.9 },
    departments: [], waiting_queue: [], full_queue: [],
    waiting_count: 0, triaged_count: 3, untriaged_count: 2, admitted_count: 1,
    recent_events: ["Hospital scenario switched to Normal Day"],
  });
});

describe("SimulationControlCenter", () => {
  it("shows the sim clock, counts, and events", async () => {
    render(<SimulationControlCenter />);
    await waitFor(() => expect(screen.getByText("10:00")).toBeInTheDocument());
    expect(screen.getByText("2")).toBeInTheDocument(); // untriaged
    expect(screen.getByText(/scenario switched to Normal Day/)).toBeInTheDocument();
  });

  it("Step +5 min calls api.step(5, ...)", async () => {
    (api.step as ReturnType<typeof vi.fn>).mockResolvedValue({});
    render(<SimulationControlCenter />);
    await waitFor(() => expect(screen.getByText("Step +5 min")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Step +5 min"));
    await waitFor(() => expect(api.step).toHaveBeenCalledWith(5, true, "default"));
  });
});
