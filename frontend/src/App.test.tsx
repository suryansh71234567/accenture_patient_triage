import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api/client";

vi.mock("./api/client", () => ({
  api: {
    createSession: vi.fn().mockResolvedValue({ session_id: "s1", role: "nurse" }),
    getSessionState: vi.fn().mockResolvedValue({}),
    listHospitals: vi.fn().mockResolvedValue([]),
    dashboard: vi.fn().mockResolvedValue({
      time: "10:00", sim_time_minutes: 0,
      scenario: { name: "n", title: "n", description: "", arrival_rate_per_hour: 1 },
      load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
      departments: [], waiting_queue: [], full_queue: [],
      waiting_count: 0, triaged_count: 0, untriaged_count: 0, admitted_count: 0, recent_events: [],
    }),
    listPatients: vi.fn().mockResolvedValue([]),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  window.history.pushState({}, "", "/");
});

describe("App routing", () => {
  it("renders the Dashboard at / inside the nav shell, with no /hospitals route dead link", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Hospital Capacity")).toBeInTheDocument());
    expect(screen.getByText("TriageGuard")).toBeInTheDocument();
    expect(screen.getByText("Hospital Network")).toBeInTheDocument();
    expect(api.dashboard).toHaveBeenCalled();
  });
});
