import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HospitalSelector } from "./HospitalSelector";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    listHospitals: vi.fn(),
    dashboard: vi.fn(),
  },
}));

const setHospitalId = vi.fn();
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ hospitalId: "default", setHospitalId, mutationTick: 0 }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  (api.listHospitals as ReturnType<typeof vi.fn>).mockResolvedValue([
    { hospital_id: "default", hospital_name: "Default Hospital", config_path: "" },
    { hospital_id: "west", hospital_name: "Westside Clinic", config_path: "" },
  ]);
  (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
    load: { operating_mode: "NORMAL", load_ratio: 0.5, lambda: 1 },
    untriaged_count: 1,
    triaged_count: 2,
    admitted_count: 3,
  });
});

describe("HospitalSelector", () => {
  it("shows the current hospital's name", async () => {
    render(<HospitalSelector />);
    await waitFor(() => expect(screen.getByText("Default Hospital")).toBeInTheDocument());
  });

  it("opens a dropdown listing every hospital and selects one", async () => {
    render(<HospitalSelector />);
    await waitFor(() => expect(screen.getByText("Default Hospital")).toBeInTheDocument());
    fireEvent.click(screen.getByTitle("Active hospital"));
    expect(screen.getByText("Westside Clinic")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Westside Clinic"));
    expect(setHospitalId).toHaveBeenCalledWith("west");
  });

  it("closes the dropdown on outside click", async () => {
    render(
      <div>
        <div data-testid="outside" />
        <HospitalSelector />
      </div>
    );
    await waitFor(() => expect(screen.getByText("Default Hospital")).toBeInTheDocument());
    fireEvent.click(screen.getByTitle("Active hospital"));
    expect(screen.getByText("Westside Clinic")).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByTestId("outside"));
    expect(screen.queryByText("Westside Clinic")).not.toBeInTheDocument();
  });
});
