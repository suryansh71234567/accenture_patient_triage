import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HospitalOnboardingWizard } from "./HospitalOnboardingWizard";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    registerHospital: vi.fn(),
    calibrationScenarios: vi.fn(),
    submitCalibration: vi.fn(),
  },
}));

const setHospitalId = vi.fn();
const bumpMutationTick = vi.fn();
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ setHospitalId, bumpMutationTick }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("HospitalOnboardingWizard", () => {
  it("requires an ID and name before registering (validated on submit, not on Next)", () => {
    render(<HospitalOnboardingWizard onClose={vi.fn()} />);
    fireEvent.click(screen.getByText("Next: Departments & Capacity"));
    fireEvent.click(screen.getByText("Register Hospital"));
    expect(screen.getByText("Hospital ID and name are required.")).toBeInTheDocument();
    expect(api.registerHospital).not.toHaveBeenCalled();
  });

  it(
    "walks Details -> Departments -> registers -> Policy Framing bound to the registered id, not a reactive session id",
    async () => {
      (api.registerHospital as ReturnType<typeof vi.fn>).mockResolvedValue({ hospital_id: "west", hospital_name: "Westside Clinic" });
      (api.calibrationScenarios as ReturnType<typeof vi.fn>).mockResolvedValue({
        hospital_id: "west",
        scenario_count: 1,
        scenarios: [{ scenario_id: "S01", description: "desc", candidate_departments: ["ICU"], preferred_department: "ICU", reason: "r" }],
      });

      render(<HospitalOnboardingWizard onClose={vi.fn()} />);
      fireEvent.change(screen.getByPlaceholderText("e.g. westside_clinic"), { target: { value: "west" } });
      fireEvent.change(screen.getByPlaceholderText("e.g. Westside Clinic"), { target: { value: "Westside Clinic" } });
      fireEvent.click(screen.getByText("Next: Departments & Capacity"));

      fireEvent.click(screen.getByText("Register Hospital"));

      await waitFor(() => expect(api.calibrationScenarios).toHaveBeenCalledWith("west"));
      expect(setHospitalId).toHaveBeenCalledWith("west");
      await waitFor(() => expect(screen.getByText(/registered and selected/)).toBeInTheDocument());
    }
  );

  it("existingHospital prop jumps straight to Policy Framing (Configure Policy shortcut)", async () => {
    (api.calibrationScenarios as ReturnType<typeof vi.fn>).mockResolvedValue({ hospital_id: "west", scenario_count: 0, scenarios: [] });
    render(<HospitalOnboardingWizard onClose={vi.fn()} existingHospital={{ hospital_id: "west", hospital_name: "Westside Clinic" }} />);
    await waitFor(() => expect(api.calibrationScenarios).toHaveBeenCalledWith("west"));
    expect(screen.getByText(/Available policy scenarios for "Westside Clinic"/)).toBeInTheDocument();
  });
});
