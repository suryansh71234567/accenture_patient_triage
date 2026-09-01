import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DepartmentQueueBoard } from "./DepartmentQueueBoard";
import { api } from "../api/client";
import type { SimulationDashboard } from "../types";

vi.mock("../api/client", () => ({
  api: {
    reorderDepartmentQueue: vi.fn(),
    overrideDepartment: vi.fn(),
    updateSimulatedVitals: vi.fn(),
    triageSimulated: vi.fn(),
  },
}));

const proposeAction = vi.fn();
const bumpMutationTick = vi.fn();
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ sessionId: "s1", proposeAction, bumpMutationTick }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

function makeDash(): SimulationDashboard {
  return {
    time: "10:00",
    sim_time_minutes: 60,
    scenario: { name: "normal", title: "Normal", description: "", arrival_rate_per_hour: 5 },
    load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
    departments: [
      { name: "ICU", capacity: 10, occupied: 5, available: 5, occupancy_pct: 50, status: "OPEN" },
      { name: "ADMITTED_GEN", capacity: 20, occupied: 5, available: 15, occupancy_pct: 25, status: "OPEN" },
    ],
    waiting_queue: [],
    full_queue: [
      {
        patient_id: "P-1", age: 60, sex: "M", chief_complaint: "chest pain", acuity: 1, status: "TRIAGED",
        vitals: { hr: 100, spo2: 95 }, arrival_time_min: 0,
        operational_decision: {
          clinical_department: "ICU", operational_department: "ICU", ai_operational_department: "ICU",
          nurse_override: false, override_reason: null, available_beds_in_clinical_dept: 5,
          operating_mode: "NORMAL", lambda: 1, capacity_warning: false,
          confirmation_required: false, recommendation_summary: "ICU need.",
        },
      } as never,
    ],
    waiting_count: 0, triaged_count: 1, untriaged_count: 0, admitted_count: 0,
    recent_events: [],
  };
}

describe("DepartmentQueueBoard", () => {
  it("groups the patient card under its operational department column", () => {
    render(<DepartmentQueueBoard dash={makeDash()} hospitalId="h1" onChanged={vi.fn()} />);
    expect(screen.getByText("P-1")).toBeInTheDocument();
    // "ICU" appears as both the column header and inside the card's own text — assert at least the header.
    expect(screen.getAllByText("ICU").length).toBeGreaterThanOrEqual(1);
    // General Ward (ADMITTED_GEN, humanized) column exists too, and starts
    // empty. Scoped to the column-header <p> since P-1's card also lists it
    // as a "Move to" option now that the keyboard move control exists.
    expect(screen.getByText("General Ward", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByText("Drop patient here")).toBeInTheDocument();
  });

  it("admits directly (no modal) when confirmation is not required", async () => {
    proposeAction.mockResolvedValue({ status: "executed" });
    const onChanged = vi.fn();
    render(<DepartmentQueueBoard dash={makeDash()} hospitalId="h1" onChanged={onChanged} />);
    fireEvent.click(screen.getByText("Admit"));
    await waitFor(() => expect(proposeAction).toHaveBeenCalledWith("admit_simulated_patient", { patient_id: "P-1", department: "ICU", hospital_id: "h1" }));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
  });

  it("opens the AdmissionConfirmModal instead of admitting directly when confirmation_required is true", () => {
    const dash = makeDash();
    (dash.full_queue![0] as any).operational_decision.confirmation_required = true;
    render(<DepartmentQueueBoard dash={dash} hospitalId="h1" onChanged={vi.fn()} />);
    fireEvent.click(screen.getByText("Review"));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByRole("button", { name: "Confirm Admission" })).toBeInTheDocument();
    expect(within(dialog).getByText("P-1")).toBeInTheDocument();
    expect(proposeAction).not.toHaveBeenCalled();
  });

  it("cross-department drag+drop opens the override reason modal instead of a native prompt", async () => {
    (api.overrideDepartment as ReturnType<typeof vi.fn>).mockResolvedValue({});
    const onChanged = vi.fn();
    render(<DepartmentQueueBoard dash={makeDash()} hospitalId="h1" onChanged={onChanged} />);

    fireEvent.dragStart(screen.getByText("P-1").closest('[draggable="true"]')!);
    // "General Ward" is ADMITTED_GEN's humanized column header (scoped to
    // the <p> since it's also now a "Move to" option text elsewhere);
    // walk up to the column (drop target).
    fireEvent.drop(screen.getByText("General Ward", { selector: "p" }).closest("div")!.parentElement!.parentElement!);

    expect(screen.getByText("Move Patient — Nurse Override")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Confirm Move"));
    await waitFor(() => expect(api.overrideDepartment).toHaveBeenCalledWith("P-1", "ADMITTED_GEN", "", "h1"));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    // Direct api.* mutation (not routed through proposeAction) must still
    // bump mutationTick, or other pages' pollers won't learn it happened.
    expect(bumpMutationTick).toHaveBeenCalled();
  });

  it("offers a keyboard-usable 'Move to' select as an alternative to drag, reaching the same override confirmation", async () => {
    (api.overrideDepartment as ReturnType<typeof vi.fn>).mockResolvedValue({});
    const onChanged = vi.fn();
    render(<DepartmentQueueBoard dash={makeDash()} hospitalId="h1" onChanged={onChanged} />);

    fireEvent.change(screen.getByLabelText("Move P-1 to a different department"), {
      target: { value: "ADMITTED_GEN" },
    });

    expect(screen.getByText("Move Patient — Nurse Override")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Confirm Move"));
    await waitFor(() => expect(api.overrideDepartment).toHaveBeenCalledWith("P-1", "ADMITTED_GEN", "", "h1"));
  });

  it("renders every department column for a hospital with entirely custom department names, including empty ones as drop targets", () => {
    const dash = makeDash();
    dash.departments = [
      { name: "Trauma Bay", capacity: 8, occupied: 8, available: 0, occupancy_pct: 100, status: "CLOSED" },
      { name: "Neurology", capacity: 6, occupied: 2, available: 4, occupancy_pct: 33.3, status: "OPEN" },
    ];
    dash.full_queue = [];
    render(<DepartmentQueueBoard dash={dash} hospitalId="h1" onChanged={vi.fn()} />);

    expect(screen.getByText("Trauma Bay")).toBeInTheDocument();
    expect(screen.getByText("Neurology")).toBeInTheDocument();
    // Both columns are empty and still render a drop target — not silently
    // omitted the way a hardcoded default-department list would omit them.
    expect(screen.getAllByText("Drop patient here")).toHaveLength(2);
  });

  it("opens the retriage detail modal from the re-triage badge", () => {
    const dash = makeDash();
    (dash.full_queue![0] as any).operational_decision.retriage = true;
    (dash.full_queue![0] as any).operational_decision.previous_operational_department = "ADMITTED_GEN";
    render(<DepartmentQueueBoard dash={dash} hospitalId="h1" onChanged={vi.fn()} />);
    fireEvent.click(screen.getByText(/Re-triaged/));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(/Re-Triage/)).toBeInTheDocument();
    expect(within(dialog).getByText("P-1")).toBeInTheDocument();
  });
});
