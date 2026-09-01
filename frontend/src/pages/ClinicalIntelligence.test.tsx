import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ClinicalIntelligence } from "./ClinicalIntelligence";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { dashboard: vi.fn() } }));
vi.mock("../state/SessionContext", () => ({ useSession: () => ({ hospitalId: "default" }) }));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ClinicalIntelligence", () => {
  it("shows the empty state when no TRIAGED patient has a clinical_assessment", async () => {
    (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
      time: "10:00", sim_time_minutes: 0,
      scenario: { name: "n", title: "n", description: "", arrival_rate_per_hour: 1 },
      load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
      departments: [], waiting_queue: [], full_queue: [],
      waiting_count: 0, triaged_count: 0, untriaged_count: 0, admitted_count: 0, recent_events: [],
    });
    render(<ClinicalIntelligence />);
    await waitFor(() => expect(screen.getByText(/No patients with detailed clinical intelligence/)).toBeInTheDocument());
  });

  it("shows the assessment for an already-triaged patient without any extra assessment call", async () => {
    (api.dashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
      time: "10:00", sim_time_minutes: 0,
      scenario: { name: "n", title: "n", description: "", arrival_rate_per_hour: 1 },
      load: { load_ratio: 0.5, operating_mode: "NORMAL", lambda: 1 },
      departments: [], waiting_queue: [],
      full_queue: [
        {
          patient_id: "P-1", chief_complaint: "chest pain", status: "TRIAGED",
          clinical_assessment: {
            department: "ICU", department_reasoning: "High risk.", acuity_tier: 1,
            reconciled_admission_risk: 0.9, reconciled_icu_risk: 0.8, branches_agree: true,
            confidence_note: "High confidence.", top_diagnoses: ["ACS"], red_flags: ["hypotension"],
            rag_narrative: "Similar cases needed ICU.",
          },
        },
      ],
      waiting_count: 0, triaged_count: 1, untriaged_count: 0, admitted_count: 0, recent_events: [],
    });
    render(<ClinicalIntelligence />);
    await waitFor(() => expect(screen.getByText("High risk.")).toBeInTheDocument());
    expect(screen.getByText(/ACS/)).toBeInTheDocument();
    expect(screen.getByText(/hypotension/)).toBeInTheDocument();
    expect(api.dashboard).toHaveBeenCalledTimes(1); // no per-patient assess() call triggered
  });
});
