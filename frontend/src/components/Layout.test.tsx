import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./Layout";

vi.mock("../api/client", () => ({
  api: { listHospitals: vi.fn().mockResolvedValue([]), dashboard: vi.fn().mockResolvedValue(null) },
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({
    ready: true,
    history: [],
    chatBusy: false,
    chatAwaitingConfirmation: false,
    awaitingEntryId: null,
    sendChat: vi.fn(),
    resolveChatConfirmation: vi.fn(),
    clearChat: vi.fn(),
    pendingDirectAction: null,
    resolvingDirectAction: false,
    resolveDirectAction: vi.fn(),
    hospitalId: "default",
    setHospitalId: vi.fn(),
    mutationTick: 0,
  }),
}));

describe("Layout", () => {
  it("renders the 4 primary nav items and the assistant panel by default", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<div>Dashboard content</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    );
    for (const label of ["Overview", "Hospital Network", "Live Operations", "Patients"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    // Phase 5: Simulation / System Architecture / Clinical Intelligence are
    // no longer in primary nav (still reachable at their existing routes).
    for (const label of ["Clinical Intelligence", "Simulation", "System Architecture"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
    expect(screen.getByText("Dashboard content")).toBeInTheDocument();
    expect(screen.getByText("TriageGuard Assistant")).toBeInTheDocument();
  });

  it("toggles the assistant panel closed and back open", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<div>Dashboard content</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    );
    const toggle = screen.getByText("Ops Assistant");
    fireEvent.click(toggle);
    expect(screen.queryByText("TriageGuard Assistant")).not.toBeInTheDocument();
    fireEvent.click(toggle);
    expect(screen.getByText("TriageGuard Assistant")).toBeInTheDocument();
  });
});
