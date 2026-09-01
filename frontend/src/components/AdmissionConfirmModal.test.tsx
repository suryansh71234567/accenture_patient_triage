import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AdmissionConfirmModal } from "./AdmissionConfirmModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    executeTool: vi.fn(),
    confirmTool: vi.fn(),
  },
}));

const bumpMutationTick = vi.fn();
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ sessionId: "s1", bumpMutationTick }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("AdmissionConfirmModal", () => {
  it("starts in the confirm phase showing the destination", () => {
    render(
      <AdmissionConfirmModal patientId="P-1" department="ICU" reason="ICU has beds" hospitalId="h1" onClose={vi.fn()} onSuccess={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: "Confirm Admission" })).toBeInTheDocument();
    expect(screen.getByText("ICU")).toBeInTheDocument();
  });

  it("goes confirm -> applying -> success when executeTool executes directly", async () => {
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "executed" });
    const onSuccess = vi.fn();
    render(
      <AdmissionConfirmModal patientId="P-1" department="ICU" reason="" hospitalId="h1" onClose={vi.fn()} onSuccess={onSuccess} />
    );
    fireEvent.click(screen.getByRole("button", { name: "Confirm Admission" }));
    await waitFor(() => expect(screen.getByText("✓ Admission complete")).toBeInTheDocument());
    expect(bumpMutationTick).toHaveBeenCalled();
    expect(onSuccess).toHaveBeenCalled();
  });

  it("calls confirmTool when executeTool returns awaiting_confirmation, then succeeds", async () => {
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "awaiting_confirmation" });
    (api.confirmTool as ReturnType<typeof vi.fn>).mockResolvedValue({ response_type: "confirmation" });
    render(
      <AdmissionConfirmModal patientId="P-1" department="ICU" reason="" hospitalId="h1" onClose={vi.fn()} onSuccess={vi.fn()} />
    );
    fireEvent.click(screen.getByRole("button", { name: "Confirm Admission" }));
    await waitFor(() => expect(api.confirmTool).toHaveBeenCalledWith("s1", true));
    await waitFor(() => expect(screen.getByText("✓ Admission complete")).toBeInTheDocument());
  });

  it("shows the failure state and allows retry when the tool fails", async () => {
    (api.executeTool as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "failed" });
    render(
      <AdmissionConfirmModal patientId="P-1" department="ICU" reason="" hospitalId="h1" onClose={vi.fn()} onSuccess={vi.fn()} />
    );
    fireEvent.click(screen.getByRole("button", { name: "Confirm Admission" }));
    await waitFor(() => expect(screen.getByText("Admission failed")).toBeInTheDocument());
    expect(screen.getByText("Try Again")).toBeInTheDocument();
  });
});
