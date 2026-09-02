import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ManualIntakeForm } from "./ManualIntakeForm";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { manualArrival: vi.fn() },
}));
vi.mock("../state/SessionContext", () => ({
  useSession: () => ({ hospitalId: "default" }),
}));

function renderForm(props: Partial<{ onSuccess: () => void; onClose: () => void }> = {}) {
  const onSuccess = props.onSuccess ?? vi.fn();
  const onClose = props.onClose ?? vi.fn();
  const utils = render(<ManualIntakeForm onSuccess={onSuccess} onClose={onClose} />);
  return { ...utils, onSuccess, onClose };
}

function fill(label: string, value: string) {
  fireEvent.change(screen.getByPlaceholderText(label), { target: { value } });
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ManualIntakeForm", () => {
  it("renders the identity/complaint/demographic/vitals fields", () => {
    renderForm();
    expect(screen.getByPlaceholderText("e.g. 52 or WALK-001")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g. chest pain and shortness of breath")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g. 67")).toBeInTheDocument(); // age
    expect(screen.getByPlaceholderText("0–10")).toBeInTheDocument(); // pain
    for (const ph of ["e.g. 112", "e.g. 22", "e.g. 94", "e.g. 138", "e.g. 88", "e.g. 37.2"]) {
      expect(screen.getByPlaceholderText(ph)).toBeInTheDocument();
    }
  });

  it("requires patient ID, complaint, and age before submitting — and never calls the API", () => {
    renderForm();
    fireEvent.click(screen.getByText("Add Patient to Queue"));
    expect(screen.getByText("Patient ID, complaint, and age are required.")).toBeInTheDocument();
    expect(api.manualArrival).not.toHaveBeenCalled();
  });

  it("rejects when only some required fields are filled (missing age)", () => {
    renderForm();
    fireEvent.change(screen.getByPlaceholderText("e.g. 52 or WALK-001"), { target: { value: "WALK-001" } });
    fireEvent.change(screen.getByPlaceholderText("e.g. chest pain and shortness of breath"), { target: { value: "fall" } });
    fireEvent.click(screen.getByText("Add Patient to Queue"));
    expect(screen.getByText("Patient ID, complaint, and age are required.")).toBeInTheDocument();
    expect(api.manualArrival).not.toHaveBeenCalled();
  });

  it("sends the exact real payload, converting numeric fields and defaulting blank vitals to null (never fabricated)", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockResolvedValue({
      patient_id: "WALK-001", has_history: false, history_text: "",
    });
    renderForm();
    fireEvent.change(screen.getByPlaceholderText("e.g. 52 or WALK-001"), { target: { value: "  WALK-001  " } });
    fireEvent.change(screen.getByPlaceholderText("e.g. chest pain and shortness of breath"), { target: { value: "  ankle sprain  " } });
    fireEvent.change(screen.getByPlaceholderText("e.g. 67"), { target: { value: "34" } });
    fireEvent.change(screen.getByPlaceholderText("e.g. 112"), { target: { value: "88" } }); // hr only
    // Leave rr/spo2/sbp/dbp/temperature/pain blank on purpose.

    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(api.manualArrival).toHaveBeenCalledTimes(1));
    expect(api.manualArrival).toHaveBeenCalledWith({
      patient_id: "WALK-001", // trimmed
      chief_complaint: "ankle sprain", // trimmed
      age: 34, // converted to number
      sex: "M", // default
      acuity: 3, // default
      hr: 88, // converted
      rr: null, spo2: null, sbp: null, dbp: null, temperature: null, pain: null, // blank -> null, not 0 or fabricated
      hospital_id: "default",
    });
  });

  it("sends every filled field with the exact typed values, including boundary values", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockResolvedValue({
      patient_id: "B-1", has_history: false, history_text: "",
    });
    renderForm();
    fill("e.g. 52 or WALK-001", "B-1");
    fill("e.g. chest pain and shortness of breath", "boundary test");
    fill("e.g. 67", "0"); // age boundary: newborn
    fill("e.g. 112", "300"); // hr at the backend's upper valid bound
    fill("e.g. 22", "0"); // rr lower bound
    fill("e.g. 94", "100"); // spo2 upper bound
    fill("e.g. 138", "0"); // sbp lower bound
    fill("e.g. 88", "200"); // dbp upper bound
    fill("e.g. 37.2", "45"); // temp upper bound (°C)
    fill("0–10", "10"); // pain upper bound
    fireEvent.change(screen.getByDisplayValue("Male"), { target: { value: "F" } });

    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(api.manualArrival).toHaveBeenCalledWith(
      expect.objectContaining({
        age: 0, hr: 300, rr: 0, spo2: 100, sbp: 0, dbp: 200, temperature: 45, pain: 10, sex: "F",
      })
    ));
  });

  it("shows the loading state and disables the submit button while the request is in flight", async () => {
    let resolveFn!: (v: unknown) => void;
    (api.manualArrival as ReturnType<typeof vi.fn>).mockReturnValue(new Promise((r) => { resolveFn = r; }));
    renderForm();
    fill("e.g. 52 or WALK-001", "B-1");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "40");

    fireEvent.click(screen.getByText("Add Patient to Queue"));
    expect(await screen.findByText("Adding…")).toBeInTheDocument();
    expect(screen.getByText("Adding…").closest("button")).toBeDisabled();

    resolveFn({ patient_id: "B-1", has_history: false, history_text: "" });
    await waitFor(() => expect(screen.getByText("Patient Added to Queue")).toBeInTheDocument());
  });

  it("a disabled submit button cannot be clicked again mid-request (no double API call)", async () => {
    let resolveFn!: (v: unknown) => void;
    (api.manualArrival as ReturnType<typeof vi.fn>).mockReturnValue(new Promise((r) => { resolveFn = r; }));
    renderForm();
    fill("e.g. 52 or WALK-001", "B-1");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "40");

    const btn = screen.getByText("Add Patient to Queue");
    fireEvent.click(btn);
    await screen.findByText("Adding…");
    // Rapid repeated clicks while disabled — jsdom (like real browsers) does
    // not dispatch click on a disabled button.
    fireEvent.click(screen.getByText("Adding…"));
    fireEvent.click(screen.getByText("Adding…"));
    expect(api.manualArrival).toHaveBeenCalledTimes(1);

    resolveFn({ patient_id: "B-1", has_history: false, history_text: "" });
    await waitFor(() => expect(screen.getByText("Patient Added to Queue")).toBeInTheDocument());
  });

  it("shows the 'prior record found' panel using the real has_history/history_text from the backend", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockResolvedValue({
      patient_id: "52", has_history: true, history_text: "Prior history: diabetes mellitus, 2 prior ED visit(s)",
    });
    renderForm();
    fill("e.g. 52 or WALK-001", "52");
    fill("e.g. chest pain and shortness of breath", "chest pain");
    fill("e.g. 67", "62");
    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(screen.getByText("✓ Prior hospital record found")).toBeInTheDocument());
    expect(screen.getByText("Prior history: diabetes mellitus, 2 prior ED visit(s)")).toBeInTheDocument();
    expect(screen.queryByText("⚠ No prior record found")).not.toBeInTheDocument();
  });

  it("shows the 'no prior record' panel when has_history is false — never fabricates history", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockResolvedValue({
      patient_id: "WALK-002", has_history: false, history_text: "",
    });
    renderForm();
    fill("e.g. 52 or WALK-001", "WALK-002");
    fill("e.g. chest pain and shortness of breath", "new walk-in");
    fill("e.g. 67", "22");
    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(screen.getByText("⚠ No prior record found")).toBeInTheDocument());
    expect(screen.queryByText("✓ Prior hospital record found")).not.toBeInTheDocument();
  });

  it("Done on the result view calls onSuccess and onClose", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockResolvedValue({
      patient_id: "52", has_history: false, history_text: "",
    });
    const { onSuccess, onClose } = renderForm();
    fill("e.g. 52 or WALK-001", "52");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "60");
    fireEvent.click(screen.getByText("Add Patient to Queue"));
    await waitFor(() => expect(screen.getByText("Done")).toBeInTheDocument());

    fireEvent.click(screen.getByText("Done"));
    expect(onSuccess).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("shows the backend's real error message on a 409 duplicate-active-patient conflict and does not advance to the result view", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("409 Patient '52' is already active in this hospital (status=ARRIVED).")
    );
    renderForm();
    fill("e.g. 52 or WALK-001", "52");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "60");
    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(screen.getByText(/already active in this hospital/)).toBeInTheDocument());
    expect(screen.queryByText("Patient Added to Queue")).not.toBeInTheDocument();
    // Recoverable: the form is still usable, button re-enabled.
    expect(screen.getByText("Add Patient to Queue").closest("button")).not.toBeDisabled();
  });

  it("shows the backend's real error message on a 400 validation failure (e.g. temperature out of range)", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("400 Vital 'temp' value 90 is outside the valid range [30, 45] (expected degrees Celsius).")
    );
    renderForm();
    fill("e.g. 52 or WALK-001", "WALK-003");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "50");
    fill("e.g. 37.2", "90");
    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(screen.getByText(/outside the valid range/)).toBeInTheDocument());
  });

  it("shows a generic error message on a 500 backend failure", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("500 Internal Server Error"));
    renderForm();
    fill("e.g. 52 or WALK-001", "WALK-004");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "50");
    fireEvent.click(screen.getByText("Add Patient to Queue"));

    await waitFor(() => expect(screen.getByText("500 Internal Server Error")).toBeInTheDocument());
  });

  it("a retry after a failed submission works (error is cleared and a fresh request is sent)", async () => {
    (api.manualArrival as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new Error("409 already active"))
      .mockResolvedValueOnce({ patient_id: "WALK-005", has_history: false, history_text: "" });
    renderForm();
    fill("e.g. 52 or WALK-001", "WALK-005");
    fill("e.g. chest pain and shortness of breath", "test");
    fill("e.g. 67", "50");
    fireEvent.click(screen.getByText("Add Patient to Queue"));
    await waitFor(() => expect(screen.getByText(/already active/)).toBeInTheDocument());

    fireEvent.click(screen.getByText("Add Patient to Queue"));
    await waitFor(() => expect(screen.getByText("Patient Added to Queue")).toBeInTheDocument());
    expect(api.manualArrival).toHaveBeenCalledTimes(2);
  });

  it("Cancel closes without calling the API", () => {
    const { onClose } = renderForm();
    fireEvent.click(screen.getByText("Cancel"));
    expect(onClose).toHaveBeenCalled();
    expect(api.manualArrival).not.toHaveBeenCalled();
  });

  it("closes on Escape (shared useModalA11y wiring)", () => {
    const { onClose } = renderForm();
    fireEvent.keyDown(screen.getByPlaceholderText("e.g. 52 or WALK-001"), { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
