import type {
  AgentResponse,
  AssessResponse,
  CalibrationScenariosResponse,
  CalibrationStatus,
  CalibrationSubmitResult,
  DepartmentConfigInput,
  HospitalInfo,
  HospitalStateResponse,
  PatientDetail,
  PatientSummary,
  OverrideResult,
  RegisterHospitalResult,
  ScenarioInfo,
  SimulationDashboard,
  ToolExecuteResult,
  TriageResult,
  WaitingPatient,
} from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

// Builds "?a=1&b=2"-style query strings, silently dropping undefined/null/""
// values so an omitted param (e.g. hospital_id) never appears on the wire —
// existing request shapes are unchanged when a caller doesn't pass one.
function qs(params: Record<string, string | number | undefined | null>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // ignore
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Session
  createSession: (role: string) =>
    req<{ session_id: string; role: string }>("/api/session", {
      method: "POST",
      body: JSON.stringify({ role }),
    }),
  getSessionState: (session_id: string) =>
    req<Record<string, unknown>>(`/api/session/${encodeURIComponent(session_id)}`),

  // Chat
  chat: (session_id: string, message: string, hospital_id?: string) =>
    req<AgentResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id, message, hospital_id }),
    }),

  // Generic tool execute / confirm
  executeTool: (session_id: string, tool_name: string, kwargs: Record<string, unknown>) =>
    req<ToolExecuteResult>("/api/tools/execute", {
      method: "POST",
      body: JSON.stringify({ session_id, tool_name, kwargs }),
    }),

  confirmTool: (session_id: string, approve: boolean) =>
    req<AgentResponse>("/api/tools/confirm", {
      method: "POST",
      body: JSON.stringify({ session_id, approve }),
    }),

  // Patients
  listPatients: () => req<PatientSummary[]>("/api/patients"),
  getPatient: (id: string) => req<PatientDetail>(`/api/patients/${encodeURIComponent(id)}`),
  assessPatient: (id: string, session_id: string, hospital_id?: string) =>
    req<AssessResponse>(
      `/api/patients/${encodeURIComponent(id)}/assess${qs({ session_id, hospital_id })}`,
      { method: "POST" }
    ),

  // Hospital
  listHospitals: () => req<HospitalInfo[]>("/api/hospitals"),
  registerHospital: (payload: {
    hospital_id: string;
    hospital_name: string;
    departments: Record<string, DepartmentConfigInput>;
  }) =>
    req<RegisterHospitalResult>("/api/hospitals", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  calibrationStatus: (hospitalId: string) =>
    req<CalibrationStatus>(`/api/hospitals/${encodeURIComponent(hospitalId)}/calibration/status`),
  calibrationScenarios: (hospitalId: string) =>
    req<CalibrationScenariosResponse>(`/api/hospitals/${encodeURIComponent(hospitalId)}/calibration/scenarios`),
  submitCalibration: (hospitalId: string, responses: Record<string, string>) =>
    req<CalibrationSubmitResult>(`/api/hospitals/${encodeURIComponent(hospitalId)}/calibration/submit`, {
      method: "POST",
      body: JSON.stringify({ responses }),
    }),
  hospitalState: (session_id?: string, hospital_id?: string) =>
    req<HospitalStateResponse>(`/api/hospital/state${qs({ session_id, hospital_id })}`),

  // Simulation
  scenarios: (hospital_id?: string) =>
    req<ScenarioInfo[]>(`/api/simulation/scenarios${qs({ hospital_id })}`),
  dashboard: (hospital_id?: string) =>
    req<SimulationDashboard>(`/api/simulation/dashboard${qs({ hospital_id })}`),
  loadScenario: (name: string, hospital_id?: string) =>
    req<SimulationDashboard>("/api/simulation/scenario", {
      method: "POST",
      body: JSON.stringify({ name, hospital_id }),
    }),
  step: (minutes: number, auto_generate_arrivals = true, hospital_id?: string) =>
    req<Record<string, unknown>>("/api/simulation/step", {
      method: "POST",
      body: JSON.stringify({ minutes, auto_generate_arrivals, hospital_id }),
    }),
  triggerArrival: (target_acuity?: number, hospital_id?: string) =>
    req<WaitingPatient>(
      `/api/simulation/arrival${qs({ target_acuity, hospital_id })}`,
      { method: "POST" }
    ),
  triageSimulated: (patient_id: string, hospital_id?: string) =>
    req<TriageResult>(
      `/api/simulation/triage/${encodeURIComponent(patient_id)}${qs({ hospital_id })}`,
      { method: "POST" }
    ),
  admitSimulated: (
    session_id: string,
    patient_id: string,
    department?: string,
    custom_los_min?: number,
    hospital_id?: string
  ) =>
    req<ToolExecuteResult>("/api/simulation/admit", {
      method: "POST",
      body: JSON.stringify({ session_id, patient_id, department, custom_los_min, hospital_id }),
    }),

  manualArrival: (payload: {
    patient_id: string;
    chief_complaint: string;
    age: number;
    sex: string;
    acuity: number;
    hr?: number | null;
    rr?: number | null;
    spo2?: number | null;
    sbp?: number | null;
    dbp?: number | null;
    temperature?: number | null;
    pain?: number | null;
    hospital_id?: string | null;
  }) =>
    req<WaitingPatient & { has_history: boolean; history_text: string }>(
      "/api/simulation/manual-arrival",
      { method: "POST", body: JSON.stringify(payload) }
    ),

  reorderQueue: (patient_id: string, new_index: number, note?: string, hospital_id?: string) =>
    req<{ moved: boolean; queue_length: number }>("/api/simulation/queue/reorder", {
      method: "POST",
      body: JSON.stringify({ patient_id, new_index, note: note ?? "", hospital_id }),
    }),

  reorderDepartmentQueue: (patient_id: string, department: string, new_index: number, hospital_id?: string) =>
    req<{ moved: boolean }>("/api/simulation/queue/reorder-department", {
      method: "POST",
      body: JSON.stringify({ patient_id, department, new_index, hospital_id }),
    }),

  overrideDepartment: (patient_id: string, department: string, reason?: string, hospital_id?: string) =>
    req<OverrideResult>("/api/simulation/queue/override", {
      method: "POST",
      body: JSON.stringify({ patient_id, department, reason: reason ?? "", hospital_id }),
    }),

  updateSimulatedVitals: (
    patient_id: string,
    vitals: { hr?: number; rr?: number; spo2?: number; sbp?: number; dbp?: number; temp?: number; pain?: number },
    hospital_id?: string
  ) =>
    req<WaitingPatient>(`/api/simulation/patient/${encodeURIComponent(patient_id)}/vitals`, {
      method: "POST",
      body: JSON.stringify({ ...vitals, hospital_id }),
    }),
};
