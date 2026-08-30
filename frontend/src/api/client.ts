import type {
  AgentResponse,
  AssessResponse,
  HospitalStateResponse,
  PatientDetail,
  PatientSummary,
  ScenarioInfo,
  SimulationDashboard,
  ToolExecuteResult,
  TriageResult,
  WaitingPatient,
} from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

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
  chat: (session_id: string, message: string) =>
    req<AgentResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id, message }),
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
  assessPatient: (id: string, session_id: string) =>
    req<AssessResponse>(`/api/patients/${encodeURIComponent(id)}/assess?session_id=${session_id}`, {
      method: "POST",
    }),

  // Hospital
  hospitalState: (session_id?: string) =>
    req<HospitalStateResponse>(`/api/hospital/state${session_id ? `?session_id=${session_id}` : ""}`),

  // Simulation
  scenarios: () => req<ScenarioInfo[]>("/api/simulation/scenarios"),
  dashboard: () => req<SimulationDashboard>("/api/simulation/dashboard"),
  loadScenario: (name: string) =>
    req<SimulationDashboard>("/api/simulation/scenario", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  step: (minutes: number, auto_generate_arrivals = true) =>
    req<Record<string, unknown>>("/api/simulation/step", {
      method: "POST",
      body: JSON.stringify({ minutes, auto_generate_arrivals }),
    }),
  triggerArrival: (target_acuity?: number) =>
    req<WaitingPatient>(
      `/api/simulation/arrival${target_acuity ? `?target_acuity=${target_acuity}` : ""}`,
      { method: "POST" }
    ),
  triageSimulated: (patient_id: string) =>
    req<TriageResult>(`/api/simulation/triage/${encodeURIComponent(patient_id)}`, { method: "POST" }),
  admitSimulated: (
    session_id: string,
    patient_id: string,
    department?: string,
    custom_los_min?: number
  ) =>
    req<ToolExecuteResult>("/api/simulation/admit", {
      method: "POST",
      body: JSON.stringify({ session_id, patient_id, department, custom_los_min }),
    }),
};
