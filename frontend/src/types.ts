// Types mirror the FastAPI contracts in api_server.py exactly — no fields
// are invented here that the backend doesn't actually return.

export interface PatientVitals {
  heart_rate: number | null;
  resp_rate: number | null;
  spo2: number | null;
  sbp: number | null;
  dbp: number | null;
  temperature: number | null;
  pain_score: number | null;
}

export interface PatientSummary {
  patient_id: string;
  age: number | null;
  sex: string | null;
  chief_complaint: string;
  acuity: number | null;
  time_elapsed_minutes: number;
  vitals: PatientVitals;
  last_updated: string;
}

export interface Observation {
  encounter_id?: string | null;
  timestamp: string;
  type: string;
  note?: string;
  [vitalField: string]: unknown;
}

export interface PatientDetail {
  summary: PatientSummary;
  observations: Observation[];
}

export interface AssessmentResult {
  department: string;
  department_reasoning: string;
  acuity_tier: number;
  reconciled_admission_risk: number;
  reconciled_icu_risk: number;
  branches_agree: boolean;
  confidence_note: string;
  top_diagnoses: string[];
  red_flags: string[];
  rag_disposition?: string;
  rag_escalation?: string;
  rag_narrative?: string;
  _xgb_raw?: Record<string, number>;
}

export interface ResourceCheck {
  preferred_department: string;
  allocated_department: string;
  capacity: number;
  occupied: number;
  available: number;
  resource_constrained: boolean;
  tight: boolean;
  note: string | null;
}

export interface AssessResponse {
  assessment: AssessmentResult;
  resource_check: ResourceCheck | null;
}

export interface DepartmentState {
  capacity: number;
  occupied: number;
  available: number;
  status: "OPEN" | "CLOSED" | "RESTRICTED";
  last_updated: string;
}

export interface HospitalStateResponse {
  departments: Record<string, DepartmentState>;
  stale_departments: string[];
  fetched_at: string;
}

export interface DashboardDepartment {
  name: string;
  capacity: number;
  occupied: number;
  available: number;
  occupancy_pct: number;
  status: string;
}

export interface WaitingPatient {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  vitals: Record<string, number>;
  acuity: number;
  status: string;
}

export interface SimulationDashboard {
  time: string;
  sim_time_minutes: number;
  scenario: {
    name: string;
    title: string;
    description: string;
    arrival_rate_per_hour: number;
  };
  load: {
    load_ratio: number;
    operating_mode: "NORMAL" | "HIGH_LOAD" | "CRITICAL" | string;
    lambda: number;
  };
  departments: DashboardDepartment[];
  waiting_queue: WaitingPatient[];
  waiting_count: number;
  admitted_count: number;
  recent_events: string[];
}

export interface ScenarioInfo {
  name: string;
  title: string;
  description: string;
  arrival_rate_per_hour: number;
}

export interface TriageResult {
  patient_id: string;
  clinical_assessment: AssessmentResult;
  operational_decision: {
    clinical_department: string;
    operational_department: string;
    available_beds_in_clinical_dept: number;
    operating_mode: string;
    lambda: number;
    capacity_warning: boolean;
    confirmation_required: boolean;
    recommendation_summary: string;
  };
  patient: WaitingPatient & { department?: string | null };
}

// ---- Agent chat ----

export type ResponseType =
  | "information"
  | "assessment"
  | "confirmation"
  | "approval_required"
  | "error";

export interface AgentAction {
  tool: string;
  status: "executed" | "failed" | "awaiting_confirmation";
  data?: Record<string, unknown> | null;
  error?: { code: string; message: string } | null;
  payload?: Record<string, unknown>;
}

export interface AgentResponse {
  message: string;
  response_type: ResponseType;
  patient_id: string | null;
  actions: AgentAction[];
  evidence: unknown[];
  human_approval_required: boolean;
}

export interface ToolExecuteResult {
  status: "executed" | "failed" | "awaiting_confirmation";
  data?: Record<string, unknown> | null;
  error?: { code: string; message: string } | null;
  tool_name?: string;
  kwargs?: Record<string, unknown>;
  description?: string;
}

export type ChatEntry =
  | { kind: "user"; text: string; id: string }
  | { kind: "agent"; response: AgentResponse; id: string };
