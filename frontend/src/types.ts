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
  rag_trajectory?: string | null;
  rag_urgency?: "low" | "moderate" | "high" | "critical" | "unknown" | null;
  rag_evidence_strength?: number | null;
  rag_escalation_concern?: boolean | null;
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

export interface DepartmentConfigInput {
  capacity: number;
  occupied: number;
  status: "OPEN" | "CLOSED" | "RESTRICTED";
}

export interface RegisterHospitalResult {
  hospital_id: string;
  hospital_name: string;
}

export interface CalibrationScenario {
  scenario_id: string;
  description: string;
  candidate_departments: string[];
  preferred_department: string;
  reason: string;
}

export interface CalibrationScenariosResponse {
  hospital_id: string;
  scenario_count: number;
  scenarios: CalibrationScenario[];
}

export interface CalibrationStatus {
  hospital_id: string;
  calibrated: boolean;
}

export interface CalibrationSubmitResult {
  hospital_id: string;
  calibrated: boolean;
  trained_scenarios: number | null;
  artifact_path: string;
}

export interface HospitalInfo {
  hospital_id: string;
  hospital_name: string;
  config_path: string;
}

export interface DashboardDepartment {
  name: string;
  capacity: number;
  occupied: number;
  available: number;
  occupancy_pct: number;
  status: string;
}

// Short-form keys as returned by /api/simulation/* endpoints (distinct from
// PatientVitals' long-form keys, which /api/patients/* returns instead).
export interface SimVitals {
  hr?: number;
  rr?: number;
  spo2?: number;
  sbp?: number;
  dbp?: number;
  temp?: number;
  pain?: number;
}

export interface WaitingPatient {
  patient_id: string;
  age: number;
  sex: string;
  chief_complaint: string;
  vitals: SimVitals;
  acuity: number;
  status: string;
  // Sim-clock minute the patient arrived — combine with the dashboard's
  // sim_time_minutes to get elapsed wait time. Always present on
  // simulation-sourced patients; optional here only for defensiveness.
  arrival_time_min?: number;
  expected_los_min?: number;
  // Only accrues once a patient is admitted (occupying a bed) — stays 0 for
  // ARRIVED/TRIAGED patients, so it is not a general "time waited" field.
  elapsed_los_min?: number;
  remaining_los_min?: number;
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
  waiting_queue: WaitingPatient[];   // compact (5 items) for dashboard preview
  full_queue: (WaitingPatient & {
    status: string;
    clinical_assessment?: Record<string, unknown> | null;
    operational_decision?: Record<string, unknown> | null;
    metadata?: Record<string, unknown> | null;
  })[];                              // full queue for Live Hospital panel
  waiting_count: number;
  triaged_count: number;
  untriaged_count: number;
  admitted_count: number;
  recent_events: string[];
}

export interface ScenarioInfo {
  name: string;
  title: string;
  description: string;
  arrival_rate_per_hour: number;
}

export interface OperationalDecision {
  clinical_department: string;
  operational_department: string;
  // Phase 9: immutable AI/policy recommendation snapshot + override tracking.
  ai_operational_department: string;
  nurse_override: boolean;
  override_reason: string | null;
  available_beds_in_clinical_dept: number;
  operating_mode: string;
  lambda: number;
  capacity_warning: boolean;
  confirmation_required: boolean;
  recommendation_summary: string;
  // Re-triage: this assessment replaced a prior one for the same patient
  // (patient was already TRIAGED, not yet admitted). Present/true only on
  // a re-triage result, not a first triage.
  retriage?: boolean;
  previous_operational_department?: string | null;
  previous_nurse_override?: boolean;
  previous_override_reason?: string | null;
}

export interface TriageResult {
  patient_id: string;
  clinical_assessment: AssessmentResult;
  operational_decision: OperationalDecision;
  patient: WaitingPatient & { department?: string | null };
}

export interface OverrideResult {
  patient_id: string;
  previous_department: string | null;
  ai_operational_department: string | null;
  operational_department: string;
  nurse_override: true;
  override_reason: string | null;
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
