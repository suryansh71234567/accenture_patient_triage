import type { DepartmentState, ResourceCheck } from "../types";

/**
 * Mirrors api_server.py::_resource_check exactly (same thresholds, same
 * fallback-to-ED_OBS rule) so that after a chat-confirmed observation write
 * already returns a fresh clinical department for free, we can show an
 * up-to-date "is this department open?" hint without an extra full
 * XGBoost+RAG pipeline call. This is UI display logic only — never treated
 * as the authoritative allocation decision (that only exists, fully
 * confirmation-gated, in the Live Hospital simulation flow).
 */
export function computeResourceCheck(
  department: string,
  departments: Record<string, DepartmentState>
): ResourceCheck {
  const state = departments[department];
  if (!state) {
    return {
      preferred_department: department,
      allocated_department: department,
      capacity: 0,
      occupied: 0,
      available: 0,
      resource_constrained: false,
      tight: false,
      note: null,
    };
  }

  const { capacity, occupied, available } = state;
  const constrained = ["ICU", "CICU", "ADMITTED_GEN"].includes(department) && available <= 0;
  const tight = ["ICU", "CICU"].includes(department) && available === 1;

  let allocated_department = department;
  let note: string | null = null;
  if (constrained) {
    allocated_department = department !== "ED_OBS" ? "ED_OBS" : department;
    note = `${department} is at capacity (${occupied}/${capacity}). Recommending ${allocated_department} pending capacity, with staff escalation for transfer.`;
  } else if (tight) {
    note = `${department} has only 1 bed remaining (${occupied}/${capacity}) — confirm before allocating it.`;
  }

  return {
    preferred_department: department,
    allocated_department,
    capacity,
    occupied,
    available,
    resource_constrained: constrained,
    tight,
    note,
  };
}
