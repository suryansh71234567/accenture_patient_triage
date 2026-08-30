"""
hospital_tools.py
-----------------
READ and WRITE tools for hospital operational state.

Design rules
------------
* The LLM never stores hospital state itself.
* get_hospital_state (READ) — always fetches fresh data from HospitalStateService.
* propose_hospital_calibration (COMPUTE) — validates a proposed update; returns
  the validated proposal WITHOUT committing it.
* commit_hospital_calibration (WRITE, requires_approval=True) — applies a
  previously validated+approved update, then triggers λ recalculation.

The agent MUST call propose first, get nurse confirmation, then commit.
The runtime enforces the approval gate on WRITE tools.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from triageguard_agent.schemas.tool_result import ToolResult

logger = logging.getLogger(__name__)

TOOL_GET_STATE = "get_hospital_state"
TOOL_PROPOSE = "propose_hospital_calibration"
TOOL_COMMIT = "commit_hospital_calibration"


def _resolve_state_service(hospital_id: Optional[str]):
    """
    Look up the HospitalStateService for `hospital_id` via the registry.
    Omitted/None resolves to the default hospital, which is bound to the
    same process-wide singleton these tools always used — existing
    single-hospital behavior is unchanged when hospital_id isn't passed.
    """
    from triageguard_agent.hospital.hospital_registry import (
        DEFAULT_HOSPITAL_ID,
        get_default_registry,
    )

    registry = get_default_registry()
    return registry.get(hospital_id or DEFAULT_HOSPITAL_ID).state_service


# ---------------------------------------------------------------------------
# Shared input schema for the "update" / "validated_update" object
# ---------------------------------------------------------------------------
#
# Field names match HospitalStateService.validate_update()'s own accepted
# keys exactly (capacity, occupied, status) — this is not a new schema, it
# is that existing validation contract made explicit to the LLM.
#
# Previously this was a bare {"type": "object"} with only a prose
# description ("Fields to update: capacity, occupied, status.") and no
# actual JSON Schema properties. An LLM tool-calling API only reliably fills
# in fields that are declared as real schema properties — prose in a
# description is not enough — so the model was calling
# propose_hospital_calibration(department="ICU", update={}) every time: a
# syntactically valid call with a structurally empty payload, which then
# failed the tool's own "update must be a non-empty dict" check identically
# on every retry. Declaring the real properties fixes that at the source.
_HOSPITAL_ID_PROPERTY: Dict[str, Any] = {
    "type": "string",
    "description": (
        "Which registered hospital this applies to. Omit to use the "
        "default hospital."
    ),
}

_HOSPITAL_UPDATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "The hospital state fields to change. Include only the field(s) "
        "actually being changed — e.g. to set ICU occupied beds to 9, pass "
        "{\"occupied\": 9}."
    ),
    "properties": {
        "capacity": {"type": "integer", "description": "Total beds/slots for this department."},
        "occupied": {"type": "integer", "description": "Currently occupied beds/slots."},
        "status": {"type": "string", "description": "One of: OPEN, CLOSED, RESTRICTED."},
    },
}


# ---------------------------------------------------------------------------
# Handler: get_hospital_state
# ---------------------------------------------------------------------------

def get_hospital_state(
    department: Optional[str] = None,
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """
    Return the current operational state of the hospital (or one department).

    Always fetches live data — never uses a cached/static copy.
    Includes a staleness flag if the state was last updated > 30 min ago.

    hospital_id : which registered hospital's state to read. Omit for the
        default hospital (unchanged single-hospital behavior).
    """
    try:
        svc = _resolve_state_service(hospital_id)

        if department:
            state = svc.get_state(department)
            if state is None:
                return ToolResult.fail(
                    TOOL_GET_STATE,
                    "DEPARTMENT_NOT_FOUND",
                    f"Department {department!r} is not in the hospital configuration.",
                )
            stale = svc.is_stale(department)
            data = {
                "department": department,
                "state": state,
                "is_stale": stale,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            all_state = svc.get_all()
            stale_depts = [d for d in all_state if svc.is_stale(d)]
            data = {
                "departments": all_state,
                "stale_departments": stale_depts,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

        return ToolResult.ok(TOOL_GET_STATE, data, metadata={"live": True, "hospital_id": hospital_id})

    except KeyError as ke:
        return ToolResult.fail(TOOL_GET_STATE, "HOSPITAL_NOT_FOUND", str(ke))
    except Exception as exc:
        logger.exception("get_hospital_state failed.")
        return ToolResult.fail(TOOL_GET_STATE, "SERVICE_ERROR", str(exc))


# ---------------------------------------------------------------------------
# Handler: propose_hospital_calibration
# ---------------------------------------------------------------------------

def propose_hospital_calibration(
    department: str,
    update: Dict[str, Any],
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """
    Validate a proposed hospital state change WITHOUT committing it.

    Parameters
    ----------
    department : The department to update (e.g. "ICU").
    update     : Dict with one or more of: capacity, occupied, status.
    hospital_id : which registered hospital this proposal applies to. Omit
        for the default hospital.

    Returns a validated proposal dict that can be shown to the nurse for
    confirmation before commit_hospital_calibration is called.
    """
    if not department:
        return ToolResult.fail(
            TOOL_PROPOSE,
            "MISSING_DEPARTMENT",
            "department is required.",
        )
    if not isinstance(update, dict) or not update:
        return ToolResult.fail(
            TOOL_PROPOSE,
            "MISSING_UPDATE",
            "update must be a non-empty dict.",
        )

    try:
        svc = _resolve_state_service(hospital_id)
        validated = svc.validate_update(department, update)
    except KeyError as ke:
        return ToolResult.fail(TOOL_PROPOSE, "HOSPITAL_NOT_FOUND", str(ke))
    except ValueError as ve:
        return ToolResult.fail(TOOL_PROPOSE, "VALIDATION_ERROR", str(ve))
    except Exception as exc:
        logger.exception("propose_hospital_calibration failed.")
        return ToolResult.fail(TOOL_PROPOSE, "SERVICE_ERROR", str(exc))

    return ToolResult.ok(
        TOOL_PROPOSE,
        {
            "department": department,
            "hospital_id": hospital_id,
            "proposed_update": validated,
            "confirmation_required": True,
            "message": (
                f"Proposed update for {department}: {validated}. "
                "Please confirm before committing. "
                "(commit_hospital_calibration must be called with this same "
                "hospital_id.)"
            ),
        },
        metadata={"requires_confirmation": True},
    )


# ---------------------------------------------------------------------------
# Handler: commit_hospital_calibration
# ---------------------------------------------------------------------------

def commit_hospital_calibration(
    department: str,
    validated_update: Dict[str, Any],
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """
    Apply a previously validated hospital state update.

    This is a WRITE tool — the runtime requires human approval before calling.
    After committing, triggers HospitalLoadController to recalculate λ.

    Parameters
    ----------
    department       : Department to update.
    validated_update : The validated dict previously returned by propose.
    hospital_id      : Must match the hospital_id the proposal was validated
        against. Omit for the default hospital.
    """
    if not department or not validated_update:
        return ToolResult.fail(
            TOOL_COMMIT,
            "MISSING_ARGS",
            "Both department and validated_update are required.",
        )

    try:
        from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController

        svc = _resolve_state_service(hospital_id)
        svc.apply_update(department, validated_update)

        # Recalculate λ after state change
        all_state = svc.get_all()
        controller = HospitalLoadController()
        load_result = controller.recalculate(all_state)

    except KeyError as ke:
        return ToolResult.fail(TOOL_COMMIT, "HOSPITAL_NOT_FOUND", str(ke))
    except ValueError as ve:
        return ToolResult.fail(TOOL_COMMIT, "VALIDATION_ERROR", str(ve))
    except Exception as exc:
        logger.exception("commit_hospital_calibration failed.")
        return ToolResult.fail(TOOL_COMMIT, "SERVICE_ERROR", str(exc))

    return ToolResult.ok(
        TOOL_COMMIT,
        {
            "department": department,
            "hospital_id": hospital_id,
            "applied_update": validated_update,
            "new_state": svc.get_state(department),
            "operating_mode": load_result["operating_mode"],
            "lambda": load_result["lambda"],
            "load_ratio": load_result["load_ratio"],
            "committed_at": datetime.now(timezone.utc).isoformat(),
        },
        metadata={"lambda_recalculated": True},
    )


# ---------------------------------------------------------------------------
# ToolSpec factories
# ---------------------------------------------------------------------------

def get_hospital_state_spec():
    from triageguard_agent.tools.registry import ToolSpec, READ
    return ToolSpec(
        name=TOOL_GET_STATE,
        description=(
            "Fetch the current operational state of the hospital or a specific department. "
            "Always fetches live data — never use remembered occupancy. "
            "Includes a staleness flag when data is > 30 minutes old."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "Optional department name (e.g. 'ICU'). Omit for full hospital state.",
                },
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
        },
        handler=lambda **kwargs: get_hospital_state(**kwargs),
        risk_level=READ,
        side_effect=False,
        requires_approval=False,
    )


def propose_hospital_calibration_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name=TOOL_PROPOSE,
        description=(
            "Validate a proposed hospital state CHANGE (e.g. occupancy update, capacity change, "
            "resource closure). Does NOT commit the change — returns a proposal for nurse confirmation. "
            "Always call this before commit_hospital_calibration. "
            "Only use this when the nurse is explicitly asking to CHANGE a value (e.g. 'set ICU "
            "occupied beds to 9'). For questions about the CURRENT value (e.g. 'how many ICU beds "
            "are free/occupied'), use get_hospital_state instead — never this tool. Do not use "
            "this tool for any request unrelated to hospital resource capacity/occupancy/status."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "update": _HOSPITAL_UPDATE_SCHEMA,
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
            "required": ["department", "update"],
        },
        handler=lambda **kwargs: propose_hospital_calibration(**kwargs),
        risk_level=COMPUTE,
        side_effect=False,
        requires_approval=False,
    )


def commit_hospital_calibration_spec():
    from triageguard_agent.tools.registry import ToolSpec, WRITE
    return ToolSpec(
        name=TOOL_COMMIT,
        description=(
            "Apply an approved hospital state CHANGE and recalculate operating mode + λ. "
            "WRITE tool — requires human approval before execution. "
            "Must only be called after propose_hospital_calibration + nurse confirmation, and only "
            "when the nurse's own request actually asked for a hospital resource capacity/occupancy/"
            "status change. Never call this for a READ/informational request, and never call this "
            "to substitute for a capability that does not exist (e.g. there is no patient-deletion "
            "tool — do not call this instead)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "department":       {"type": "string"},
                "validated_update": _HOSPITAL_UPDATE_SCHEMA,
                "hospital_id":      _HOSPITAL_ID_PROPERTY,
            },
            "required": ["department", "validated_update"],
        },
        handler=lambda **kwargs: commit_hospital_calibration(**kwargs),
        risk_level=WRITE,
        side_effect=True,
        requires_approval=True,
    )
