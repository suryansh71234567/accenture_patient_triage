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


# ---------------------------------------------------------------------------
# Handler: get_hospital_state
# ---------------------------------------------------------------------------

def get_hospital_state(department: Optional[str] = None) -> ToolResult:
    """
    Return the current operational state of the hospital (or one department).

    Always fetches live data — never uses a cached/static copy.
    Includes a staleness flag if the state was last updated > 30 min ago.
    """
    try:
        from triageguard_agent.hospital.hospital_state_service import HospitalStateService
        svc = HospitalStateService.instance()

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

        return ToolResult.ok(TOOL_GET_STATE, data, metadata={"live": True})

    except Exception as exc:
        logger.exception("get_hospital_state failed.")
        return ToolResult.fail(TOOL_GET_STATE, "SERVICE_ERROR", str(exc))


# ---------------------------------------------------------------------------
# Handler: propose_hospital_calibration
# ---------------------------------------------------------------------------

def propose_hospital_calibration(
    department: str,
    update: Dict[str, Any],
) -> ToolResult:
    """
    Validate a proposed hospital state change WITHOUT committing it.

    Parameters
    ----------
    department : The department to update (e.g. "ICU").
    update     : Dict with one or more of: capacity, occupied, status.

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
        from triageguard_agent.hospital.hospital_state_service import HospitalStateService
        svc = HospitalStateService.instance()
        validated = svc.validate_update(department, update)
    except ValueError as ve:
        return ToolResult.fail(TOOL_PROPOSE, "VALIDATION_ERROR", str(ve))
    except Exception as exc:
        logger.exception("propose_hospital_calibration failed.")
        return ToolResult.fail(TOOL_PROPOSE, "SERVICE_ERROR", str(exc))

    return ToolResult.ok(
        TOOL_PROPOSE,
        {
            "department": department,
            "proposed_update": validated,
            "confirmation_required": True,
            "message": (
                f"Proposed update for {department}: {validated}. "
                "Please confirm before committing."
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
) -> ToolResult:
    """
    Apply a previously validated hospital state update.

    This is a WRITE tool — the runtime requires human approval before calling.
    After committing, triggers HospitalLoadController to recalculate λ.

    Parameters
    ----------
    department       : Department to update.
    validated_update : The validated dict previously returned by propose.
    """
    if not department or not validated_update:
        return ToolResult.fail(
            TOOL_COMMIT,
            "MISSING_ARGS",
            "Both department and validated_update are required.",
        )

    try:
        from triageguard_agent.hospital.hospital_state_service import HospitalStateService
        from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController

        svc = HospitalStateService.instance()
        svc.apply_update(department, validated_update)

        # Recalculate λ after state change
        all_state = svc.get_all()
        controller = HospitalLoadController()
        load_result = controller.recalculate(all_state)

    except ValueError as ve:
        return ToolResult.fail(TOOL_COMMIT, "VALIDATION_ERROR", str(ve))
    except Exception as exc:
        logger.exception("commit_hospital_calibration failed.")
        return ToolResult.fail(TOOL_COMMIT, "SERVICE_ERROR", str(exc))

    return ToolResult.ok(
        TOOL_COMMIT,
        {
            "department": department,
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
                }
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
            "Validate a proposed hospital state change (e.g. occupancy update, capacity change, "
            "resource closure). Does NOT commit the change — returns a proposal for nurse confirmation. "
            "Always call this before commit_hospital_calibration."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "update": {
                    "type": "object",
                    "description": "Fields to update: capacity, occupied, status.",
                },
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
            "Apply an approved hospital state update and recalculate operating mode + λ. "
            "WRITE tool — requires human approval before execution. "
            "Must only be called after propose_hospital_calibration + nurse confirmation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "department":       {"type": "string"},
                "validated_update": {"type": "object"},
            },
            "required": ["department", "validated_update"],
        },
        handler=lambda **kwargs: commit_hospital_calibration(**kwargs),
        risk_level=WRITE,
        side_effect=True,
        requires_approval=True,
    )
