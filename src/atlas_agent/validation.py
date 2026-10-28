"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    """Prevent identity and resolved-field substitution."""
    errors: list[str] = []
    args = decision.arguments

    if "user_id" in args and args["user_id"] != request.authenticated_user_id:
        errors.append("user_id_must_match_authenticated_user")

    for field in ("project_id", "resource_id", "start", "end"):
        if field not in args:
            continue
        trusted_value = getattr(request, field)
        if trusted_value is None:
            # Read-only lookups may explore an unresolved field, but a
            # reservation must never fill one in on the model's say-so.
            if decision.tool == "reserve_resource":
                errors.append(f"{field}_unresolved_in_request")
        elif args[field] != trusted_value:
            errors.append(f"{field}_does_not_match_request")

    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    """Return an error for every required context fact that is false."""
    checks = [
        (context.arguments_complete, "reservation_arguments_incomplete"),
        (context.policy_retrieved, "policy_not_retrieved"),
        (context.authorizations_checked, "authorizations_not_checked"),
        (context.resource_status_checked, "resource_status_not_checked"),
        (context.project_authorized, "project_not_authorized"),
        (context.qualification_current, "required_qualification_missing"),
        (context.resource_available, "resource_not_available"),
        (context.calibration_current, "calibration_not_current"),
        (context.maintenance_clear, "resource_under_maintenance"),
        (context.after_hours_approval_satisfied, "after_hours_approval_missing"),
    ]
    return [error for satisfied, error in checks if not satisfied]


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    """Validate citations, claimed actions, and final status."""
    errors: list[str] = []

    # 3a. Every citation must come from a successful search in this run.
    unretrieved = sorted(set(response.citations) - retrieved_passage_ids(trace))
    if unretrieved:
        errors.append(f"citations_not_retrieved: {', '.join(unretrieved)}")

    # 3b. Every claimed action must match a successful mutation in the trace.
    successful_mutations = [
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    ]
    if any(action not in successful_mutations for action in response.completed_actions):
        errors.append("completed_action_not_in_successful_trace")

    # 3c. "completed" requires a real mutation and at least one claimed action.
    if response.status == "completed":
        if not successful_mutations:
            errors.append("completed_without_successful_mutation")
        if not response.completed_actions:
            errors.append("completed_without_claimed_action")

    # 3d. Any other status must not claim actions or hide a mutation.
    else:
        if response.completed_actions:
            errors.append("non_completed_status_claims_actions")
        if successful_mutations:
            errors.append("successful_mutation_not_reported")

    return errors
