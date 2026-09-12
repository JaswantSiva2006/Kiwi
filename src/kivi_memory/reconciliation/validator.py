"""Strict validation for model-facing reconciliation decisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kivi_memory.reconciliation.models import ReconciliationFailure, ReconciliationOp

TARGETED_OPS = {
    ReconciliationOp.REINFORCE,
    ReconciliationOp.SUPERSEDE,
    ReconciliationOp.RETRACT,
}
UNTARGETED_OPS = {ReconciliationOp.ADD, ReconciliationOp.NO_MEMORY}


class ModelReconciliationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: ReconciliationOp
    targets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cardinality(self) -> "ModelReconciliationDecision":
        if self.op in UNTARGETED_OPS and self.targets:
            raise ValueError(f"{self.op.value} requires empty targets")
        if self.op in TARGETED_OPS and len(self.targets) != 1:
            raise ValueError(f"{self.op.value} requires exactly one target")
        return self


MODEL_DECISION_SCHEMA = ModelReconciliationDecision.model_json_schema()


def validate_model_decision(raw: dict[str, Any], candidate_map: dict[str, str]) -> ModelReconciliationDecision:
    try:
        decision = ModelReconciliationDecision.model_validate(raw)
    except ValidationError as exc:
        raise ReconciliationFailure(f"invalid reconciliation output: {exc}") from exc

    invalid_targets = [target for target in decision.targets if target not in candidate_map]
    if invalid_targets:
        raise ReconciliationFailure(f"invalid candidate target(s): {invalid_targets}")
    return decision
