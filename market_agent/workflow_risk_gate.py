from __future__ import annotations

from dataclasses import dataclass

from market_agent.workflow_contracts import (
    Action,
    DecisionDraft,
    KnowledgeStatus,
    RiskAssessment,
    TechnicalAnalysis,
)


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    min_confidence: float = 0.60
    maximum_stop_distance_fraction: float = 0.20
    require_technical_confirmation: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if not 0.0 < self.maximum_stop_distance_fraction < 1.0:
            raise ValueError("maximum_stop_distance_fraction must be between zero and one")


def _rejected(*codes: str, insufficient: bool = False) -> RiskAssessment:
    return RiskAssessment(
        knowledge_status=KnowledgeStatus.INSUFFICIENT if insufficient else KnowledgeStatus.KNOWN,
        uncertainty_reason="risk gate lacks sufficient evidence" if insufficient else None,
        accepted=False,
        reason_codes=tuple(codes),
        requires_escalation=False,
    )


def evaluate_risk(
    decision: DecisionDraft,
    *,
    technical: TechnicalAnalysis | None,
    policy: RiskPolicy = RiskPolicy(),
) -> RiskAssessment:
    if decision.knowledge_status is KnowledgeStatus.INSUFFICIENT:
        return _rejected("insufficient_decision_evidence", insufficient=True)
    if decision.action is Action.NO_TRADE:
        return _rejected("no_trade_decision")
    if decision.decision_confidence < policy.min_confidence:
        return _rejected("confidence_below_policy")
    if decision.entry_price is None or decision.stop_price is None:
        return _rejected("missing_trade_levels", insufficient=True)
    if decision.action is Action.LONG and decision.stop_price >= decision.entry_price:
        return _rejected("long_stop_not_below_entry")
    if decision.action is Action.SHORT and decision.stop_price <= decision.entry_price:
        return _rejected("short_stop_not_above_entry")
    distance = abs(decision.entry_price - decision.stop_price) / decision.entry_price
    if distance > policy.maximum_stop_distance_fraction:
        return _rejected("stop_distance_exceeds_policy")
    if policy.require_technical_confirmation:
        if technical is None or technical.knowledge_status is KnowledgeStatus.INSUFFICIENT:
            return _rejected("technical_confirmation_missing", insufficient=True)
        setup = technical.long_setup if decision.action is Action.LONG else technical.short_setup
        if not setup.viable:
            return _rejected("technical_setup_not_viable")
    return RiskAssessment(
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        accepted=True,
        reason_codes=("risk_policy_satisfied",),
        requires_escalation=decision.decision_confidence < min(0.85, policy.min_confidence + 0.20),
    )
