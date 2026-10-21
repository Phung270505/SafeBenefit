"""Authoritative method registry shared by tests and the experiment runner."""

from __future__ import annotations

from src.common.config import METHOD_IDS
from src.methods.base import PlanSelector
from src.methods.linucb import LinUCBSelector
from src.methods.safe_benefit import SafeBenefitSelector
from src.methods.static import StaticSelector
from src.methods.switch_cost_threshold import SwitchCostThresholdSelector

METHOD_ORDER = METHOD_IDS


def registered_methods() -> tuple[str, ...]:
    """Return the immutable execution order registered by the protocol."""
    return METHOD_ORDER


def create_selector(method_id: str) -> PlanSelector:
    """Construct one selector using registered default hyperparameters."""
    factories = {
        "STATIC": StaticSelector,
        "SWITCH_COST_THRESHOLD": SwitchCostThresholdSelector,
        "SAFE_BENEFIT": SafeBenefitSelector,
        "LINUCB": LinUCBSelector,
    }
    try:
        return factories[method_id]()
    except KeyError as error:
        raise ValueError(f"unregistered method: {method_id}") from error
