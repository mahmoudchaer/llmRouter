from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from routing.schemas.request import RoutingRequest
from routing.schemas.routing_result import TierRouterPrediction


class DedicatedTierRouter(ABC):
    @abstractmethod
    def predict(self, request: RoutingRequest) -> TierRouterPrediction: ...


class CallableTierRouter(DedicatedTierRouter):
    """Adapter for the selected checkpoint runtime callable, added after model selection."""
    def __init__(self, predictor: Callable[[RoutingRequest], TierRouterPrediction]): self.predictor = predictor
    def predict(self, request: RoutingRequest) -> TierRouterPrediction: return self.predictor(request)


class UnavailableTierRouter(DedicatedTierRouter):
    def predict(self, request: RoutingRequest) -> TierRouterPrediction:
        raise RuntimeError("Dedicated Tier Router checkpoint has not been configured")

