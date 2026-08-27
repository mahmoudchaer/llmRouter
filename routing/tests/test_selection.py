import pytest
from routing.mock_registry import build_mock_registry
from routing.policies.customer_budget import ceiling_from_reference_model
from routing.schemas.model import ModelRecord
from routing.schemas.request import CustomerPriceCeiling,HardRequirements,RoutingRequest
from routing.selection.compatibility import compatibility_reasons
from routing.selection.model_selector import ModelSelector,NoSuitableModelError


def request(ceiling=CustomerPriceCeiling(1,3),requirements=HardRequirements(context_tokens=1000)):
    return RoutingRequest("r","task",ceiling,requirements,expected_output_tokens=500)


def test_selects_cheapest_capable_compatible_model():
    result=ModelSelector().select(build_mock_registry(),request(),"code",3)
    assert result.model.model_id=="mock/strong"


def test_price_ceiling_is_hard_and_never_exceeded():
    with pytest.raises(NoSuitableModelError):
        ModelSelector().select(build_mock_registry(),request(CustomerPriceCeiling(.21,.61)),"code",3)


def test_reference_ceiling_is_price_snapshot():
    model=build_mock_registry()[1];ceiling=ceiling_from_reference_model(model)
    assert ceiling.max_input_price_per_1m==.2 and ceiling.max_output_price_per_1m==.6


def test_all_hard_filters_are_enforced():
    model=build_mock_registry()[1]
    req=request(requirements=HardRequirements(context_tokens=200000,requires_tools=True,
        input_modalities=frozenset({"image"}),requires_structured_output=True,
        provider_blocklist=frozenset({"mock"}),model_blocklist=frozenset({model.model_id})))
    reasons=compatibility_reasons(model,req)
    assert {"context_window","input_modalities","provider_blocklist","model_blocklist"}<=set(reasons)

