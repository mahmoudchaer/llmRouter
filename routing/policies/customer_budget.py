from routing.schemas.model import ModelRecord
from routing.schemas.request import CustomerPriceCeiling


def within_price_ceiling(model: ModelRecord, ceiling: CustomerPriceCeiling) -> bool:
    return (model.input_price_per_1m <= ceiling.max_input_price_per_1m and
            model.output_price_per_1m <= ceiling.max_output_price_per_1m)


def ceiling_from_reference_model(model: ModelRecord) -> CustomerPriceCeiling:
    """Snapshot current prices; this intentionally does not retain a live model dependency."""
    return CustomerPriceCeiling(model.input_price_per_1m, model.output_price_per_1m)

