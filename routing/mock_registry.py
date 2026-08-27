from routing.schemas.model import ModelRecord


def build_mock_registry() -> list[ModelRecord]:
    domains=("math","code","logic","knowledge","instruction_following","tool_use","affective")
    return [
        ModelRecord("mock/cheap","mock",{d:1 for d in domains},.05,.10,32768),
        ModelRecord("mock/mid","mock",{d:2 for d in domains},.20,.60,131072,True,
                    supports_structured_output=True),
        ModelRecord("mock/strong","mock",{d:3 for d in domains},.60,1.80,262144,True,
                    supports_structured_output=True),
        ModelRecord("mock/frontier","mock",{d:4 for d in domains},1.00,3.00,1048576,True,
                    input_modalities=frozenset({"text","image"}),supports_structured_output=True),
    ]

