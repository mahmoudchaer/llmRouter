from __future__ import annotations

import math
import numpy as np
from .request_representation import Request

FEATURE_NAMES = [
    "log1p_total_tokens", "log1p_character_count", "log1p_message_count",
    "log1p_chunk_count", "log1p_system_tokens", "log1p_user_tokens",
    "log1p_history_tokens", "log1p_retrieved_context_tokens",
    "log1p_tool_schema_tokens", "log1p_tool_count", "structured_output_flag",
    "json_schema_requirement_flag", "modality_text_flag", "modality_image_flag",
    "modality_audio_flag",
]


def extract_structural(req: Request, total_tokens: int, chunk_count: int) -> np.ndarray:
    # Benchmark rows expose no role/history/tool metadata; zeros mean unavailable/absent.
    counts = [total_tokens, len(req.task_text) + sum(map(len, req.context_sections)),
              max(1, len(req.messages)), chunk_count, 0, total_tokens, 0, 0, 0,
              req.tool_count]
    values = [math.log1p(max(0, x)) for x in counts]
    values += [float(req.structured_output), float(req.json_schema_required),
               float("text" in req.modalities), float("image" in req.modalities),
               float("audio" in req.modalities)]
    return np.asarray(values, dtype=np.float32)

