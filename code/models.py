"""Model registry for the 40 evaluated (checkpoint, reasoning-mode) configurations.

``MODELS`` is the single source of truth for

* which ``model_id`` values belong to the paper's 40-configuration set,
* each configuration's parameter count (``size_b``) and reasoning mode,
* how models are grouped into the four size groups used by every table
  and figure in the paper (``size_group``).

``model_id`` is the identifier stored in every result file. For checkpoints
that were run in both reasoning modes, the non-reasoning run carries a
suffix so the two runs never collide:

* Qwen 3.5 / 3.6 ................. ``-nothink``   (enable_thinking=false)
* GPT-OSS ......................... ``-nr``        (reasoning_effort=low)
* Gemma 4 reasoning run ........... ``-r``         (enable_thinking=true)
* GPT-OSS-20B reasoning run ....... ``-r``         (reasoning_effort=medium)

New models can be added here or given on the command line
(``evaluate.py --size-b``); the experiment scripts fall back to the
``meta.json`` written next to each result file.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Size groups (paper Section 4): label, lower bound, upper bound (inclusive).
# ---------------------------------------------------------------------------
SIZE_GROUPS: list[tuple[str, float, float]] = [
    ("<=4B", 0.0, 4.0),
    ("5-9B", 4.0, 9.0),
    ("10-29B", 9.0, 29.0),
    (">=30B", 29.0, float("inf")),
]
SIZE_GROUP_LABELS = [g[0] for g in SIZE_GROUPS]


def size_group(size_b: float) -> str:
    """Map a parameter count (in billions) to its paper size group."""
    for label, lo, hi in SIZE_GROUPS:
        if lo < size_b <= hi:
            return label
    raise ValueError(f"size_b out of range: {size_b}")


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    family: str
    size_b: float
    reasoning: bool  # True = chain-of-thought reasoning engaged in this run

    @property
    def size_group(self) -> str:
        return size_group(self.size_b)

    @property
    def slug(self) -> str:
        """Filesystem-safe name used for the per-model result directory."""
        return model_slug(self.model_id)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


# ---------------------------------------------------------------------------
# The 40 configurations (Appendix A, Table "models_list").
# ---------------------------------------------------------------------------
_M = ModelInfo
MODELS: list[ModelInfo] = [
    # GLM
    _M("zai-org/GLM-4.7-Flash", "GLM", 30, True),
    # GPT-OSS
    _M("openai/gpt-oss-20b-nr", "GPT-OSS", 20, False),
    _M("openai/gpt-oss-20b-r", "GPT-OSS", 20, True),
    _M("openai/gpt-oss-120b-nr", "GPT-OSS", 120, False),
    _M("openai/gpt-oss-120b", "GPT-OSS", 120, True),
    # Gemma 4
    _M("google/gemma-4-E2B-it", "Gemma", 2, False),
    _M("google/gemma-4-E2B-it-r", "Gemma", 2, True),
    _M("google/gemma-4-E4B-it", "Gemma", 4, False),
    _M("google/gemma-4-E4B-it-r", "Gemma", 4, True),
    _M("google/gemma-4-26B-A4B-it", "Gemma", 26, False),
    _M("google/gemma-4-26B-A4B-it-r", "Gemma", 26, True),
    _M("google/gemma-4-31B-it", "Gemma", 31, False),
    _M("google/gemma-4-31B-it-r", "Gemma", 31, True),
    # Llama
    _M("meta-llama/Llama-3.2-1B-Instruct", "Llama", 1, False),
    _M("meta-llama/Llama-3.2-3B-Instruct", "Llama", 3, False),
    _M("meta-llama/Llama-3.1-8B-Instruct", "Llama", 8, False),
    _M("meta-llama/Llama-3.3-70B-Instruct", "Llama", 70, False),
    # Mistral
    _M("mistralai/Ministral-3-3B-Instruct-2512", "Mistral", 3, False),
    _M("mistralai/Ministral-3-8B-Instruct-2512", "Mistral", 8, False),
    _M("mistralai/Ministral-3-14B-Instruct-2512", "Mistral", 14, False),
    # Nemotron
    _M("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "Nemotron", 9, True),
    # Olmo
    _M("allenai/Olmo-3-7B-Instruct", "Olmo", 7, False),
    _M("allenai/Olmo-3-7B-Think", "Olmo", 7, True),
    _M("allenai/Olmo-3.1-32B-Instruct", "Olmo", 32, False),
    _M("allenai/Olmo-3.1-32B-Think", "Olmo", 32, True),
    # Phi
    _M("microsoft/Phi-4-mini-instruct", "Phi", 3.8, False),
    _M("microsoft/phi-4", "Phi", 14, False),
    # Qwen
    _M("Qwen/Qwen3.5-2B", "Qwen", 2, True),
    _M("Qwen/Qwen3.5-2B-nothink", "Qwen", 2, False),
    _M("Qwen/Qwen3.5-4B", "Qwen", 4, True),
    _M("Qwen/Qwen3.5-4B-nothink", "Qwen", 4, False),
    _M("Qwen/Qwen3-8B", "Qwen", 8, True),
    _M("Qwen/Qwen3.5-9B", "Qwen", 9, True),
    _M("Qwen/Qwen3.5-9B-nothink", "Qwen", 9, False),
    _M("Qwen/Qwen3-14B", "Qwen", 14, True),
    _M("Qwen/Qwen3.6-27B", "Qwen", 27, True),
    _M("Qwen/Qwen3.6-27B-nothink", "Qwen", 27, False),
    _M("Qwen/Qwen3-30B-A3B", "Qwen", 30, True),
    _M("Qwen/Qwen3.6-35B-A3B", "Qwen", 35, True),
    _M("Qwen/Qwen3.6-35B-A3B-nothink", "Qwen", 35, False),
]
assert len(MODELS) == 40, len(MODELS)

BY_ID: dict[str, ModelInfo] = {m.model_id: m for m in MODELS}
BY_SLUG: dict[str, ModelInfo] = {m.slug: m for m in MODELS}


def lookup(model_id: str) -> ModelInfo | None:
    return BY_ID.get(model_id)

