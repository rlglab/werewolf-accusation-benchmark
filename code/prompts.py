"""Prompt rendering for the two evaluation stages.

This is a faithful port of the prompt renderer used for the paper. The
rendered text is byte-identical to the prompts stored in the original run
traces, so new models can be evaluated under exactly the same protocol.

Stage 1 (``part1``):  game context  -> 7-level P(wolf) for accuser + targets
Stage 2 (``part2``):  game context + stage-1 belief + trigger message
                           -> 7-level belief SHIFT for accuser + targets
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# 7-level scales ------------------------------------------------------------
P_WOLF_LABELS: tuple[str, ...] = (
    "definitely_villager",  # -3
    "likely_villager",      # -2
    "lean_villager",        # -1
    "unsure",               #  0
    "lean_wolf",            # +1
    "likely_wolf",          # +2
    "definitely_wolf",      # +3
)
# Prior belief on the paper's -3..+3 wolf-positive scale.
P_WOLF_TO_INT: dict[str, int] = {lbl: i - 3 for i, lbl in enumerate(P_WOLF_LABELS)}

SHIFT_LABELS: tuple[str, ...] = (
    "much_less_wolf",       # -3
    "less_wolf",            # -2
    "slightly_less_wolf",   # -1
    "same",                 #  0
    "slightly_more_wolf",   # +1
    "more_wolf",            # +2
    "much_more_wolf",       # +3
)
SHIFT_TO_INT: dict[str, int] = {lbl: i - 3 for i, lbl in enumerate(SHIFT_LABELS)}


@dataclass(slots=True)
class RenderedPrompt:
    prompt_text: str
    schema: dict
    subject_player_names: list[str]


# ---------------------------------------------------------------------------
# Subjects: the accuser first, then the named targets (alive, not the observer)
# ---------------------------------------------------------------------------

def resolve_subjects(spec: Mapping[str, Any]) -> list[str]:
    """Players the model is asked about: the accuser first, then the named
    target(s) — restricted to players alive at the message and excluding the
    observer itself. Player names are unique within a game."""
    accuser = str(spec["accuser_player_name"])
    observer = str(spec["observer_player_name"])
    targets = [str(t) for t in spec["named_target_player_names"]]
    universe = {str(a) for a in spec["alive_player_names"]} - {observer}

    ordered: list[str] = []
    if accuser in universe:
        ordered.append(accuser)
    for t in targets:
        if t in universe and t not in ordered:
            ordered.append(t)
    return ordered


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _trigger_event_block(speaker_name: str, utterance: str) -> str:
    text = (utterance or "").strip()
    return (
        "### Trigger Event\n"
        "The following event occurred AFTER your t0 assessment (treat it as "
        "the next event after the Event History above) and is the focal "
        "accusation in this experiment:\n\n"
        f"  {speaker_name} said: \"{text}\""
    )


def _prior_belief_block(prior_beliefs: Sequence[Mapping[str, Any]], subject_names: Sequence[str]) -> str:
    by_name = {b["player"]: b for b in prior_beliefs}
    n = len(subject_names)
    lines = [
        "### Your t0 Belief",
        f"Just before the trigger event below, you assessed these {n} players. Your assessment was:",
        "",
    ]
    for name in subject_names:
        b = by_name.get(name)
        if b is None:
            lines.append(f"- {name}: p_wolf=?")
            continue
        label = b.get("p_wolf_label") or b.get("p_wolf") or "?"
        evidence = (b.get("evidence") or "").strip()
        lines.append(f"- {name}: p_wolf={label}")
        if evidence:
            lines.append(f"  Reasoning: \"{evidence}\"")
    return "\n".join(lines)


def _build_exemplar(schema: dict, *, key: str | None = None, index: int = 0) -> Any:
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        values = schema["enum"]
        return values[index % len(values)] if values else None
    t = schema.get("type")
    if t == "object":
        return {k: _build_exemplar(p, key=k, index=index) for k, p in schema.get("properties", {}).items()}
    if t == "array":
        items = schema.get("items", {})
        return [_build_exemplar(items, key=key, index=i) for i in range(schema.get("minItems", 1))]
    if t == "string":
        return f"<{key}>" if key else "<string>"
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    return None


def _schema_and_example_block(schema: dict) -> str:
    exemplar = _build_exemplar(schema)
    return (
        "#### JSON SCHEMA\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        "#### EXAMPLE OUTPUT\n"
        f"```json\n{json.dumps(exemplar, indent=2)}\n```"
    )


def _task_block_prior(subject_names: Sequence[str], schema: dict) -> str:
    labels_str = ", ".join(f'"{lbl}"' for lbl in P_WOLF_LABELS)
    enum_str = ", ".join(f'"{p}"' for p in subject_names)
    order_csv = ", then ".join(subject_names)
    n = len(subject_names)
    framing = (
        "You are mid-game. Assess the following "
        f"{n} players' likelihood of being a Werewolf based on what "
        "you have observed so far."
    )
    per_item_keys = [
        "- player: exact name match — must be one of " + enum_str,
        f"- p_wolf: ONE of (use exact text): {labels_str}",
    ]
    constraints = [
        f"- Output exactly {n} belief entries, in the order: {order_csv}.",
        "- Each player must appear exactly once.",
        "- p_wolf must be one of the seven listed labels (exact text).",
        "- Output JSON ONLY — no markdown fences, no prose outside the JSON.",
        "- Your response MUST be a single, valid JSON object.",
    ]
    return (
        "#### TASK\n"
        f"{framing}\n\n"
        "For EACH player listed below (and ONLY these), output:\n"
        + "\n".join(per_item_keys)
        + "\n\n"
        f"Players to assess (in this order): {', '.join(subject_names)}\n\n"
        "#### CONSTRAINTS\n"
        + "\n".join(constraints)
        + "\n\n"
        + _schema_and_example_block(schema)
    )


def _task_block_shift(subject_names: Sequence[str], schema: dict) -> str:
    labels_str = ", ".join(f'"{lbl}"' for lbl in SHIFT_LABELS)
    enum_str = ", ".join(f'"{p}"' for p in subject_names)
    order_csv = ", then ".join(subject_names)
    n = len(subject_names)
    per_item_keys = [
        "- player: exact name match — must be one of " + enum_str,
        f"- shift: ONE of (use exact text): {labels_str}",
    ]
    constraints = [
        f"- Output exactly {n} shift entries, in the order: {order_csv}.",
        "- Each player must appear exactly once.",
        "- shift must be one of the seven listed labels (exact text).",
        "- \"same\" is the correct answer if no post-t0 event meaningfully updated your belief.",
        "- Output JSON ONLY — no markdown fences, no prose outside the JSON.",
        "- Your response MUST be a single, valid JSON object.",
    ]
    return (
        "#### TASK\n"
        f"For each of the {n} players below (and ONLY these), report how "
        "your belief about them has SHIFTED relative to your t0 "
        "assessment above. Base your shift on:\n"
        "1. The trigger event quoted above, and\n"
        "2. Any events that occurred AFTER your t0 assessment (visible "
        "in Event History).\n\n"
        "For each player, output:\n"
        + "\n".join(per_item_keys)
        + "\n\n"
        f"Players to assess (in this order): {', '.join(subject_names)}\n\n"
        "#### CONSTRAINTS\n"
        + "\n".join(constraints)
        + "\n\n"
        + _schema_and_example_block(schema)
    )


# ---------------------------------------------------------------------------
# JSON schemas (also sent to the server as structured-output constraints)
# ---------------------------------------------------------------------------

def schema_prior(subject_names: Sequence[str]) -> dict:
    n = len(subject_names)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["beliefs"],
        "properties": {
            "reasoning": {"type": "string"},
            "beliefs": {
                "type": "array", "minItems": n, "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["player", "p_wolf"],
                    "properties": {
                        "player": {"type": "string", "enum": list(subject_names)},
                        "p_wolf": {"type": "string", "enum": list(P_WOLF_LABELS)},
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    }


def schema_shift(subject_names: Sequence[str]) -> dict:
    n = len(subject_names)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["shifts"],
        "properties": {
            "reasoning": {"type": "string"},
            "shifts": {
                "type": "array", "minItems": n, "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["player", "shift"],
                    "properties": {
                        "player": {"type": "string", "enum": list(subject_names)},
                        "shift": {"type": "string", "enum": list(SHIFT_LABELS)},
                        "shift_reason": {"type": "string"},
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def render_part1(spec: Mapping[str, Any]) -> RenderedPrompt:
    """Stage 1: prior belief about accuser + targets, before the message."""
    names = resolve_subjects(spec)
    if not names:
        raise ValueError(f"spec {spec.get('spec_id')}: no subjects to assess")
    schema = schema_prior(names)
    prompt = "\n\n".join([spec["prior_context"] or "", _task_block_prior(names, schema)])
    return RenderedPrompt(prompt, schema, names)


def render_part2(spec: Mapping[str, Any], prior_beliefs: Sequence[Mapping[str, Any]]) -> RenderedPrompt:
    """Stage 2: belief shift after the message, given the stage-1 belief.

    ``prior_beliefs`` is a list of ``{"player": <name>, "p_wolf": <label>}``
    from this model's own stage-1 answer for the same spec.
    """
    names = resolve_subjects(spec)
    if not names:
        raise ValueError(f"spec {spec.get('spec_id')}: no subjects to assess")
    schema = schema_shift(names)
    prompt = "\n\n".join([
        spec["prior_context"] or "",
        _prior_belief_block(prior_beliefs, names),
        _trigger_event_block(str(spec["accuser_player_name"]), spec["utterance"]),
        _task_block_shift(names, schema),
    ])
    return RenderedPrompt(prompt, schema, names)


__all__ = [
    "P_WOLF_LABELS", "P_WOLF_TO_INT", "SHIFT_LABELS", "SHIFT_TO_INT",
    "RenderedPrompt", "resolve_subjects", "render_part1", "render_part2",
    "schema_prior", "schema_shift",
]
