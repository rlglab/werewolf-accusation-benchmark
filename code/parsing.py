"""Parse model responses for both stages.

Stage 1 response:  {"beliefs": [{"player": ..., "p_wolf": <label>}, ...]}
Stage 2 response:  {"shifts":  [{"player": ..., "shift":  <label>}, ...]}

Decoding is mildly tolerant (trailing padding after the JSON value, a
missing final ``}`` after a closed array, and a flat ``{player: label}``
dict) — identical to the parser used for the paper's runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from prompts import P_WOLF_LABELS, P_WOLF_TO_INT, SHIFT_LABELS, SHIFT_TO_INT

OK = "ok"
PARSE_FAIL = "parse_fail"
MISSING_SUBJECT = "missing_subject"
INVALID_ENUM = "invalid_enum"


@dataclass(slots=True)
class SubjectResult:
    player_name: str
    label: str | None
    value: int | None          # -3..+3
    parse_status: str
    parse_error: str | None


def parse_part1(raw: str, subject_names: Sequence[str]) -> list[SubjectResult]:
    return _parse(raw, subject_names, array_key="beliefs", label_key="p_wolf",
                  labels=P_WOLF_LABELS, to_int=P_WOLF_TO_INT)


def parse_part2(raw: str, subject_names: Sequence[str]) -> list[SubjectResult]:
    return _parse(raw, subject_names, array_key="shifts", label_key="shift",
                  labels=SHIFT_LABELS, to_int=SHIFT_TO_INT)


def _decode_lenient(raw: str) -> tuple[Any, str | None]:
    decoder = json.JSONDecoder()
    s = raw.lstrip() if isinstance(raw, str) else ""
    try:
        obj, _ = decoder.raw_decode(s)
        return obj, None
    except json.JSONDecodeError as exc:
        first_err = str(exc)
    s2 = s.rstrip()
    if s2.endswith("]"):
        try:
            obj, _ = decoder.raw_decode(s2 + "}")
            return obj, None
        except json.JSONDecodeError:
            pass
    return None, first_err


def _coerce_flat(obj: Any, *, array_key: str, label_key: str, labels: tuple[str, ...]) -> Any:
    if not isinstance(obj, dict) or array_key in obj or not obj:
        return obj
    if not all(isinstance(k, str) for k in obj):
        return obj
    if not all(isinstance(v, str) and v in labels for v in obj.values()):
        return obj
    return {array_key: [{"player": k, label_key: v} for k, v in obj.items()]}


def _all(names: Sequence[str], status: str, error: str) -> list[SubjectResult]:
    return [SubjectResult(n, None, None, status, error) for n in names]


def _parse(raw: str, subject_names: Sequence[str], *, array_key: str, label_key: str,
           labels: tuple[str, ...], to_int: dict[str, int]) -> list[SubjectResult]:
    requested = list(subject_names)
    obj, err = _decode_lenient(raw)
    if err is not None:
        return _all(requested, PARSE_FAIL, f"json decode failed: {err}")
    if not isinstance(obj, dict):
        return _all(requested, PARSE_FAIL, f"top-level not object: got {type(obj).__name__}")
    arr = obj.get(array_key)
    if not isinstance(arr, list):
        obj = _coerce_flat(obj, array_key=array_key, label_key=label_key, labels=labels)
        arr = obj.get(array_key)
    if not isinstance(arr, list):
        return _all(requested, PARSE_FAIL, f"missing or non-list '{array_key}'")

    by_name: dict[str, dict] = {}
    for item in arr:
        if isinstance(item, dict) and isinstance(item.get("player"), str) and item["player"] in requested:
            by_name.setdefault(item["player"], item)

    out: list[SubjectResult] = []
    for name in requested:
        item = by_name.get(name)
        if item is None:
            out.append(SubjectResult(name, None, None, MISSING_SUBJECT, "not present in response"))
            continue
        label = item.get(label_key)
        if not isinstance(label, str) or label not in to_int:
            out.append(SubjectResult(name, None, None, INVALID_ENUM, f"unknown {label_key}={label!r}"))
            continue
        out.append(SubjectResult(name, label, to_int[label], OK, None))
    return out


__all__ = ["SubjectResult", "parse_part1", "parse_part2", "OK", "PARSE_FAIL",
           "MISSING_SUBJECT", "INVALID_ENUM"]
