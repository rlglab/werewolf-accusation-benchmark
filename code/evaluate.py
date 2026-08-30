"""Evaluate one model on the belief-shift benchmark (two stages).

Stage 1 — prior belief (before the accusation message)::

    python evaluate.py --part 1 --model Qwen/Qwen3-8B \
        --base-url http://localhost:8000/v1

Stage 2 — belief shift (after the message; needs the stage-1 result)::

    python evaluate.py --part 2 --model Qwen/Qwen3-8B \
        --base-url http://localhost:8000/v1 \
        --part1 ../results/Qwen__Qwen3-8B/part1_prior.parquet

The server must expose an OpenAI-compatible ``/v1/chat/completions`` endpoint
with JSON-schema structured output (vLLM does). Everything the model sees is
in ``data/specs.parquet`` (default ``--specs``); results are written to
``results/<model-slug>/part{1,2}_*.parquet`` (default ``--out``) together with
``meta.json`` (model size / reasoning flag used by the ``exp*.py`` scripts).

Runs are resumable: specs already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from models import lookup, model_slug, size_group
from parsing import OK, PARSE_FAIL, parse_part1, parse_part2
from prompts import render_part1, render_part2

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent

PART1_FILE = "part1_prior.parquet"
PART2_FILE = "part2_shift.parquet"
PART1_COLS = ["spec_id", "subject_player_name", "p_wolf_label", "prior_belief",
              "parse_status", "raw_response"]
PART2_COLS = ["spec_id", "subject_player_name", "shift_label", "belief_shift",
              "parse_status", "raw_response"]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

class Client:
    def __init__(self, args: argparse.Namespace) -> None:
        self.client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key,
                                  timeout=args.timeout, max_retries=0)
        self.sem = asyncio.Semaphore(args.concurrency)
        self.max_retries = args.max_retries
        self.chat_params: dict[str, Any] = {"model": args.model, "max_tokens": args.max_tokens}
        if args.temperature is not None:
            self.chat_params["temperature"] = args.temperature
        if args.reasoning_effort:
            self.chat_params["reasoning_effort"] = args.reasoning_effort
        extra: dict[str, Any] = json.loads(args.extra_body) if args.extra_body else {}
        if args.enable_thinking is not None:
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = args.enable_thinking
        if extra:
            self.chat_params["extra_body"] = extra

    async def generate(self, prompt: str, schema: dict) -> tuple[str, str | None, int]:
        """Return (content, reasoning text if the server exposes one, latency in ms)."""
        payload = {
            **self.chat_params,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "action", "strict": True, "schema": schema},
            },
        }
        async with self.sem:
            t_start = time.time()
            last: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    resp = await self.client.chat.completions.create(**payload)
                    if not resp.choices:
                        raise RuntimeError(f"response has no choices: {getattr(resp, 'error', None)}")
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep((2 ** attempt) * 0.2 + random.random() * 0.1)
            else:  # pragma: no cover
                raise last  # type: ignore[misc]
            latency = int((time.time() - t_start) * 1000)
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = None
        for key in ("reasoning", "reasoning_content", "reasoning_text"):
            val = getattr(msg, key, None)
            if val is None and hasattr(msg, "model_extra") and msg.model_extra:
                val = msg.model_extra.get(key)
            if val:
                reasoning = str(val)
                break
        return content, reasoning, latency


# ---------------------------------------------------------------------------
# Per-spec work
# ---------------------------------------------------------------------------

def _rows(spec_id: str, names: list[str], results, *, part: int, raw: str | None) -> list[dict]:
    label_key, value_key = ("p_wolf_label", "prior_belief") if part == 1 else ("shift_label", "belief_shift")
    return [{
        "spec_id": spec_id, "subject_player_name": pname,
        label_key: r.label, value_key: r.value,
        "parse_status": r.parse_status, "raw_response": raw,
    } for pname, r in zip(names, results)]


async def run_spec(client: Client, spec: dict, part: int,
                   prior_beliefs: list[dict] | None) -> list[dict]:
    if part == 1:
        rp = render_part1(spec)
    else:
        rp = render_part2(spec, prior_beliefs or [])
    try:
        raw, _reasoning, _latency = await client.generate(rp.prompt_text, rp.schema)
    except Exception as exc:  # noqa: BLE001
        print(f"[part{part}] {spec['spec_id']}: call failed: {exc}", file=sys.stderr)
        fake = [type("R", (), {"label": None, "value": None, "parse_status": PARSE_FAIL})()] * len(rp.subject_player_names)
        return _rows(spec["spec_id"], rp.subject_player_names, fake, part=part, raw=None)
    parsed = (parse_part1 if part == 1 else parse_part2)(raw, rp.subject_player_names)
    return _rows(spec["spec_id"], rp.subject_player_names, parsed, part=part, raw=raw)


def load_prior_beliefs(part1_path: Path) -> dict[str, list[dict]]:
    """spec_id -> [{"player", "p_wolf"}, ...] for specs whose stage-1 parse is fully ok."""
    df = pd.read_parquet(part1_path)
    out: dict[str, list[dict]] = {}
    for sid, g in df.groupby("spec_id"):
        if (g["parse_status"] == OK).all():
            out[sid] = [{"player": r.subject_player_name, "p_wolf": r.p_wolf_label}
                        for r in g.itertuples(index=False)]
    return out


def _spec_records(specs: pd.DataFrame) -> list[dict]:
    recs = specs.to_dict("records")
    for r in recs:
        for c in ("named_target_player_names", "alive_player_names"):
            r[c] = list(r[c])
    return recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    specs = pd.read_parquet(args.specs)
    if args.limit:
        specs = specs.head(args.limit)
    out_dir = Path(args.out) if args.out else RELEASE / "results" / model_slug(args.model_id)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (PART1_FILE if args.part == 1 else PART2_FILE)
    cols = PART1_COLS if args.part == 1 else PART2_COLS

    # meta.json — model size / reasoning flag, consumed by common.py (exp*.py)
    info = lookup(args.model_id)
    prev = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
    size_b = args.size_b if args.size_b is not None else prev.get("size_b", info.size_b if info else None)
    reasoning = (args.reasoning if args.reasoning is not None
                 else prev.get("reasoning", info.reasoning if info else None))
    if size_b is None:
        sys.exit("unknown model: pass --size-b (billions of parameters) so it can be grouped")
    meta = {
        "model_id": args.model_id, "family": (info.family if info else args.model_id.split("/")[0]),
        "size_b": size_b, "reasoning": bool(reasoning), "size_group": size_group(size_b),
        "served_model": args.model, "base_url": args.base_url, "chat_params": {
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort, "enable_thinking": args.enable_thinking,
            "extra_body": args.extra_body},
    }
    if not args.dry_run:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    prior_beliefs: dict[str, list[dict]] = {}
    if args.part == 2:
        if not args.part1:
            cand = out_dir / PART1_FILE
            if not cand.exists():
                sys.exit("--part 2 needs --part1 <part1_prior.parquet> (stage-1 result of this model)")
            args.part1 = cand
        prior_beliefs = load_prior_beliefs(Path(args.part1))
        missing = [s for s in specs.spec_id if s not in prior_beliefs]
        if missing:
            print(f"[part2] {len(missing)} specs have no usable stage-1 belief and will be skipped")
        specs = specs[specs.spec_id.isin(prior_beliefs)]

    done: pd.DataFrame | None = None
    if out_file.exists() and not args.overwrite:
        done = pd.read_parquet(out_file)
        specs = specs[~specs.spec_id.isin(set(done.spec_id))]
        print(f"resume: {len(done)} rows already in {out_file.name}; {len(specs)} specs pending")
    if specs.empty:
        print("nothing to do")
        return
    if args.dry_run:
        rp = render_part1(_spec_records(specs.head(1))[0]) if args.part == 1 else \
            render_part2(_spec_records(specs.head(1))[0], prior_beliefs[specs.iloc[0].spec_id])
        print(rp.prompt_text)
        print(f"\n[dry-run] {len(specs)} specs pending; example prompt above "
              f"({len(rp.prompt_text)} chars, subjects={rp.subject_player_names})")
        return

    client = Client(args)
    recs = _spec_records(specs)
    buffer: list[dict] = []
    n_ok = n_fail = 0
    t_start = time.time()

    def flush() -> None:
        nonlocal done, buffer
        if not buffer:
            return
        new = pd.DataFrame(buffer, columns=cols)
        done = new if done is None else pd.concat([done, new], ignore_index=True)
        done.to_parquet(out_file, index=False)
        buffer = []

    tasks = [run_spec(client, r, args.part, prior_beliefs.get(r["spec_id"])) for r in recs]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        rows = await fut
        buffer.extend(rows)
        if all(r["parse_status"] == OK for r in rows):
            n_ok += 1
        else:
            n_fail += 1
        if i % args.flush_every == 0 or i == len(tasks):
            flush()
            el = time.time() - t_start
            print(f"[part{args.part}] {i}/{len(tasks)} specs  ok={n_ok} failed={n_fail}  "
                  f"{el/60:.1f} min elapsed, ~{el/i*(len(tasks)-i)/60:.1f} min left", flush=True)
    flush()
    print(f"wrote {out_file}  ({len(done)} rows)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", type=int, choices=(1, 2), required=True,
                   help="1 = prior belief, 2 = belief shift")
    p.add_argument("--model", required=True, help="model name sent to the API (served model name)")
    p.add_argument("--model-id", default=None,
                   help="identifier stored in the results (default: --model); use a distinct id, "
                        "e.g. 'Qwen/Qwen3.5-4B-nothink', when the same checkpoint is run in two modes")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--specs", default=str(RELEASE / "data/specs.parquet"))
    p.add_argument("--part1", default=None, help="(part 2) path to this model's part1_prior.parquet")
    p.add_argument("--out", default=None, help="output dir (default: results/<model-slug>/)")
    # model metadata (needed for grouping when the model is not in models.py)
    p.add_argument("--size-b", type=float, default=None, help="parameter count in billions")
    p.add_argument("--reasoning", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="whether chain-of-thought reasoning is engaged in this run (true/false)")
    # generation params
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=None, help="unset = server default")
    p.add_argument("--reasoning-effort", default=None, help="e.g. low/medium/high (gpt-oss)")
    p.add_argument("--enable-thinking", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="sets chat_template_kwargs.enable_thinking (Qwen3 / Gemma 4 / Ministral 3)")
    p.add_argument("--extra-body", default=None, help="JSON merged into the request's extra_body")
    # runtime
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=480.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="only the first N specs (smoke test)")
    p.add_argument("--overwrite", action="store_true", help="ignore an existing output file")
    p.add_argument("--dry-run", action="store_true", help="print one rendered prompt and exit")
    args = p.parse_args(argv)
    args.model_id = args.model_id or args.model
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
