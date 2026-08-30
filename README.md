# Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf

This is the official repository of the EMNLP 2026 paper [Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf](https://rlg.iis.sinica.edu.tw/papers/werewolf-accusation-benchmark).

If you use this work for research, please consider citing our paper as follows:
```bibtex
@inproceedings{yang_llms_2026,
  title = {Do {{LLMs Trust}} the {{Accuser}} or the {{Accusation}}? {{Measuring Belief Shifts}} in {{Werewolf}}},
  booktitle = {Proceedings of the 2026 {{Conference}} on {{Empirical Methods}} in {{Natural Language Processing}}},
  author = {Yang, Yu-Yu and Wu, Ti-Rong and Guei, Hung and Chen, Hsing-Yu and Wu, I-Chen},
  year = 2026,
  month = oct,
  publisher = {Association for Computational Linguistics},
  address = {Budapest, Hungary}
}
```

This repository releases the belief-shift benchmark built from LLM-played Werewolf games, the responses of the 40 evaluated open-weight LLM configurations, and the code to evaluate new models and to reproduce the experiments in the paper.

The benchmark measures how an LLM that observes a Werewolf game updates its
belief about two players — the **accuser** and the **accused target** — after
one player suspects or accuses another. Evaluation is a two-stage protocol:

| Stage | Input | Output (7-level scale) |
|---|---|---|
| **Part 1 — prior belief** | game history as seen by the observer, *before* the message | `p_wolf` for accuser + target(s): `definitely_villager` (−3) … `definitely_wolf` (+3) |
| **Part 2 — belief shift** | same history + the model's own part-1 answer + the message | `shift` for accuser + target(s): `much_less_wolf` (−3) … `much_more_wolf` (+3) |

Positive values are always *toward wolf*; the observer is always on the
village team.

## Results

`outputs/` contains the three experiments of the paper recomputed from the
released responses in `results/` (they reproduce every number in the paper's
Tables 1–2 / Fig. 2 and the reported Spearman statistics):

| File | Paper | Content |
|---|---|---|
| `exp1_prior_belief.csv` | Fig. 2 | prior belief for true villagers vs. true wolves, by model size (with 95% CIs) |
| `exp2_belief_shift.csv` | Table 1 (Table 4 with CIs) | shift of target / accuser by accuser team × message strength |
| `exp3_shift_by_trust.csv` | Table 2 (Table 5 with CIs) | shift by prior trust in the accuser (Trust / Neutral / Distrust) |

Each comes with a per-model table (`exp*_per_model.csv`) and the Spearman
size trends (`exp*_size_trend.csv`). See [Compute the experiments](#compute-the-experiments).

## Repository Layout

```
werewolf-accusation-benchmark/
├── data/
│   ├── messages.csv        1,224 annotated suspicion/accusation messages (one row each)
│   ├── specs.parquet       2,422 evaluation items = message × observer perspective
│   │                       (includes the full game context each model sees)
│   └── labels.parquet      ground-truth role/team of every player a model may be asked about
├── results/<model>/        responses of the 40 evaluated configurations
│   ├── part1_prior.parquet   stage-1 answers (one row per rated player)
│   ├── part2_shift.parquet   stage-2 answers
│   └── meta.json             model size / reasoning mode (used for size grouping)
├── outputs/                exp1/exp2/exp3 CSVs recomputed from results/ (= the paper's numbers)
├── code/
│   ├── evaluate.py         evaluate one model (part 1, then part 2)
│   ├── exp1.py exp2.py exp3.py   compute the paper's three experiments from a results folder
│   ├── models.py           the 40 configurations, their sizes and the size-group definition
│   ├── prompts.py parsing.py     prompt rendering / response parsing (frozen paper protocol)
│   └── common.py           shared result loading / aggregation helpers
├── run_all_exps.sh
└── requirements.txt
```

## Setup

Clone this repository:

```bash
git clone https://github.com/rlglab/werewolf-accusation-benchmark.git
cd werewolf-accusation-benchmark
```

Create a Python 3.12 environment with [uv](https://docs.astral.sh/uv/) and
install the dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt     # pandas, pyarrow, scipy, openai
```

## Evaluate a new model

`evaluate.py` talks to any OpenAI-compatible chat endpoint that supports
JSON-schema structured output (`response_format={"type": "json_schema"}`).
`--base-url` is the endpoint's `/v1` root:

| Server | `--base-url` |
|---|---|
| local vLLM (used in the paper) | `http://localhost:8000/v1` |
| OpenAI API | `https://api.openai.com/v1` |
| any other OpenAI-compatible provider | its documented `/v1` URL |

**Local vLLM** — [vLLM](https://docs.vllm.ai/) serves any open-weight model
behind an OpenAI-compatible endpoint; see its docs for installation. Launch a
server (single-GPU example used in the paper):

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768 --reasoning-parser qwen3
```

then run the two stages:

```bash
cd code

# Part 1: prior beliefs
python evaluate.py --part 1 --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1

# Part 2: belief shifts (needs this model's part-1 file)
python evaluate.py --part 2 --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 \
    --part1 ../results/Qwen__Qwen3-8B/part1_prior.parquet
```

**Commercial API (e.g. OpenAI)** — pass the provider URL and your API key:

```bash
python evaluate.py --part 1 --model gpt-4o-mini --size-b 8 --reasoning false \
    --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
python evaluate.py --part 2 --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
```

> ⚠️ **This costs real money.** A full run sends 2,422 long game histories
> per stage (median ≈ 12.6k characters each, twice for the two stages) —
> tens of millions of input tokens per model. Estimate the price from your
> provider's rates first, and try `--limit 10` before a full run.
> (`--size-b` is only used to place the model in a size group for the
> `exp*.py` tables; for closed models whose size is unknown, pick a
> nominal value.)

* `--specs` defaults to `../data/specs.parquet`; `--out` defaults to
  `../results/<model-slug>/` (`/` in the model id becomes `__`). If `--part1`
  is omitted, part 2 looks for `part1_prior.parquet` in the output directory.
* Runs are **resumable** — rerun the same command and only pending items are sent.
* Models not listed in `models.py` need `--size-b <billions>` (and
  optionally `--reasoning true|false`) so the `exp*.py` scripts can group
  them. The values are stored in `meta.json`.
* Same checkpoint in two reasoning modes: give each run its own id with
  `--model-id`, e.g. `--model Qwen/Qwen3.5-4B --model-id Qwen/Qwen3.5-4B-nothink --enable-thinking false`.
* Generation settings used in the paper: `max_tokens 16384`, sampling
  parameters left at the server default, `--enable-thinking true/false`
  for Qwen 3.x / Gemma 4 / Ministral 3 hybrids, `--reasoning-effort
  medium` (reasoning) or `low` (non-reasoning) for GPT-OSS. Other
  provider parameters can be passed as JSON via `--extra-body`.
* `--dry-run` prints one fully rendered prompt; `--limit N` runs a subset.

## Compute the experiments

Run all three experiments at once (reads `results/`, writes the CSVs into
`outputs/`, overwriting the shipped ones):

```bash
./run_all_exps.sh
```

The script runs with the `python3` of the active environment, so activate
the uv environment from [Setup](#setup) first.

Or run the experiments one by one — each script takes a results folder
(one sub-directory per model) and an output folder:

```bash
cd code
python exp1.py --results ../results --out-dir ../outputs
python exp2.py --results ../results --out-dir ../outputs
python exp3.py --results ../results --out-dir ../outputs
```

Each script loads every `<results>/<model>/` directory that has both stage
files, keeps items whose stage-1 *and* stage-2 answers parsed, restricts to
the accuser and the accused target(s), groups models by size using
`models.py` / `meta.json`, and writes CSVs:

| Script | Paper | Main CSV | Also |
|---|---|---|---|
| `exp1.py` | Fig. 2 — prior belief for true villagers vs. true wolves | `exp1_prior_belief.csv` | `exp1_per_model.csv`, `exp1_size_trend.csv` |
| `exp2.py` | Tables 1 / 4 — shift of target / accuser by accuser team × message strength | `exp2_belief_shift.csv` | `exp2_per_model.csv`, `exp2_size_trend.csv` |
| `exp3.py` | Tables 2 / 5 — shift by prior trust in the accuser (Trust / Neutral / Distrust) | `exp3_shift_by_trust.csv` | `exp3_per_model.csv`, `exp3_size_trend.csv` |

Size groups (`models.py: SIZE_GROUPS`): `<=4B`, `5-9B`, `10-29B`, `>=30B`
(12 / 8 / 9 / 11 of the 40 configurations). Every cell is a flat mean over all
observations of the models in the group (the paper's row-level averaging),
with a 95% CI half-width (`*_ci95`) and the observation count (`*_n`). The
`*_size_trend.csv` files hold the Spearman rank correlation between each
per-model mean and model size across all models (the paper's trend tests).

## Data schema

Player names (`Alice` … `Grace`) are unique within a game and serve as the
player keys throughout; team values are `Villagers` / `Werewolves`.

### `data/messages.csv` — 1,224 rows, one per annotated message

| Column | Description |
|---|---|
| `annotation_id` | message key; links to `specs.parquet` |
| `game_id`, `day` | which game / in-game day the message occurred |
| `strength` | `suspect` (soft suspicion) or `accuse` (direct accusation) |
| `accuser_player_name`, `accuser_role_truth`, `accuser_team_truth`, `accuser_model` | the speaker: in-game name, true role, true team, and the LLM that played the seat |
| `named_target_player_names`, `target_role_truth`, `target_team_truth`, `target_model` | the accused player(s); lists aligned by position (most messages name one target) |
| `highlight_text` | the annotated accusation span |
| `utterance` | the full chat message |
| `reason` | the annotator's rationale for the label |
| `winner_team` | final outcome of that game |

### `data/specs.parquet` — 2,422 rows, one per evaluation item

Each message is paired with village-side observers who were alive at that
moment: two per message (26 messages have only one eligible observer). All
statistics in the paper are computed over these 2,422 items.

| Column | Description |
|---|---|
| `spec_id` | item key; links to `labels.parquet` and the result files |
| `annotation_id`, `game_id`, `day` | the message this item comes from |
| `observer_player_name`, `observer_role_truth`, `observer_team_truth` | the observer whose perspective the model takes (always village-side: Villager / Seer / Doctor) |
| `accuser_player_name`, `accuser_role_truth`, `accuser_team_truth` | the speaker |
| `named_target_player_names` | the accused player(s) |
| `alive_player_names` | players alive at the message |
| `strength` | `suspect` / `accuse` |
| `prior_context` | the full game history rendered from the observer's point of view, strictly *before* the message (what the model reads in part 1) |
| `utterance` | the message itself (shown as the trigger event in part 2) |
| `winner_team` | final outcome of that game |

The model is asked about the accuser and the named target(s), excluding the
observer itself.

### `data/labels.parquet` — 11,222 rows, one per (item × assessable player)

| Column | Description |
|---|---|
| `spec_id` | item key |
| `subject_player_name` | a player the model may be asked about (alive at the message, not the observer) |
| `subject_role_truth`, `subject_team_truth` | that player's true role and team |

### `results/<model>/` — one directory per evaluated configuration

`part1_prior.parquet` and `part2_shift.parquet` have 4,656 rows each: one
per (item × rated player). The rows of one item come from a single API call,
so `raw_response` repeats within an item.

| Column | Description |
|---|---|
| `spec_id`, `subject_player_name` | the item and the rated player |
| `p_wolf_label`, `prior_belief` | part 1 answer: 7-level label and its value on −3..+3 |
| `shift_label`, `belief_shift` | part 2 answer (replaces the two columns above) |
| `parse_status` | `ok`, or `parse_fail` / `missing_subject` / `invalid_enum` |
| `raw_response` | the model's raw JSON reply |

`meta.json` carries `model_id`, `family`, `size_b`, `reasoning`, and
`size_group` (used by the `exp*.py` scripts); `results/coverage.csv` lists
per-model row and parse-ok counts.
