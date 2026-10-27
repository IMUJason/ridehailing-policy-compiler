# ridehailing-policy-compiler

Code and de-identified data for compiling natural-language shared ride-hailing policy
statements into simulation configurations, and for testing whether a compiled policy
selects the same rollout decision as its gold counterpart.

The pipeline has four layers: a slot- and unit-preserving output contract, a source-value
guard that checks every number against the statement, a rollout certificate that sends
decision-sensitive readings to review, and a counterfactual simulator for service outcomes.

## Layout

```
src/trc_ridehail/     contract schemas, parser, provenance guard, simulator, rollout, certification
src/lm_client.py      client for an OpenAI-compatible local serving endpoint
scripts/              experiment drivers and summarisation
data/policies/        120-statement bilingual policy corpus
data/derived/         demand scenarios, zone-pair travel model, scenario summary
results/              model outputs per contract, route-level records, rollout and ambiguity results
survey/               questionnaire database, screening and analysis scripts, quality protocol
```

## Environment

```
pip install -r requirements.txt
```

Python 3.10. The translation experiments expect a local serving endpoint that implements
the OpenAI chat-completions API with JSON Schema response formats; set `base_url` in
`src/lm_client.py` if it is not on `http://127.0.0.1:1234/v1`.

## Translation experiment

```
PYTHONPATH=src python scripts/run_policy_pilot.py \
  --model qwen/qwen3-coder-30b --schema-mode unit --shots 0 \
  --corpus evaluation --limit 120
```

`schema-mode` selects the output contract (`static`, `rich`, `activated`, `unit`,
`subset`). `scripts/audit_components.py` reproduces the component audit reported in the
paper: activation precision and recall, the rule-based baseline, the strict guard variant,
and the numeric-extraction probes. `scripts/audit_rollout_regret.py`
recomputes rollout match and regret from the saved records, separating statements that
produced no configuration from those that reached simulation. Records are written to `results/policy-<corpus>/`; `scripts/summarize_results.py`
recomputes exactness, source acceptance, and false acceptance from those records.

## Rollout and ambiguity experiments

```
PYTHONPATH=src python scripts/evaluate_policy_rollout.py
PYTHONPATH=src python scripts/certify_ambiguity.py
```

## Questionnaire analysis

```
cd survey
python screen.py --db survey-release.db --out analysis
python analyze.py --db analysis/survey_clean.db --out analysis/clean
```

The screening rules are fixed in `survey/QUALITY_PROTOCOL.md`. The released database
contains participant codes, verdicts, response times, confidence ratings, and free-text
notes about configurations; it contains no names or contact details.

## Data sources

Trip demand is derived from the New York City Taxi and Limousine Commission
high-volume for-hire vehicle records, which are public. Raw trip files are not
redistributed here; `scripts/prepare_data.py` documents how the derived scenarios were
built from them.

## Licence

Code: MIT. Derived data and the policy corpus: CC BY 4.0.
