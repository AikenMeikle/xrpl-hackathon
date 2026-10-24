# Decode & Detect: transaction-level laundering ranking on a synthetic XRPL ledger

Scores every test transaction in `[0, 1]` (higher = more suspicious) for the UK CBT "Decode & Detect: The XRPL AI
Hack". The pipeline reconstructs one economic payment per transaction from the XRPL-shaped tables, builds wallet,
value-flow and eight-typology motif features, trains a group-blocked LightGBM ranker, trains a directed edge-aware
GraphSAGE as a comparison, and blends only when out-of-fold average precision supports it.

Everything is reproducible with one command once the data is in place. Nothing in the repo contains data or secrets.

## Layout

```
run.py                   CLI (inspect | baseline | features | validate | train --final | predict | verify | explain | package | all)
payments.py              organiser loader (used by the baseline only)
src/config.py            paths, seeds, windows, caps, model hyper-parameters
src/normalize.py         canonical transaction table + transfer-leg audit table (XRPL semantics)
src/graph_build.py       Graph A economic-payment indexes (sorted per-wallet window index), static views
src/features_transaction.py, src/features_wallet.py, src/features_motifs.py
src/validation.py        unlabelled flow groups, StratifiedGroupKFold, metrics
src/models/lgbm.py       LightGBM with AP early stopping, TreeSHAP contributions
src/models/gnn.py        DirectedEdgeGraphSAGE (native PyTorch, CPU/GPU)
src/pipeline.py          orchestration (features -> validate -> final fit -> predict)
src/explain.py           evidence JSON/CSV, case studies, feature dictionary
src/predict.py           rank blend, writer, submission verifier
tests/                   ledger-semantics, motif-fixture and pipeline contract tests (pytest)
docs/METHODOLOGY.md      4-page methodology; docs/AI_USAGE.md; docs/CASE_STUDIES.md
artifacts/               generated: features, validation.json, manifest.json, evidence (parquets ignored by git)
submission/              predictions.csv and the code ZIP (ignored by git)
```

## Data placement

Default search paths (see `src/config.py`, override with `--train/--test`):

- train: `data/decode-and-detect-training-data/train/` (must contain `labels.csv`)
- test: `data/test-data/` (the unzipped test download: `transactions.parquet`, `transfers.parquet`, `blocks.parquet`, `market_rates.csv`)

## Reproduce

Python 3.13 (tested with 3.13.12 on Windows 11, 16-core laptop CPU, 32 GB RAM; no GPU required).

```sh
pip install -r requirements.txt
python -m pytest -q                      # ~30 s
python run.py all                        # inspect -> baseline -> features -> validate -> train -> predict -> verify -> explain -> package
```

or step by step:

```sh
python run.py inspect
python run.py baseline                   # organiser starter model -> submission/baseline_predictions.csv
python run.py features                   # ~6 min per split (motif ego-graph stage dominates)
python run.py validate                   # blocked CV, ablations, GNN OOF, blend choice -> artifacts/validation.json (~35 min)
python run.py train --final
python run.py predict --output submission/predictions.csv
python run.py verify --file submission/predictions.csv
python run.py explain
python run.py package                    # submission/code_submission.zip
```

Approximate wall time end to end: 55 minutes on the machine above. Seeds are fixed (42); LightGBM is deterministic
on the same hardware/threads; the GNN uses fixed torch seeds on CPU.

## Output

`submission/predictions.csv` with header `TRANSACTION_HASH,score`, exactly one row per test transaction, unrounded
floats in `[0, 1]`. The verifier checks header, 40,101 unique hashes equal to the test set, finiteness, range,
absence of constant/binary scores and size <= 12 MiB.

## Results (see `artifacts/validation.json` and `docs/METHODOLOGY.md`)

Primary metric: sklearn average precision over all transactions, 3-fold StratifiedGroupKFold over label-free flow
groups (21,143 groups, largest 8%).

| model | blocked OOF AP x100 (mean of folds / pooled) | folds |
|---|---|---|
| organiser logistic baseline (random holdout) | 24.6 | - |
| LightGBM, 501 features, scale_pos_weight 3 (**shipped**) | 69.7 / 68.4 | 70.7, 74.5, 63.9 |
| DirectedEdgeGraphSAGE (CPU), same folds | 55.6 / 54.5 | 56.0, 58.6, 52.4 |
| rank blend, GNN weight 0.15 / 0.30 / 0.50 (pooled) | 66.1 / 64.4 / 62.4 | GNN weight 0.0 chosen |
| LightGBM, fast settings, scenario-grouped folds (strictest) | 53.8 | - |
| LightGBM, fast settings, random split (optimistic) | 83.1 | - |

The GNN is implemented, measured and documented but did not improve out-of-fold AP at any blend weight, so no
submitted file uses it. See `docs/METHODOLOGY.md` for the honest reading of these numbers.

### Stage 2: score propagation (the later submissions)

`src/stage2.py` adds label-free neighbour aggregates of stage-1 scores (same sender / receiver / pair / community,
downstream and upstream within 24h, self excluded) and retrains. Evaluated with nested, within-fold-only neighbours
on scenario-grouped folds (the strictest protocol), fast settings: stage 1 55.7 / 49.7 / 55.0 -> stage 2
60.2 / 50.9 / 57.7 per fold. Absolute-time features (`tx_hour_of_day`, `tx_day_index`, ...) are dropped in stage 2.

```sh
python -m src.stage2 cv scenario        # nested CV report -> artifacts/stage2_cv_scenario.json
python -m src.stage2 final              # stage-2 trees chosen by scenario-grouped early stopping -> submission/predictions_stage2.csv
python -m src.stage2 final 500          # stage-2 with 500 trees                                  -> submission/predictions_stage2_500.csv
```

Leaderboard runs on 24 Oct 2026: stage-1 LightGBM (`run.py predict`) scored 60.5 (P@1% 89%, R@1% 28%, ROC-AUC
0.90); the stage-2 files were uploaded afterwards and the team's best run is whichever the portal shows.

## Licence and attribution

MIT (see `LICENSE`). Organiser files: `payments.py`, `notebooks/starter_notebook.ipynb`. AI assistance and
third-party components are listed in `docs/AI_USAGE.md`.
