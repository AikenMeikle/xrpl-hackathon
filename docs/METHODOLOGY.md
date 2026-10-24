# Methodology (4 pages / <10 slides)

## 1. Problem, data and correct XRPL reconstruction

**Task.** Rank 40,101 test transactions so planted laundering rises to the top; score = 100 x average precision over
all transactions. Synthetic private ledger, seven days per split, 700 wallets per split, five assets (XRP plus
USD/GBP/EUR tokens from two fictional issuers). Wallets and hashes are disjoint between train and test, so no
identity may be used, only shape.

**What we verified in the package** (`artifacts/inspection_report.md`): 40,131 labelled train transactions
(1,250 suspicious = 3.11%), 60,557 transfer legs (1/2/3 legs: 23,681 / 12,474 / 3,976), all `Payment` and
`tesSUCCESS`, `TIMESTAMP` in Unix seconds, hourly `market_rates.csv` keyed by `(hour, issuer:currency | XRP)`,
zero missing rates, no duplicated hashes. Labels carry `family/difficulty/scenario_id/cohort_id` annotation that is
used only for reporting and fold diagnostics, never as features.

**Reconstruction rules** (`src/normalize.py`, unit-tested in `tests/test_ledger_semantics.py`):

- XRP amounts are drops: `"250000000"` is 250 XRP. Token amounts are never divided.
- Asset identity is `issuer:currency`; USD from issuer A and USD from issuer B are different assets, priced separately.
- `Amount/DeliverMax/SendMax` are ceilings. Paid = first executed transfer leg; delivered = metadata `DELIVERED_AMOUNT`
  (cross-checked against the last leg). Partial-payment and ceiling-vs-delivered discrepancies are kept as flags.
- A 2- or 3-leg conversion is ONE economic payment edge (Graph A). Liquidity-provider legs live only in the audit
  table (Graph B) and feed conversion spread / asset-transition features. They are never laundering hops.
- Fees (10-15 drops) are burned XRP stored separately; never a recipient. Failed transactions would keep a scored row
  with zero graph context (none exist in this package, but the path is implemented and tested).
- USD uses the reference price of the exact asset in the UTC hour containing the payment; missing rates stay NaN with
  a flag (never zero, never a future price).
- Ordering is `(LEDGER_INDEX, TRANSACTION_INDEX)`, not timestamp or hash order.

## 2. Flow graph, eight pattern families, directed GNN

**Graph A** is a directed temporal multigraph: one edge per economic payment with sender, receiver, time, paid and
delivered asset/USD, fee, conversion flag. Two sorted per-wallet indexes (outgoing by sender, incoming by receiver)
answer every "events of wallet w in [t-back, t+fwd]" query with two `searchsorted` calls, so all window features are
vectorised over the 40k edges (`src/graph_build.py`). A collapsed undirected view gives components, k-core,
clustering, PageRank and Louvain communities.

**Feature families** (501 columns, dictionary in `artifacts/feature_dictionary.csv`):

- `tx_` transaction: log USD paid/delivered, legs, conversion spread, value retention, fee, UTC time, split-stable
  symbol flags, roundness, amount relative to sender / receiver / asset medians, activity scarcity.
- `full_` / `asof_` / `win_` / `fwd_` wallet: seven-day profile (counts, USD, degrees, asset diversity, burst share,
  temporal entropy, one-use counterparties, concentration, pass-through, flow mismatch, hub- and service-likeness,
  gap regularity, pair history, 1- and 2-hop neighbour means) projected to sender and receiver with differences; as-of
  (strictly before) and centred rolling windows at 10m/1h/6h/24h; forward windows for the receiver. Prefixes make the
  information horizon explicit: `full_` features are offline-batch features, not real-time alerts.
- `match_` value-aware temporal matching: for A->B at t1 and B->C at t2>t1, quality
  `exp(-|log(out/in)| / 0.20)`, best and batch-cumulative matches at 1h/6h/24h in both directions (a 100 deposit split
  into 4x25, or several deposits swept at once), delays, counts, same-asset flag, witness hash pairs. We report
  "flow-consistent evidence", not fund identity.
- Eight detectors (`src/features_motifs.py`, fixtures in `tests/test_motifs.py`), each with counts, best soft
  quality, lags, USD scale, size and witness hashes attached to the particular edge:
  fan-in (collector sweep ratio, largest exit share, deposit clustering, one-use senders, exit-role support);
  fan-out (payout CV/entropy/equal-size fraction, just-funded ratio, recipient forwarding and one-use fractions,
  funding-edge projection); scatter-gather (distinct chronologically valid middlemen per source/destination via a
  2-hop pair table, branch hold times, value match, bounded 3-hop); gather-scatter (6h hub in/out degree, flow
  mismatch, pass-through, counterparty disjointness, in-to-out lag, exits-after-funding); cycles (time-respecting
  value-matched simple cycles of 2-5 hops within 24h/7d by iterative pair joins, no repeated vertex, dust excluded);
  layered stack (forward/backward DP longest value-matched chain, single-use intermediary fraction, hop-delay
  statistics, retention, peel ratio); bipartite (2x2 payer/receiver rectangles via `C = B B^T`, hub-free and same-day
  variants, block size/density/external contact/synchrony); irregular network (day-window ego graph with branching
  and merging nodes, 2-5-hop route counts, local cycle support, community volume concentration, bridge status,
  complexity score).
- Legitimate look-alikes get positive motif scores by design; the model learns the counter-evidence (sustained
  activity, repeat counterparties, retention, no purposeless hops).

**DirectedEdgeGraphSAGE** (`src/models/gnn.py`, native PyTorch, no PyG): shared encoders for node profile and edge
attributes; two layers that aggregate incoming-direction and outgoing-direction messages separately with degree
normalisation, combine with the node state through a residual + LayerNorm + GELU + dropout block; edge head on
`concat(h_src, h_dst, |h_src-h_dst|, h_src*h_dst, enc(edge), motif features)` -> 128 -> 64 -> 1. BCEWithLogits,
AdamW (lr 3e-3, wd 1e-4, full-batch so one step per epoch, up to 150 epochs, patience 20), dropout 0.2, early
stopping on held-out AP, scalers fitted on the training fold only, no wallet-id embeddings; 3-fold OOF on exactly the
LightGBM folds (best epochs 49 / 54 / 39). The RTX 5070 Laptop GPU (8 GB) was detected by `nvidia-smi`, but the
installed torch 2.14.1 is a CPU build; rather than spend the hackathon on a CUDA reinstall, the CPU path (about
1-4 s/epoch for the whole 40k-edge graph) is the supported path and is what produced the numbers below.

## 3. Blocked validation, measured AP, ablations

**Primary protocol.** 3-fold `StratifiedGroupKFold` over *flow groups* built from unlabelled structure: union-find
over value-matched forwards (quality >= 0.6 within 6h through non-hub intermediates), same-wallet 10-minute bursts
and repeated pairs; the 22 highest-degree service hubs do not link. Result: 21,143 groups, largest 8.0% of edges,
416/417/417 positives per fold. Because the split has only 700 wallets, every validation edge shares endpoints with
training edges (wallet overlap 100%): the protocol isolates flows and scenarios, not wallets. Secondary diagnostics:
random split (optimistic bound), scenario-grouped folds (annotation used for fold assignment only), last-2-day time
holdout with a 6-hour embargo, and an as-of-only feature stress test.

**Measured (every number is in `artifacts/validation.json`; "mean" = mean of the three fold APs, "pooled" = AP of
the concatenated OOF scores, which is what a single 40k-row test file resembles):**

| Model | protocol | AP x100 |
|---|---|---|
| Organiser logistic baseline | random 20% holdout | 24.6 |
| LightGBM, transaction features only (52) | blocked flow groups, fast settings | 11.4 |
| LightGBM, transaction + wallet/window features (352) | blocked flow groups (earlier, coarser grouping) | 68.8 mean (68.0 / 70.8 / 67.6) |
| LightGBM, + matching + eight motif families (501), spw=1 | blocked flow groups | 69.6 mean (70.7 / 74.3 / 63.8) |
| **LightGBM, 501 features, spw=3 (shipped)** | blocked flow groups | **69.7 mean (70.7 / 74.5 / 63.9), 68.4 pooled**; P@1% 0.84, R@1% 0.27 |
| DirectedEdgeGraphSAGE alone, same folds | blocked flow groups | 55.6 mean (56.0 / 58.6 / 52.4), 54.5 pooled |
| Rank blends, GNN weight 0 / 0.15 / 0.30 / 0.50 | blocked flow groups, pooled | 68.4 / 66.1 / 64.4 / 62.4 -> weight 0.0 chosen |
| LightGBM, fast settings | random 3-fold (optimistic bound) | 83.1 |
| LightGBM, fast settings | scenario-grouped folds (organiser annotation for fold assignment only) | 53.8 |
| LightGBM, fast settings, unweighted vs spw=3 | blocked flow groups | 68.4 vs 68.9 |

Reading these honestly: the label-free flow groups block most but not all scenario leakage (scenario-grouped AP is
about 15 points lower than flow-group AP under identical fast settings), so the hidden test score is more likely to
sit between the scenario-grouped and flow-grouped estimates than above them. The GNN is a working, measured model
but is clearly weaker than the tree model on this data and lowers AP at every blend weight, so the submitted score is
LightGBM alone. The full per-family cumulative ablation was started but stopped for time; the three measured points
(transaction only 11.4 -> +wallet windows 68.8 -> +matching and motifs 69.6) are reported instead of invented deltas.

**LightGBM.** binary objective, lr 0.035, 31 leaves, min_child_samples 60, lambda 5, colsample 0.85, up to 2,500
trees with AP early stopping (100 rounds); the modest positive weight (3) was adopted because it improved fast-setting
OOF AP (68.4 -> 68.9). Final stage-1 model: 1,347 trees (median best iteration), refit on all 40,131 labelled rows.
Leaderboard (organiser private labels): **60.5**, P@1% 89%, R@1% 28%, ROC-AUC 0.90; easy 40%, medium 10%,
observable-hard 0%, ambiguous 0% of each class in the top 1%. That lands between the flow-group and scenario-grouped
estimates, as predicted above.

**Stage 2: score propagation (`src/stage2.py`).** A planted scenario spans about seven payments and the quiet ones
sit next to the loud ones. Stage 2 adds label-free aggregates of stage-1 scores over *other* edges sharing the sender,
receiver, pair or community, and over the receiver's later sends / sender's earlier receipts within 24h, then retrains.
Validation is nested and graph-faithful: training rows use inner out-of-fold stage-1 scores and see only training-row
neighbours; validation rows use scores from a model that never saw their labels and see only validation-row
neighbours, which is exactly the independent-test-graph situation. On scenario-grouped folds (fast settings) stage 2
improved every fold: 55.7 -> 60.2, 49.7 -> 50.9, 55.0 -> 57.7 (pooled 50.8 -> 53.2). With full settings the
scenario-grouped early-stopping point is very early (83 trees, OOF 54.2 -> 55.4), so two stage-2 files were produced:
`predictions_stage2.csv` (83 trees) and `predictions_stage2_500.csv` (500 trees); absolute-time features were dropped
in both. Hard-case recall at the top 1% did not move under the strict protocol; that remains the open problem.

**Leakage controls.** No hashes, wallet ids, issuer addresses, scenario/cohort ids or labels in features (asserted by
name and by source inspection in `tests/test_pipeline.py`); scalers fitted per fold; folds never split a flow group;
train and test graphs are built independently (disjoint wallets, different issuers); no neighbour-label aggregation.

## 4. Evidence, false positives, limitations, reproducibility

Every scored alert in `artifacts/test_alert_evidence.json` lists exact TreeSHAP drivers, motif scores, witness hash
sequences (forward/backward matches, scatter-gather branches, cycle or chain paths), sender/receiver summaries and
data-quality flags. `docs/CASE_STUDIES.md` walks through three complementary test alerts, the highest-scoring
legitimate training transactions (payroll/exchange/merchant look-alikes) and benign transactions inside busy
irregular neighbourhoods, all phrased as "flagged for review".

**Limitations.** Synthetic scenario intent is not real-world guilt; 25 ambiguous positives per split are unlearnable
by design and cap achievable AP; a seven-day window hides longer histories; USD valuations are reference prices;
account-based ledgers do not allow deterministic fund tracing, so motif evidence is flow-consistency, not identity;
the GNN was trained full-batch on CPU with a small budget; strict per-fold inductive graph rebuilding was not done
(features use the whole unlabelled split, which the rules allow for this offline task).

**Reproducibility.** `python run.py all` (or the step list in `README.md`) regenerates everything from the two data
folders; `artifacts/manifest.json` records input SHA-256s, package versions, git commit, seed and commands. Tests:
`python -m pytest -q` (ledger semantics, eight motif fixtures with look-alikes, pipeline contracts).
