# Decode & Detect — Data Guide

## 1. The task in one sentence
Score **every transaction** (not transfer) 0–1 for how suspicious it is, so AP (average precision) ranks the planted laundering cases above the ~40k legit ones. Scoring = `100 × average precision` on all 40,101 test transactions; top-1% precision/recall and ROC-AUC are shown but don't count toward the leaderboard.

## 2. The five files and what they're for

| File | Grain | Purpose |
|---|---|---|
| `transactions.csv/.parquet` | 1 row per on-ledger transaction | Who sent it, fee, success, raw JSON payment fields |
| `transfers.csv/.parquet` | 1–3 rows per transaction | The *actual money legs* (conversions use 2–3 legs) |
| `blocks.csv/.parquet` | 1 row per ledger | Timing context (ledger close time) — minor, mostly for joins/sanity checks |
| `market_rates.csv` | 1 row per (hour, asset) | Hourly USD reference price, so you can compare amounts in different assets |
| `labels.csv` | training only | `is_suspicious` (0/1), plus `family`/`difficulty`/`scenario_id`/`cohort_id` (metadata, NOT usable as features on test) |

**Key cardinality fact:** a transaction can have 1, 2 or 3 transfer rows:
- 1 leg → simple `direct_payment` (same asset, sender → receiver)
- 2–3 legs → `path_payment` (a conversion): the payment hops through one or two funded liquidity providers. Each leg is a real movement of existing funds, not a simulated order book — so you can trust each leg's `VALUE`.

## 3. Understanding values — the part everyone gets wrong first

**XRP is in drops.** `1 XRP = 1,000,000 drops`. For XRP rows, `SOURCE_VALUE` is drops; `VALUE` is already converted to XRP units. Transaction `FEE`/`SOURCE_FEE` are also in drops (10–15 typically) and are **always burned in XRP**, even for IOU payments.

**An asset = `TOKEN_ADDRESS` + `SYMBOL` together**, not just the currency code. `USD` issued by issuer A and `USD` issued by issuer B are two different assets — never sum or compare them directly. XRP has a sentinel token address (`0x1111...2222` in the sample).

**`SOURCE_VALUE` vs `VALUE`:** for XRP these differ by the drops/XRP factor; for IOUs they're both in token units (same number). Prefer `VALUE` for XRP-unit/token-unit amounts.

**`SendMax` is NOT what was paid** — it's a ceiling. The transaction's `EXTRA_TRANSACTION_FIELDS` JSON has `Amount`, `DeliverMax`, `Destination`, and sometimes `SendMax`/`Paths`. `DELIVERED_AMOUNT` tells you what the destination actually received. **The real cost to the sender is the first transfer leg** (`TRANSFER_INDEX = 0` for that transaction) — read `transfers.csv`, not the transaction JSON, for actual spend.

**Converting to USD:** use `market_rates.csv`. `ASSET` is either `"XRP"` or `"issuer:currency"`. Price = `PRICE_MICRO_USD / 1,000,000` USD per unit. Match on the **hour containing the transfer's `TIMESTAMP`** (floor to the hour, join on `(hour, asset_key)`). These are reference valuations, not executable quotes — real legs have spreads baked in, so don't expect perfect consistency between legs of a conversion.

## 4. Joining the tables

```
transfers.TRANSACTION_HASH  →  transactions.TRANSACTION_HASH   (many transfers : 1 transaction)
transfers/transactions.(LEDGER_INDEX, LEDGER_HASH) → blocks.(LEDGER_INDEX, LEDGER_HASH)
```
Do **not** join on `blocks.TRANSACTION_HASH` as if it points to one transaction — it doesn't represent a real FK there.

`labels.csv` joins to `transactions` (train only) via `TRANSACTION_HASH`.

### Building "one edge per payment" (what `payments.py` does for you)
For each transaction:
1. Take the first transfer leg (`TRANSFER_INDEX == 0`) → real sender, real asset, real amount spent.
2. Take the last transfer leg → real receiver, real asset, real amount received.
3. Convert both to USD via `market_rates.csv` at the transaction's hour.
4. You now have one row per transaction: `sender, receiver, asset_in, asset_out, usd_value, timestamp, fee, success, transfer_type`.

This collapses the 1–3 transfer-row complexity into a clean payment graph you can build features on (sender/receiver history, fan-in/fan-out, timing deltas, round-trips).

## 5. What "suspicious" looks like (from the training brief)

The dataset plants 8 classic laundering topologies (from IBM AMLworld-style synthetic generators):

| Shape | Pattern | Legit look-alike |
|---|---|---|
| Fan-in | many senders → 1 collector → 1 big exit | shop/fundraiser/exchange deposit |
| Fan-out | 1 source → many similar-sized payouts | payroll, refunds |
| Scatter-gather | split via middlemen, re-merges at one destination | internal treasury rebalancing |
| Gather-scatter | hub collects from many, pays out to many, low retained balance | exchange hot wallet |
| Cycle | funds leave and return to origin (possibly via other assets) | arbitrage/market making |
| Layered stack | money passes through several rows of one-off middlemen, value preserved | correspondent payment chains |
| Bipartite | dense many-to-many between two wallet groups | marketplace buyers/sellers |
| Irregular network | no clean template, splits/merges/returns | ordinary peer-to-peer life |

Common suspicious *signals* across shapes: rapid pass-through (money leaves soon after arriving), purposeless hops (middle wallets with no other activity), amount splitting, currency hopping across several assets, round-tripping, and patterns only visible across the whole 7-day window (not a single payment).

**Difficulty mix:** suspicious txns are 64% easy / 18% medium / 18% hard. Of the hard ones, ~89% have genuine planted historical patterns (need multi-payment/graph context), the rest (~11% of hard, 25 per split) are *deliberately unlabelable noise* — ambiguous ordinary payments with no detectable signal. Don't over-fit chasing those; expect a ceiling.

## 6. Practical pitfalls checklist (from the 6 traps slide)
1. Score **transactions**, not transfers — one row per `TRANSACTION_HASH` in your submission.
2. XRP is in drops — don't treat drops as XRP.
3. Asset = `TOKEN_ADDRESS` + `SYMBOL`, never currency code alone.
4. `SendMax` ≠ actual cost — use the first transfer leg.
5. Compare everything in USD via `market_rates.csv`, matched to the hour.
6. `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT` are JSON strings — parse them, don't regex them.

## 7. Suggested pipeline
1. **Load & join**: transactions + transfers + blocks; sanity check row counts (1–3 transfers per transaction, every transaction has ≥1 transfer).
2. **Build payment edges** (`payments.py` logic above) — one row per transaction, USD-valued.
3. **Run the baseline** (`starter_notebook.ipynb`) to get a valid submission and a number to beat.
4. **Exploit train labels**: pull out `is_suspicious==1` rows, look at their wallets' *entire* history (not just the flagged transaction) — plot in/out flows per wallet over the week.
5. **Engineer features** per transaction/wallet, e.g.:
   - sender/receiver transaction counts & distinct counterparties in rolling windows (10 min / 1 hr / 1 day)
   - time since wallet's previous in/out transaction
   - ratio of amount to sum of that wallet's recent inflows (pass-through detection)
   - number of distinct assets touched by a wallet in the window (currency hopping)
   - graph cycle detection: does money return to the origin wallet within N hops/hours?
   - how "round" / similar payout amounts are among a sender's recent transactions (fan-out splitting)
   - net balance retained by a wallet vs. throughput (gather-scatter hub detection)
6. **Train** (logistic regression to start, as starter notebook does — interpretable, shows which rules matter; move to gradient boosting/graph features once you have more signal).
7. **Validate**: hold out a fold of `train/`, compute AP yourself the same way the site does, before burning a daily submission.
8. **Score test data**, write `TRANSACTION_HASH,score` CSV, package code + README + requirements + LICENSE + "what I didn't write" disclosure into the ZIP.

## 8. Scoring mechanics recap
- Score = 100 × Average Precision over the full ranked list (grouping tied scores). Only *order* matters — scores don't need to be calibrated probabilities, just avoid excessive ties.
- Top-1% (~402 transactions) precision/recall shown separately, ties broken by hash ascending — not part of ranking but a useful proxy for "what would an analyst's queue look like."
