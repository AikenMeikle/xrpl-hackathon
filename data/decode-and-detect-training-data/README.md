# Decode & Detect

Find the suspicious payments on a fictional ledger.

You get a week of synthetic payment activity. Some of it was planted to look like money laundering. Your job is to build a detector that gives **every transaction a suspicion score from 0 to 1**, so that the suspicious ones rise to the top. You upload your scores and your code at **https://hackathon.ukcbt.org**, and the site scores you straight away.

This is a **synthetic** dataset of activity on a fictional private ledger. Its Parquet field names and types match the inspected AWS/SonarX XRPL dataset. Nothing here is a real allegation, customer record or live blockchain transaction.

## What you need to do

1. **Download both packages** from the site. This one is the **training data**: labelled examples, unlabelled practice data, a starter notebook and this guide. The **test data** is a separate download: the transactions you will be scored on.
2. **Build your detector.** Use `train/` and its labels to learn what suspicious activity looks like. The quickest start is `starter_notebook.ipynb` (see below). Use any language and any tools.
3. **Score the test data.** Run your detector on the test download and write a CSV with one score for every test transaction (format below).
4. **Package your code** as a ZIP (contents below).
5. **Upload both files** at https://hackathon.ukcbt.org: your code ZIP and your predictions CSV. You see your score, precision and recall at once, and your team's best run appears on the public leaderboard.
6. **Improve and repeat.** Each team can upload **50 runs per day**; the count resets at 00:00 UTC. Only your team's best score counts on the leaderboard.

Sign up on the site with your name and email, then create a team or join one through a teammate's invite link. Everyone in a team submits under the team's name and shares its 50 daily runs.

## Any programming language

**You can use any programming language, library or tool.** Python, R, Julia, JavaScript, Rust, Java, Go, SQL with DuckDB, a notebook: all are fine. The site never runs your code. It scores your CSV and stores your ZIP so the organisers can review and reproduce your result.

The starter notebook happens to be Python. The data is standard Parquet and CSV, which almost every language can read.

## What is in each package

**Training data** (this package):

| Path | What it is |
|---|---|
| `train/` | Labelled transactions to learn from |
| `train/labels.csv` | The labels for `train/` |
| `validation/` | Unlabelled practice data, to check your pipeline runs end to end |
| `starter_notebook.ipynb` | A Jupyter notebook that trains a simple model and writes an uploadable CSV |
| `payments.py` | Loads the payments and values them in US dollars; used by the notebook |
| `checksums.json` | SHA-256 of every file, to check your download |

**Test data** (separate download): `transactions.parquet`, `transfers.parquet`, `blocks.parquet` and `market_rates.csv`, with **40,101 transactions**. These are the transactions you score and submit.

The site only scores the test data. It does not score `validation/`. To measure your own progress before you upload, hold back part of `train/` and score yourself on it.

Each split has `transactions.parquet`, `transfers.parquet` and `blocks.parquet`, plus `market_rates.csv`. Evaluation labels are retained by the organisers. Wallets and transaction hashes are disjoint between splits.

## Understanding the data

The data covers **successful XRP, issued-currency (IOU), and cross-currency payments**. There are one, two or three transfer rows per transaction. Synthetic USD, GBP and EUR tokens are issued by two fictional accounts; USD from one issuer is a different asset from USD from the other. Asset and conversion choices do not inspect suspicious labels. Both legitimate and suspicious activity include conversions.

Trust lines and funded token holdings exist before the supplied window. Issuers have Default Ripple enabled, no freeze and no transfer fee. A direct conversion exchanges two assets through a funded liquidity provider; an XRP bridge exchanges three assets through two providers. Each executed leg moves existing funds. The destination receives the stated delivered amount; `SendMax` is the source spending ceiling, **not the actual source debit**. Read the first executed leg for actual cost. Providers retain inventory and spreads. Quotes vary over time and by payment.

This models funded atomic conversions, not a full DEX order book, AMM or rippled execution engine. It does not simulate issuance/redemption during the window, NFTs, partial payments, failures or batch transactions. **Every transaction fee is paid and burned in XRP**, including IOU payments. Token amounts use six decimal places, a restricted subset of XRPL issued-currency precision. Optional signing fields are null. Hashes and account digests are simulation identifiers, not XRPL cryptographic proofs. `VALIDATED` records acceptance by the simulated ledger, not validation on the public XRP Ledger.

### Joining the tables

- Join transfers to transactions using `TRANSACTION_HASH` (many to one). The transfer key is `(TRANSACTION_HASH, TRANSFER_INDEX)`; indices order executed legs.
- Conversions use `TRANSFER_TYPE=path_payment`; same-asset payments use `direct_payment`, following the observed AWS values.
- **Submit one score per transaction, never one per transfer.**
- Join either table to blocks using `LEDGER_INDEX` and `LEDGER_HASH`. Do not treat a block's `TRANSACTION_HASH` as a foreign key to one transaction.
- For payments, `EXTRA_TRANSACTION_FIELDS` is a JSON string containing destination and amount fields; `DELIVERED_AMOUNT` is also JSON text. These are denormalised data records, not transactions ready to sign.

### Assets and values

| Asset | Amount JSON | TOKEN_ADDRESS / SYMBOL | SOURCE_VALUE | VALUE |
|---|---|---|---|---|
| XRP | String of integer drops | XRP sentinel / XRP | Drops | XRP units |
| IOU | `{"currency":"USD","issuer":"r…","value":"12.34"}` | Issuer address / currency code | Token units (12.34) | Token units (12.34) |

Group assets by **both `TOKEN_ADDRESS` and `SYMBOL`**. Do not compare nominal values of different assets as equivalent money. `market_rates.csv` gives hourly fictional reference USD prices for each asset, including issuer identity: `ASSET` is XRP or `issuer:currency`, and you divide `PRICE_MICRO_USD` by 1,000,000 for the price of one asset unit. Use the hour containing the ledger timestamp. Reference prices are valuations, not executable quotes; actual costs include variable spreads and integer rounding. `TOTAL_COINS` measures XRP supply only. See the [XRPL currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats) for the protocol representation.

### What you are looking for

The data includes commerce, peer payments, exchange/payroll-like hubs, legitimate multi-hop treasury/service activity, and planted suspicious motifs. Difficult cases can share characteristics with legitimate activity. A suspicious label describes the synthetic scenario's intent, not a rule that every matching real-world pattern is criminal.

Suspicious transactions are **64% easy, 18% medium and 18% hard**. Legitimate activity is additional. Each full split spans **seven days**. Normal activity includes recurring commercial relationships and settlements. Difficult cases can involve several payments and changing counterparties across longer histories: look at aggregate flows and context, not just isolated payment amounts or fixed routes. The test data includes batch variations absent from training.

Of the 225 hard transactions per full split, 200 have planted historical patterns; 25 are deliberately ambiguous ordinary payments whose labels are independent of observable features. These 25 count towards the overall score and are reported separately. Lucky guesses remain possible; scores are not artificially capped. Private scenario/cohort IDs in the training labels are annotation metadata, not features available in the test data.

## Your predictions CSV

```csv
TRANSACTION_HASH,score
<a transaction hash from the test data>,0.82
```

- One row for **every** test transaction, exactly once. Missing, extra or duplicate IDs are rejected.
- `score` is a number from 0 to 1. Higher means more suspicious. 0 and 1 are both allowed. Spread your scores across the range so they rank the transactions.
- An optional `reason` column is accepted but not scored.
- Maximum 12 MiB and 100,000 rows.

The site checks your file in the browser before uploading, so a missing or wrong ID is caught without using up a run.

## Your code ZIP

Upload the code that produced your CSV. Include:

- **All your source code** (scripts or notebooks).
- **A README with the exact command** that produces your predictions CSV from the test data, plus the hardware or time it needs.
- **A dependency list** (for example `requirements.txt`, `package.json`, `renv.lock` or `Cargo.toml`) and the language version.
- **A LICENSE file** with an open-source licence such as MIT or BSD 3-Clause. The terms require your original work to be released under one.
- **A list of what you did not write yourself**: libraries, pre-trained models, outside services and AI assistance.

Leave out the datasets, installed dependencies (such as `node_modules` or a virtual environment) and any passwords or API keys.

Limits: 8 MiB compressed, 64 MiB uncompressed, 2,000 files. Encrypted, multi-part and ZIP64 archives are rejected.

**Make sure it reproduces.** The organisers may re-run your code. If they cannot reproduce your predictions from your instructions, the entry can be rejected.

## How you are scored

- **Score = 100 × average precision (AP)**, grouping tied scores at the same threshold. AP rewards putting the suspicious transactions at the top of your ranking.
- **Precision and recall** are also shown for the top 1% of your scores. Ties within that review budget are broken by transaction hash, ascending.
- The leaderboard shows each team's best run on the test data.

## Rules

- You may use the whole supplied transaction window. This is an offline investigation, not real-time detection.
- Do not use external labels, organiser files, generator seeds or test answers.
- Which split a transaction comes from is not a feature.
- The full rules, including team size, prizes and what your final entry must contain, are in the terms at https://hackathon.ukcbt.org/terms.

## Starter notebook

`starter_notebook.ipynb` is the easiest place to begin. It:

1. loads the payments,
2. turns each one into a few simple rules, such as "the receiver passes on a similar amount within an hour" or "many payments from one sender in ten minutes",
3. trains a **logistic regression** on `train/` and its labels, which learns how much each rule matters,
4. checks itself on a held-back fifth of the training data, using the same score as the site,
5. scores the test data and saves `predictions.csv`, ready to upload.

`payments.py` gives one effective sender-to-recipient edge per transaction, valued using the public reference prices. Raw transfer legs remain available for more detailed analysis. Keep `payments.py` alongside the notebook.

It runs in under a minute on a laptop. Put it next to `train/` and `payments.py`, unzip the test download into `test-data/`, then:

```sh
pip install pandas pyarrow scikit-learn jupyter
jupyter notebook starter_notebook.ipynb
```

To improve your score, add your own rules in the marked cell and run it again. The notebook shows which rules the model relied on, so you can see whether a new idea helps.
