# Decode & Detect

Score every transaction in the UK CBT hackathon test set from 0 to 1. Higher means more suspicious.

## Layout

```
payments.py          organiser loader (one USD-valued edge per payment)
src/                 detector code
notebooks/           starter notebook
data/train/          labelled split (junction, not committed)
data/validation/     unlabelled practice split (junction, not committed)
data/test-data/      unzip the scored download here
```

`data/train` and `data/validation` are Windows junctions to `C:\Users\anass\Downloads\decode-and-detect-training-data`. The parquet files stay out of this OneDrive repo and out of git.

## Reproduce

Python 3.13. From the repo root, after the test download is unzipped into `data/test-data/`:

```sh
pip install -r requirements.txt
jupyter notebook notebooks/starter_notebook.ipynb
```

Run all cells. The notebook trains on `data/train` and writes `predictions.csv` in the working directory. That file is one row per test transaction: `TRANSACTION_HASH,score`. It needs a laptop and finishes in under a minute on the starter model.

## Not written here

- `payments.py` and `notebooks/starter_notebook.ipynb` come from the hackathon organisers. Paths in the notebook were pointed at `data/`.
- Libraries: pandas, pyarrow, scikit-learn, numpy, jupyter.
