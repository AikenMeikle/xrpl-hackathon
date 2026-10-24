# AI usage disclosure

The organisers require disclosure of AI assistance. This entry was built with substantial AI assistance.

## Tooling

- IDE: Cursor (Agent mode, with a Plan-mode design step before Build).
- Model: Claude (Anthropic) as exposed in Cursor, with extended thinking enabled. The team member driving the session
  reviewed outputs, approved the plan, ran commands and uploaded the results.
- Prompts: the main instruction prompt is stored in condensed form (all sections and constraints kept) in
  `prompts.txt` (repo root). It specifies the XRPL data semantics, the eight typology detectors, the leakage rules,
  the validation protocol and the deliverables.

## What the AI did

- Read the organiser README, `payments.py`, the starter notebook, the teammate's earlier detector and the actual
  Parquet schemas before writing code.
- Wrote all modules under `src/`, `tests/` and `run.py`, the documentation in `docs/`, and iterated on failing tests
  and runtime errors.
- Proposed feature definitions, the blocked flow-group validation protocol, the GNN architecture and the blend rule.

## What humans did

- Decided the goals, constraints and priorities (prompt), approved the plan, monitored runs, and are responsible for
  the final upload through the organiser portal.
- Earlier team work reused in spirit: a teammate's `aiken/basic.py` (vectorised window index and graph-grouped CV);
  its component ideas were re-implemented here, not copied wholesale.

## Not written by us

- Libraries: numpy, pandas, pyarrow, scikit-learn, scipy, LightGBM, networkx, PyTorch, joblib, pytest.
- Organiser files: `payments.py`, `notebooks/starter_notebook.ipynb` (paths adapted), data README.
- XRPL Commons / UKCBT training material informed the data semantics section of the methodology (CC BY-SA
  attribution where adapted). No pre-trained models, external data, live XRPL calls or outside services were used.
  No LLM is called inside the prediction pipeline.
