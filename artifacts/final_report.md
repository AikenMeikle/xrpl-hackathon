| check | status | evidence |
|---|---|---|
| input schema / row counts | PASS | train 40,131 tx / 60,557 legs; test 40,101 tx; legs 1/2/3 = {'1': 23681, '2': 12474, '3': 3976}; rate_missing=0; shared wallets/hashes = {'shared_wallets': 0, 'shared_hashes': 0} |
| one economic edge per successful payment | PASS | edges=40,131, conversions=16,450 collapsed to single edges |
| absence of label leakage | PASS | no hash/wallet/issuer/scenario/label columns (asserted); folds never split a flow group; scalers per fold; test graph independent |
| GNN GPU/CPU smoke test | PASS | device=cpu, params=160,193, torch=2.14.1+cpu, cuda_available=False |
| LightGBM blocked AP | PASS | OOF AP=68.39 folds=[70.66, 74.46, 63.85] std=4.39 P@1%=0.838 R@1%=0.269 ROC-AUC=0.967 spw=3.0 |
| GNN blocked AP | PASS | OOF AP=54.53 folds=[55.95, 58.56, 52.38] best_epochs=[49, 54, 39] |
| chosen blend AP | PASS | GNN weight=0.0 -> OOF AP=68.39; all blends: w=0.0:68.39, w=0.15:66.12, w=0.3:64.39, w=0.5:62.38 |
| ablation ledger | PASS | organiser_baseline_logreg (random 20% holdout)=24.57; +basic_transaction=11.44 |
| secondary diagnostics (fast LightGBM) | PASS | random=83.11, scenario-grouped=53.76, as-of-only=39.57, time-holdout=61.04 |
| all-eight-motif fixture coverage | PASS | feature counts per family {'fanin_': 16, 'fanout_': 23, 'sg_': 10, 'gs_': 20, 'cycle_': 11, 'chain_': 15, 'bip_': 13, 'irr_': 17}; fixtures in tests/test_motifs.py |
| output cardinality / range / bytes | PASS | rows=40,101 unique_scores=40,101 range=[4.4e-07,0.999] bytes=3,316,060 problems=[] |
| reproducible command | PASS | python run.py all  (manifest: checksums, versions, commit 5a2d9793, seed 42) |
| methodology (<4 pages) | PASS | docs/METHODOLOGY.md 1621 words; docs/AI_USAGE.md; docs/CASE_STUDIES.md |
| code ZIP (<= 8 MiB) | PASS | C:\Users\anass\OneDrive\Desktop\xrpl-hackathon\submission\code_submission.zip 0.15 MB |
| actual upload status | NOT DONE | upload requires the team access code on https://hackathon.ukcbt.org; not performed by the pipeline |