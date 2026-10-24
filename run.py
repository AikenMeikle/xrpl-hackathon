"""Decode & Detect pipeline entry point.

    python run.py inspect
    python run.py baseline
    python run.py features
    python run.py validate
    python run.py train --final
    python run.py predict --output submission/predictions.csv
    python run.py verify --file submission/predictions.csv
    python run.py explain
    python run.py package
    python run.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*highly fragmented.*")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS, SUBMISSION, TEST_CANDIDATES, TRAIN_CANDIDATES, first_existing  # noqa: E402

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _dirs(args):
    train = Path(args.train) if getattr(args, "train", None) else first_existing(TRAIN_CANDIDATES)
    test = Path(args.test) if getattr(args, "test", None) else first_existing(TEST_CANDIDATES)
    return train, test


def cmd_inspect(args):
    from src.inspect_data import write_report
    train, test = _dirs(args)
    rep = write_report(train, test)
    log(f"train: {rep['train']['n_transactions']:,} tx, {rep['train']['economic_edges']:,} edges, "
        f"failed={rep['train']['n_failed']}, rate_missing={rep['train']['paid_rate_missing']}")
    log(f"test:  {rep['test']['n_transactions']:,} tx, {rep['test']['economic_edges']:,} edges, "
        f"failed={rep['test']['n_failed']}, rate_missing={rep['test']['paid_rate_missing']}")
    log(f"cross split: {rep['cross_split']}")
    log(f"wrote {ARTIFACTS / 'inspection_report.md'}")


def cmd_baseline(args):
    from src.baseline import run_baseline
    from src.predict import verify_file
    import pandas as pd
    train, test = _dirs(args)
    out = SUBMISSION / "baseline_predictions.csv"
    info = run_baseline(train, test, out)
    log(f"organiser baseline: random-holdout AP={100 * info['random_holdout_ap']:.2f}, rows={info['rows']}")
    exp = pd.read_parquet(test / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"]
    v = verify_file(out, exp)
    log(f"verify {out.name}: ok={v['ok']} {v['problems']}")
    (ARTIFACTS / "baseline_metrics.json").write_text(json.dumps({**info, "verify": v}, indent=2))


def cmd_features(args):
    from src.pipeline import build_features_for_split
    train, test = _dirs(args)
    build_features_for_split("train", train, force=args.force)
    build_features_for_split("test", test, force=args.force)


def cmd_validate(args):
    from src.pipeline import validate
    train, test = _dirs(args)
    validate(train, test, skip_gnn=args.skip_gnn, force=args.force, stage=args.stage)


def cmd_train(args):
    from src.pipeline import train_final
    train, test = _dirs(args)
    train_final(train, test, final=args.final)


def cmd_predict(args):
    from src.pipeline import predict
    train, test = _dirs(args)
    predict(train, test, Path(args.output))


def cmd_verify(args):
    from src.predict import verify_file
    import pandas as pd
    _train, test = _dirs(args)
    exp = pd.read_parquet(test / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"]
    v = verify_file(Path(args.file), exp)
    print(json.dumps(v, indent=2))
    if not v["ok"]:
        sys.exit(1)


def cmd_explain(args):
    from src.explain import explain
    train, test = _dirs(args)
    explain(train, test)


def cmd_package(args):
    from src.package import package
    out = package()
    log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def cmd_report(args):
    from src.report import build_report
    _train, test = _dirs(args)
    print(build_report(test, run_tests=not getattr(args, "no_tests", False)))


def cmd_all(args):
    cmd_inspect(args)
    cmd_baseline(args)
    cmd_features(args)
    cmd_validate(args)
    args.final = True
    cmd_train(args)
    args.output = str(SUBMISSION / "predictions.csv")
    cmd_predict(args)
    args.file = args.output
    cmd_verify(args)
    cmd_explain(args)
    cmd_package(args)
    cmd_report(args)


def main():
    p = argparse.ArgumentParser(description="Decode & Detect AML pipeline")
    p.add_argument("--train", default=None)
    p.add_argument("--test", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    sub.add_parser("baseline").set_defaults(fn=cmd_baseline)
    f = sub.add_parser("features"); f.add_argument("--force", action="store_true"); f.set_defaults(fn=cmd_features)
    v = sub.add_parser("validate"); v.add_argument("--skip-gnn", action="store_true"); v.add_argument("--force", action="store_true")
    v.add_argument("--stage", default="all", choices=["all", "lgbm", "diagnostics", "ablation", "gnn", "merge"]); v.set_defaults(fn=cmd_validate)
    t = sub.add_parser("train"); t.add_argument("--final", action="store_true"); t.set_defaults(fn=cmd_train)
    pr = sub.add_parser("predict"); pr.add_argument("--output", default=str(SUBMISSION / "predictions.csv")); pr.set_defaults(fn=cmd_predict)
    ve = sub.add_parser("verify"); ve.add_argument("--file", required=True); ve.set_defaults(fn=cmd_verify)
    sub.add_parser("explain").set_defaults(fn=cmd_explain)
    sub.add_parser("package").set_defaults(fn=cmd_package)
    rp = sub.add_parser("report"); rp.add_argument("--no-tests", action="store_true"); rp.set_defaults(fn=cmd_report)
    a = sub.add_parser("all"); a.add_argument("--force", action="store_true"); a.add_argument("--skip-gnn", action="store_true")
    a.add_argument("--stage", default="all"); a.set_defaults(fn=cmd_all)
    args = p.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    args.fn(args)
    log("done")


if __name__ == "__main__":
    main()
