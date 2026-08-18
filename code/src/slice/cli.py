from __future__ import annotations

import argparse
from pathlib import Path

from .etl import build_features
from .freeze import run_cli as run_freeze_cli
from .handcode import export_handcode_pack
from .judge import score_all
from .kappa_gate import run_cli as run_kappa_cli
from .metrics import compute_metrics
from .red_team import run_red_team_probe_from_config
from .runner import run_all
from .council import build_council_cases, run_council


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m slice.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "score", "etl", "metrics", "handcode-export"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="config.yaml")
        if name == "run":
            sub.add_argument("--retry-missing", action="store_true")
        if name == "score":
            sub.add_argument("--tier", choices=["auto", "council", "cheap_panel"], default="auto")
    council = subparsers.add_parser("council")
    council.add_argument("--config", default="config.yaml")
    council.add_argument("--build-cases", action="store_true")
    kappa = subparsers.add_parser("kappa")
    kappa.add_argument("--config", default="config.yaml")
    kappa.add_argument("--template")
    kappa.add_argument("--emit-gate")
    kappa.add_argument("--demote", action="append", default=[])
    kappa.add_argument("--dry-run-council", action="store_true")
    kappa.add_argument("--final-test")
    kappa.add_argument("--gate")
    kappa.add_argument("--calibration-gate")
    red_team = subparsers.add_parser("red-team")
    red_team.add_argument("--config", default="config.yaml")

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", default="config.yaml")
    freeze.add_argument("--date-stamp")
    freeze.add_argument("--verify", action="store_true")
    freeze.add_argument("--freeze-calibration", action="store_true")
    freeze.add_argument("--freeze-preflight", action="store_true")
    freeze.add_argument("--add-deviation", nargs=3, metavar=("FILE", "WHAT_CHANGED", "WHY"))
    freeze.add_argument("--external-timestamp-method")
    freeze.add_argument("--external-timestamp-reference")
    freeze.add_argument("--external-timestamp-instructions")
    freeze_calibration = subparsers.add_parser("freeze-calibration")
    freeze_calibration.add_argument("--config", default="config.yaml")
    freeze_preflight = subparsers.add_parser("freeze-preflight")
    freeze_preflight.add_argument("--config", default="config.yaml")

    args = parser.parse_args(argv)
    config = Path(args.config)
    if args.command == "run":
        print(run_all(config, retry_missing=args.retry_missing))
    elif args.command == "score":
        print(score_all(config, tier=None if args.tier == "auto" else args.tier))
    elif args.command == "council":
        print(build_council_cases(config) if args.build_cases else run_council(config))
    elif args.command == "etl":
        print(build_features(config))
    elif args.command == "metrics":
        print(compute_metrics(config))
    elif args.command == "handcode-export":
        print(export_handcode_pack(config))
    elif args.command == "kappa":
        kappa_args = ["--config", str(config)]
        if args.template:
            kappa_args.extend(["--template", args.template])
        if args.emit_gate:
            kappa_args.extend(["--emit-gate", args.emit_gate])
        for demotion in args.demote:
            kappa_args.extend(["--demote", demotion])
        if args.dry_run_council:
            kappa_args.append("--dry-run-council")
        if args.final_test:
            kappa_args.extend(["--final-test", args.final_test])
        if args.gate:
            kappa_args.extend(["--gate", args.gate])
        if args.calibration_gate:
            kappa_args.extend(["--calibration-gate", args.calibration_gate])
        print(run_kappa_cli(kappa_args))
    elif args.command == "red-team":
        print(run_red_team_probe_from_config(config))
    elif args.command == "freeze":
        freeze_args = ["--config", str(config)]
        if args.date_stamp:
            freeze_args.extend(["--date-stamp", args.date_stamp])
        if args.verify:
            freeze_args.append("--verify")
        if args.freeze_calibration:
            freeze_args.append("--freeze-calibration")
        if args.freeze_preflight:
            freeze_args.append("--freeze-preflight")
        if args.add_deviation:
            freeze_args.extend(["--add-deviation", *args.add_deviation])
        if args.external_timestamp_method:
            freeze_args.extend(["--external-timestamp-method", args.external_timestamp_method])
        if args.external_timestamp_reference:
            freeze_args.extend(["--external-timestamp-reference", args.external_timestamp_reference])
        if args.external_timestamp_instructions:
            freeze_args.extend(["--external-timestamp-instructions", args.external_timestamp_instructions])
        print(run_freeze_cli(freeze_args))
    elif args.command == "freeze-calibration":
        print(run_freeze_cli(["--config", str(config), "--freeze-calibration"]))
    elif args.command == "freeze-preflight":
        print(run_freeze_cli(["--config", str(config), "--freeze-preflight"]))


if __name__ == "__main__":
    main()
