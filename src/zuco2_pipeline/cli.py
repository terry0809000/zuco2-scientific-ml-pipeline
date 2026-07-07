from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, write_config_template
from .io import discover_local_zuco_files
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ZuCo 2.0 scientific ML pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run extraction, feature generation, and model evaluation.")
    run.add_argument("--config", default="configs/default.yaml", help="Path to a YAML configuration file.")

    discover = subparsers.add_parser("discover", help="List local ZuCo files under a data directory.")
    discover.add_argument("--data-dir", default="data", help="Directory to scan for results*_*.mat files.")

    init = subparsers.add_parser("init-config", help="Write a default configuration template.")
    init.add_argument("path", help="Destination YAML path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(args.config)
        outputs = run_pipeline(config)
        print("Pipeline completed.")
        for label, path in outputs.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "discover":
        files = discover_local_zuco_files([Path(args.data_dir)])
        if files.empty:
            print("No results*_*.mat files found.")
        else:
            printable = files.copy()
            printable["path"] = printable["path"].astype(str)
            print(printable.to_string(index=False))
        return 0

    if args.command == "init-config":
        write_config_template(args.path)
        print(f"Wrote {args.path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
