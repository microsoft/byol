# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CLI for the BYOL pipeline runner.

Usage::

    python -m byol.pipeline status --tgt-lang gug
    python -m byol.pipeline run --tgt-lang gug --step data-prep-cpt
    python -m byol.pipeline run --tgt-lang gug  # runs next step
    python -m byol.pipeline run-all --tgt-lang gug --device 3
    python -m byol.pipeline run-all --tgt-lang gug --max-samples 10  # test mode
"""

from __future__ import annotations

import argparse
import logging
import sys

from .runner import PipelineRunner


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byol-pipeline",
        description="BYOL Pipeline — guided end-to-end workflow for new languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status --tgt-lang gug
  %(prog)s run --tgt-lang gug                     # run next pending step
  %(prog)s run --tgt-lang gug --step data-prep-cpt
  %(prog)s run-all --tgt-lang gug --device 3
  %(prog)s run-all --tgt-lang gug --max-samples 10  # quick test
  %(prog)s clean --tgt-lang gug                   # remove all artifacts
  %(prog)s clean --tgt-lang gug --yes             # skip confirmation
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Shared args
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tgt-lang", required=True, help="Target language ISO-3 code (e.g. gug, nya, mri)")
        p.add_argument("--device", default="0", help="GPU device ID (default: 0)")
        p.add_argument("--model", default="google/gemma-3-4b-pt", help="Base pretrained model (G_PT)")
        p.add_argument("--instruct-model", default=None, help="Instruct model (G_IT) for merging (default: auto from --model)")
        p.add_argument("--max-samples", type=int, default=None, help="Limit samples per step (test mode)")
        p.add_argument("--translators", default="", help="Comma-separated translator models for LRA")
        p.add_argument("--llms", default="", help="Comma-separated LLMs for LRA")

    # status
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    add_common(status_parser)

    # run (single step or next)
    run_parser = subparsers.add_parser("run", help="Run a single step (or next pending)")
    add_common(run_parser)
    run_parser.add_argument("--step", default=None, help="Step ID to run (default: next pending)")

    # run-all
    run_all_parser = subparsers.add_parser("run-all", help="Run all pending steps")
    add_common(run_all_parser)
    run_all_parser.add_argument("--include-optional", action="store_true", default=False,
                                help="Include optional steps like find-best-llm (default: skip)")
    run_all_parser.add_argument("--quick-test", action="store_true", default=False,
                                help="Use reduced training settings (epochs=1, batch=1) for quick validation")

    # clean
    clean_parser = subparsers.add_parser("clean", help="Remove all artifacts for a language")
    clean_parser.add_argument("--tgt-lang", required=True, help="Target language ISO-3 code")
    clean_parser.add_argument("--device", default="0", help=argparse.SUPPRESS)
    clean_parser.add_argument("--model", default="google/gemma-3-4b-pt", help=argparse.SUPPRESS)
    clean_parser.add_argument("--max-samples", type=int, default=None, help=argparse.SUPPRESS)
    clean_parser.add_argument("--translators", default="", help=argparse.SUPPRESS)
    clean_parser.add_argument("--llms", default="", help=argparse.SUPPRESS)
    clean_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    return parser


def _make_runner(args: argparse.Namespace) -> PipelineRunner:
    return PipelineRunner(
        tgt_lang=args.tgt_lang,
        device=args.device,
        max_samples=args.max_samples,
        model=args.model,
        instruct_model=args.instruct_model,
        quick_test=getattr(args, "quick_test", False),
        translators=getattr(args, "translators", ""),
        llms=getattr(args, "llms", ""),
    )


def main(args: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = create_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    runner = _make_runner(parsed)

    if parsed.command == "status":
        runner.print_status()
        return 0

    if parsed.command == "run":
        if parsed.step:
            result = runner.run_step(parsed.step)
        else:
            result = runner.run_next()
        if result is None:
            return 0
        return 0 if result.success else 1

    if parsed.command == "run-all":
        results = runner.run_all(skip_optional=not parsed.include_optional)
        if not results:
            print("\n  🎉 Nothing to do — all steps complete!")
            return 0
        failed = [r for r in results if not r.success]
        if failed:
            return 1
        # Print final status
        runner.print_status()
        return 0

    if parsed.command == "clean":
        ok = runner.clean(force=parsed.yes)
        return 0 if ok else 1

    parser.print_help()
    return 0
