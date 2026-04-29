# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Command-line interface for BYOL Evaluation Framework.

Provides a CLI for running model evaluations using lm-evaluation-harness
and LLM-as-Judge frameworks.

Usage:
    byol-eval --model google/gemma-3-4b-pt --type base --tgt-lang mri --gpus 0
    byol-eval judge --model-config <config> --dataset-config <config>
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from .config import EvalConfig, ModelConfig, TaskConfig
from .constants import (
    CONFIG_FILENAME_TEMPLATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIGS_DIR,
    DEFAULT_DTYPE,
    DEFAULT_GPUS,
    DEFAULT_JUDGE_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    KNOWN_LANGS,
    VALID_DTYPES,
    VALID_TYPES,
)
from .runner import EvaluationRunner

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments.  Uses ``sys.argv`` if ``None``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="byol-eval",
        description="BYOL Model Evaluation Framework — Evaluate LLMs using lm-eval or LLM-as-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model google/gemma-3-4b-pt --type base --tgt-lang mri --device 0
  %(prog)s --model google/gemma-3-4b-it --type instruct --tgt-lang eng --dry-run
  %(prog)s judge --model-config configs/eval/judge_models.yaml --dataset-config configs/eval/judge_datasets.yaml
        """,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # ── Judge subcommand ─────────────────────────────────────────────────
    judge_parser = subparsers.add_parser("judge", help="Run LLM-as-Judge evaluation")
    judge_parser.add_argument(
        "--model-config", "-m", type=str,
        help="Path to model configuration YAML file",
    )
    judge_parser.add_argument(
        "--dataset-config", "-d", type=str,
        help="Path to dataset configuration YAML file",
    )
    judge_parser.add_argument(
        "--output-dir", "-o", type=str, default=DEFAULT_JUDGE_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_JUDGE_OUTPUT_DIR})",
    )

    # ── Add-language subcommand ──────────────────────────────────────────
    add_lang_parser = subparsers.add_parser(
        "add-language",
        help="Generate eval task YAMLs and benchmark configs for a new language",
    )
    add_lang_parser.add_argument(
        "--lang", required=True, type=str,
        help="ISO 639-3 language code (e.g. gug, nya, mri)",
    )
    add_lang_parser.add_argument(
        "--name", required=True, type=str,
        help="Language name (e.g. Guarani, Chichewa, Maori)",
    )
    add_lang_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data subdirectory name under data/eval/ (default: lowercase --name)",
    )
    add_lang_parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be created without writing files",
    )

    # ── Benchmark mode (default) ────────────────────────────────────────
    parser.add_argument(
        "--model", "-m", type=str,
        help="Model path or HuggingFace ID to evaluate (required for benchmark mode)",
    )
    parser.add_argument(
        "--type", type=str, choices=sorted(VALID_TYPES),
        help="Evaluation type: 'base' (few-shot) or 'instruct' (0-shot + chat template)",
    )
    parser.add_argument(
        "--tgt-lang", type=str,
        help="Target language code, e.g. eng, mri, nya, gug (required for benchmark mode)",
    )

    # Optional arguments
    parser.add_argument(
        "--config", "-c", type=str,
        help="Path to YAML configuration file (overrides --type/--tgt-lang)",
    )
    parser.add_argument("--model-name", type=str, help="Human-readable model name")
    parser.add_argument(
        "--dtype", type=str, default=DEFAULT_DTYPE, choices=list(VALID_DTYPES),
        help=f"Model data type (default: {DEFAULT_DTYPE})",
    )
    parser.add_argument(
        "--tasks", "-t", type=str, default="all",
        help="Comma-separated task names or 'all' (default: all)",
    )
    parser.add_argument("--tasks-path", type=str, help="Path to custom task definitions directory")
    parser.add_argument(
        "--num-fewshot", "-n", type=int, default=None,
        help="Number of few-shot examples (default: auto from config registry)",
    )
    parser.add_argument("--limit", type=int, help="Maximum samples per task")
    parser.add_argument(
        "--device", "--gpus", "-g", dest="gpus", type=str, default=DEFAULT_GPUS,
        help=f"Comma-separated GPU device IDs (default: {DEFAULT_GPUS})",
    )
    parser.add_argument(
        "--batch-size", "-b", type=str, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size or 'auto:N' (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without executing")
    parser.add_argument(
        "--skip", action="store_true",
        help="Skip tasks that already have completed outputs",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run evaluations even if results already exist",
    )
    parser.add_argument("--log-samples", action="store_true", help="Log individual samples")
    parser.add_argument(
        "--data-dir", type=str, default="",
        help="Path to a folder of translated eval JSONL files (overrides default data paths)",
    )
    parser.add_argument(
        "--apply-chat-template", action="store_true",
        help="Apply chat template for instruct models",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity (-v for DEBUG)",
    )

    return parser.parse_args(args)


# =============================================================================
# Argument Validation
# =============================================================================


def validate_args(parsed: argparse.Namespace) -> None:
    """Validate parsed arguments for benchmark mode.

    Args:
        parsed: Parsed CLI arguments.

    Raises:
        ValueError: If required arguments are missing.
    """
    missing: list[str] = []
    if not parsed.model:
        missing.append("--model")
    if not parsed.type:
        missing.append("--type")
    if not parsed.tgt_lang:
        missing.append("--tgt-lang")
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


# =============================================================================
# Config Resolution
# =============================================================================


def _resolve_config_path(eval_type: str, lang: str) -> str:
    """Resolve a config file path from ``--type`` and ``--tgt-lang``.

    Searches for ``configs/eval/benchmark_{type}_{lang}.yaml`` relative to the
    package directory, then ``cwd``.  For ``merged`` type, falls back to the
    ``instruct`` config since merged models use the same benchmarks.

    Args:
        eval_type: ``'base'``, ``'instruct'``, or ``'merged'``.
        lang: Language code (e.g. ``'eng'``, ``'mri'``, ``'nya'``, ``'gug'``).

    Returns:
        Resolved path to the config YAML file.

    Raises:
        FileNotFoundError: If no matching config file is found.
    """
    # merged uses the same benchmarks as instruct
    types_to_try = [eval_type]
    if eval_type == "merged":
        types_to_try.append("instruct")

    for try_type in types_to_try:
        filename = CONFIG_FILENAME_TEMPLATE.format(type=try_type, lang=lang)
        pkg_dir = Path(__file__).resolve().parent  # byol/eval/
        candidates = [
            pkg_dir.parent.parent / DEFAULT_CONFIGS_DIR / filename,  # repo/configs/eval/
            Path.cwd() / DEFAULT_CONFIGS_DIR / filename,              # cwd/configs/eval/
            Path.cwd() / "configs" / filename,                        # legacy flat configs/
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    hint = ""
    if lang not in KNOWN_LANGS:
        hint = (
            f"\n\n  Hint: Language '{lang}' does not have pre-built eval configs.\n"
            f"  Generate them with:\n"
            f"    python -m byol.eval add-language --lang {lang} --name <LanguageName>\n"
        )
    raise FileNotFoundError(
        f"Config file not found: {filename}. "
        f"Searched: {[str(c) for c in candidates]}{hint}"
    )


def build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    """Build :class:`EvalConfig` from CLI arguments.

    Supports two modes:
        1. Explicit ``--config`` path (overrides ``--type``/``--tgt-lang``).
        2. Auto-resolve from ``--type`` and ``--tgt-lang``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured ``EvalConfig`` instance.

    Raises:
        ValueError: If required arguments are missing or tasks don't match.
    """
    config_path = args.config or _resolve_config_path(args.type, args.tgt_lang)
    config = EvalConfig.from_yaml(config_path)

    # CLI overrides
    config.gpus = args.gpus
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.tasks_path:
        config.tasks_path = args.tasks_path
    if args.log_samples:
        config.log_samples = True
    if hasattr(args, "apply_chat_template") and args.apply_chat_template:
        config.apply_chat_template = True

    # Override model from CLI
    config.models = [
        ModelConfig(
            name=args.model_name or Path(args.model).name,
            path=args.model,
            dtype=args.dtype or "bfloat16",
        )
    ]

    # ── Filter tasks when --tasks is not "all" ──────────────────────────
    if args.tasks and args.tasks != "all":
        requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
        filtered: list[TaskConfig] = []
        for tc in config.tasks:
            tc_names = {n.strip() for n in tc.name.split(",")}
            matching = tc_names & set(requested)
            if matching:
                filtered.append(
                    TaskConfig(
                        name=",".join(sorted(matching)),
                        num_fewshot=tc.num_fewshot,
                        limit=tc.limit,
                        batch_size=tc.batch_size,
                        apply_chat_template=tc.apply_chat_template,
                    )
                )
            else:
                # Fall back to substring match
                for req in requested:
                    if any(req in tn for tn in tc_names):
                        filtered.append(
                            TaskConfig(
                                name=tc.name,
                                num_fewshot=tc.num_fewshot,
                                limit=tc.limit,
                                batch_size=tc.batch_size,
                                apply_chat_template=tc.apply_chat_template,
                            )
                        )
                        break
        if not filtered:
            available = [tc.name for tc in config.tasks]
            raise ValueError(
                f"No matching tasks found for '{args.tasks}'. Available tasks: {available}"
            )
        config.tasks = filtered

    return config


# =============================================================================
# Judge Sub-command
# =============================================================================


def run_judge(args: argparse.Namespace) -> int:
    """Run LLM-as-Judge evaluation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from .judge import LLMJudgeRunner

    try:
        runner = LLMJudgeRunner(args.model_config, args.dataset_config, args.output_dir)
        runner.run()
        return 0
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Judge evaluation failed: {e}")
        return 1


# =============================================================================
# Main Entry Point
# =============================================================================


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments.  Uses ``sys.argv`` if ``None``.

    Returns:
        Exit code (0 for success, 1 for failure, 130 for keyboard interrupt).
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        parsed = parse_args(args)

        # ── Configure logging ────────────────────────────────────────────
        level = logging.DEBUG if parsed.verbose > 0 else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
        )

        # ── Judge sub-command ────────────────────────────────────────────
        if parsed.mode == "judge":
            return run_judge(parsed)

        # ── Add-language sub-command ─────────────────────────────────────
        if parsed.mode == "add-language":
            from .add_language import run_add_language
            return run_add_language(parsed)

        # ── Benchmark mode ───────────────────────────────────────────────
        validate_args(parsed)

        config = build_config_from_args(parsed)
        config_path = parsed.config or _resolve_config_path(parsed.type, parsed.tgt_lang)

        runner = EvaluationRunner(
            config,
            dry_run=parsed.dry_run,
            skip_existing=parsed.skip,
            overwrite=parsed.overwrite,
            eval_type=parsed.type,
            tgt_lang=parsed.tgt_lang,
            config_path=config_path,
            data_dir=parsed.data_dir,
        )
        results = runner.run_all()
        runner.print_summary(results)
        failed = sum(1 for r in results if r.status == "failed")
        passed = sum(1 for r in results if r.status == "success")
        if failed > 0 and passed == 0:
            return 1  # Total failure
        return 0  # Partial or full success

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        if hasattr(parsed, "verbose") and parsed.verbose > 0:  # type: ignore[possibly-undefined]
            traceback.print_exc()
        return 1
