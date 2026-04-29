# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CLI entry point for byol-data-prep command."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import CPTDataPrepConfig, EvalDataPrepConfig, SFTDataPrepConfig
from .constants import (
    DEFAULT_CONFIGS_DIR,
    DEFAULT_EVAL_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    LANG_NAMES,
    REPO_ROOT,
    SUPPORTED_STAGES,
)
from .cpt_data_prep_runner import CPTDataPrepRunner
from .eval_data_prep_runner import EvalDataPrepRunner
from .sft_data_prep_runner import SFTDataPrepRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("byol-data-prep")


def create_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="byol-data-prep",
        description="BYOL Data Preparation — download, refine, and translate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  # Full CPT pipeline for Chichewa
  python -m byol.data_prep --stage cpt --tgt-lang nya

  # Full SFT pipeline for Chichewa
  python -m byol.data_prep --stage sft --tgt-lang nya

  # CPT with config file
  python -m byol.data_prep --stage cpt --config configs/data_prep/cpt/nya.yaml

  # SFT with config file
  python -m byol.data_prep --stage sft --config configs/data_prep/sft/nya.yaml

  # Eval data prep (translate benchmarks)
  python -m byol.data_prep --stage eval --tgt-lang nya

  # Eval with config file
  python -m byol.data_prep --stage eval --config configs/data_prep/eval/nya.yaml

  # Eval with output directory and max samples
  python -m byol.data_prep --stage eval --tgt-lang nya --output-dir /tmp/eval --max-samples 10

  # Download-only CPT (no refinement / translation)
  python -m byol.data_prep --stage cpt --tgt-lang nya --no-refine-tgt-lang --no-refine-eng --no-translate

  # Download-only SFT (no translation)
  python -m byol.data_prep --stage sft --tgt-lang nya --no-translate-smoltalk2 --no-translate-aya

  # Dry run
  python -m byol.data_prep --stage cpt --tgt-lang nya --dry-run
        """,
    )

    # ── Required ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--stage", "-s",
        type=str,
        required=True,
        choices=SUPPORTED_STAGES,
        help="Data preparation stage: cpt, sft, or eval",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default=None,
        help="Target language ISO 639-3 code (e.g. nya, mri)",
    )

    # ── Config file ──────────────────────────────────────────────────────
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI flags)",
    )

    # ── Step toggles ─────────────────────────────────────────────────────
    step_group = parser.add_argument_group("Step toggles")
    step_group.add_argument(
        "--no-download-tgt-lang-fineweb2",
        dest="download_tgt_lang_fineweb2",
        action="store_false",
        default=True,
        help="Skip downloading FineWeb-2 target-language data",
    )
    step_group.add_argument(
        "--no-refine-tgt-lang",
        dest="refine_tgt_lang",
        action="store_false",
        default=True,
        help="Skip refining target-language text",
    )
    step_group.add_argument(
        "--no-download-eng-finewebedu",
        dest="download_eng_finewebedu",
        action="store_false",
        default=True,
        help="Skip downloading FineWeb-Edu English data",
    )
    step_group.add_argument(
        "--no-refine-eng",
        dest="refine_eng",
        action="store_false",
        default=True,
        help="Skip refining English text",
    )
    step_group.add_argument(
        "--no-translate",
        dest="translate_eng_to_tgt_lang",
        action="store_false",
        default=True,
        help="Skip translating English to target language (CPT)",
    )

    # ── SFT step toggles ─────────────────────────────────────────────────
    sft_step_group = parser.add_argument_group("SFT step toggles")
    sft_step_group.add_argument(
        "--no-download-smoltalk2",
        dest="download_smoltalk2",
        action="store_false",
        default=True,
        help="Skip downloading SmolTalk2 subsets (SFT)",
    )
    sft_step_group.add_argument(
        "--no-translate-smoltalk2",
        dest="translate_smoltalk2",
        action="store_false",
        default=True,
        help="Skip translating SmolTalk2 (SFT)",
    )
    sft_step_group.add_argument(
        "--no-download-aya",
        dest="download_aya",
        action="store_false",
        default=True,
        help="Skip downloading AYA dataset (SFT)",
    )
    sft_step_group.add_argument(
        "--no-translate-aya",
        dest="translate_aya",
        action="store_false",
        default=True,
        help="Skip translating AYA dataset (SFT)",
    )

    # ── Eval options ─────────────────────────────────────────────────────
    eval_group = parser.add_argument_group("Eval options")
    eval_group.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        default=None,
        help="Eval benchmarks to translate (default: all). "
        "e.g. --benchmarks copa ai2_arc_hard mmlu_lite",
    )
    eval_group.add_argument(
        "--translator", "--eval-translator",
        type=str,
        default=None,
        dest="eval_translator",
        help="Override default translator for all eval benchmarks "
        "(e.g. microsoft-translator, nllb-200-3.3b, gpt-5)",
    )
    eval_group.add_argument(
        "--eval-max-workers",
        type=int,
        default=None,
        help="Number of parallel translation workers for eval (default: 8)",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Override output base directory (default: {DEFAULT_OUTPUT_DIR}).",
    )

    # ── Sampling limit ────────────────────────────────────────────────
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit each step to at most N samples (for quick testing)",
    )
    # ── General ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU device for local translator models (e.g., 0 or 2,3). Only needed for eval stage with local models like NLLB.",
    )
    parser.add_argument(        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output data. By default, steps are skipped if output already exists.",
    )
    parser.add_argument(        "--dry-run",
        action="store_true",
        default=False,
        help="Print the plan without executing anything",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )

    return parser


def _build_cpt_config(args: argparse.Namespace) -> CPTDataPrepConfig:
    """Build a CPT config from CLI args, optionally layering a YAML file."""
    if args.config:
        config = CPTDataPrepConfig.from_yaml(args.config)
        if args.tgt_lang:
            config.tgt_lang_code = args.tgt_lang
            config.tgt_lang_name = LANG_NAMES.get(args.tgt_lang, args.tgt_lang)
    else:
        if not args.tgt_lang:
            raise SystemExit("Error: --tgt-lang is required when --config is not provided.")
        config = CPTDataPrepConfig(
            tgt_lang_code=args.tgt_lang,
            tgt_lang_name=LANG_NAMES.get(args.tgt_lang, args.tgt_lang),
        )

    for step_attr in (
        "download_tgt_lang_fineweb2", "refine_tgt_lang",
        "download_eng_finewebedu",
        "refine_eng", "translate_eng_to_tgt_lang",
    ):
        cli_val = getattr(args, step_attr)
        if not cli_val:
            setattr(config, step_attr, False)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.overwrite:
        config.overwrite = True

    return config


def _build_sft_config(args: argparse.Namespace) -> SFTDataPrepConfig:
    """Build an SFT config from CLI args, optionally layering a YAML file."""
    if args.config:
        config = SFTDataPrepConfig.from_yaml(args.config)
        if args.tgt_lang:
            config.tgt_lang_code = args.tgt_lang
            config.tgt_lang_name = LANG_NAMES.get(args.tgt_lang, args.tgt_lang)
    else:
        if not args.tgt_lang:
            raise SystemExit("Error: --tgt-lang is required when --config is not provided.")
        config = SFTDataPrepConfig(
            tgt_lang_code=args.tgt_lang,
            tgt_lang_name=LANG_NAMES.get(args.tgt_lang, args.tgt_lang),
        )

    # Apply SFT step-toggle overrides
    for step_attr in (
        "download_smoltalk2", "translate_smoltalk2",
        "download_aya", "translate_aya",
    ):
        cli_val = getattr(args, step_attr)
        if not cli_val:
            setattr(config, step_attr, False)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.overwrite:
        config.overwrite = True

    return config


def _build_eval_config(args: argparse.Namespace) -> EvalDataPrepConfig:
    """Build an Eval config from CLI args, optionally layering a YAML file."""
    if args.config:
        config = EvalDataPrepConfig.from_yaml(args.config)
        if args.tgt_lang:
            config.tgt_lang_code = args.tgt_lang
            config.tgt_lang_name = LANG_NAMES.get(args.tgt_lang, args.tgt_lang)
    else:
        if not args.tgt_lang:
            raise SystemExit("Error: --tgt-lang is required when --config is not provided.")

        # Build benchmark list from --benchmarks flag
        from .config import EvalBenchmarkConfig
        from .constants import EVAL_BENCHMARK_DEFAULTS

        benchmarks: list[EvalBenchmarkConfig] = []
        if args.benchmarks:
            for name in args.benchmarks:
                if name not in EVAL_BENCHMARK_DEFAULTS:
                    raise SystemExit(
                        f"Error: Unknown benchmark '{name}'. "
                        f"Available: {', '.join(EVAL_BENCHMARK_DEFAULTS.keys())}"
                    )
                bm_kwargs: dict = {"name": name}
                if args.eval_translator:
                    bm_kwargs["translator"] = args.eval_translator
                benchmarks.append(EvalBenchmarkConfig(**bm_kwargs))
        # If empty, EvalDataPrepConfig.__post_init__ enables all defaults.

        config = EvalDataPrepConfig(
            tgt_lang_code=args.tgt_lang,
            tgt_lang_name=LANG_NAMES.get(args.tgt_lang, args.tgt_lang),
            benchmarks=benchmarks,
        )

    # CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.overwrite:
        config.overwrite = True
    if args.eval_max_workers is not None:
        config.max_workers = args.eval_max_workers
    if args.eval_translator:
        # Validate the translator name exists in the registry
        from byol.translation_backends.registry import MODEL_REGISTRY
        if args.eval_translator not in MODEL_REGISTRY:
            from byol.common.exceptions import TranslatorNotFoundError
            raise TranslatorNotFoundError(args.eval_translator, available=list(MODEL_REGISTRY.keys()))
        # Override default AND all per-benchmark translators
        config.default_translator = args.eval_translator
        for bm in config.benchmarks:
            bm.translator = args.eval_translator
    if args.device is not None:
        config.device = args.device

    return config


def _maybe_save_config(config, stage: str, args: argparse.Namespace) -> None:
    """Save config YAML — create if new, update if CLI overrides were applied."""
    if args.config:
        return  # user supplied an explicit config — don't auto-generate
    if getattr(args, "dry_run", False):
        return  # dry run — no side effects
    config_path = REPO_ROOT / DEFAULT_CONFIGS_DIR / stage / f"{config.tgt_lang_code}.yaml"
    if config_path.exists():
        # Update existing config if CLI overrides changed it
        config.to_yaml(config_path)
        logger.info(f"Updated config: {config_path}")
    else:
        config.to_yaml(config_path)
        logger.info(f"Generated config: {config_path}")


def _warn_api_costs(stage: str, config) -> None:
    """Print a cost warning when using paid API translators."""
    _PAID_PREFIXES = ("gpt-", "deepseek", "claude")

    if stage == "eval":
        translators = set()
        for bm in config.benchmarks:
            if bm.enabled:
                t = bm.translator or config.default_translator
                translators.add(t)
        paid = [t for t in translators if any(t.lower().startswith(p) for p in _PAID_PREFIXES)]
        if paid:
            logger.warning(
                "⚠️  This will make API calls using paid translator(s): %s. "
                "Use --max-samples N to limit costs during testing.",
                ", ".join(paid),
            )
    elif stage in ("cpt", "sft"):
        translator = getattr(config, "translator", None) or getattr(config, "translate_model", None)
        if translator and any(translator.lower().startswith(p) for p in _PAID_PREFIXES):
            logger.warning(
                "⚠️  This will make API calls using %s. "
                "Use --max-samples N to limit costs during testing.",
                translator,
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = create_parser()
    args = parser.parse_args(argv)

    # Set CUDA_VISIBLE_DEVICES before any torch imports
    if args.device is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.stage == "cpt":
        try:
            config = _build_cpt_config(args)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 1
        _maybe_save_config(config, "cpt", args)
        _warn_api_costs("cpt", config)
        runner = CPTDataPrepRunner(config, dry_run=args.dry_run)
        result = runner.run()
    elif args.stage == "sft":
        try:
            config = _build_sft_config(args)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 1
        _maybe_save_config(config, "sft", args)
        _warn_api_costs("sft", config)
        runner = SFTDataPrepRunner(config, dry_run=args.dry_run)
        result = runner.run()
    elif args.stage == "eval":
        try:
            config = _build_eval_config(args)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 1
        _maybe_save_config(config, "eval", args)
        _warn_api_costs("eval", config)
        runner = EvalDataPrepRunner(config, dry_run=args.dry_run)
        result = runner.run()
    else:
        logger.error(f"Stage '{args.stage}' is not yet implemented.")
        return 1

    if not result.success:
        logger.error(f"Pipeline failed: {result.error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
