# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CLI entry point for byol-train command."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LoraConfig, TrainConfig
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIG_CPT,
    DEFAULT_CONFIG_DPO,
    DEFAULT_CONFIG_SFT,
    DEFAULT_CUTOFF_LEN,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    SUPPORTED_STAGES,
)
from .runner import TrainingRunner

# Configure logging once at CLI entry
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("byol-train")


def parse_overrides(overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Parse key=value override arguments into typed dictionary."""
    if not overrides:
        return {}

    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            logger.warning(f"Ignoring invalid override (no '='): {item}")
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Type inference
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        elif value.isdigit():
            result[key] = int(value)
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value

    return result


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="byol-train",
        description="BYOL Training Framework - LlamaFactory wrapper with best practices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # Disable prefix matching - require exact argument names
        epilog="""
Examples:
  # Train with config file
  byol-train sft --config train_config.yaml

  # Train with CLI arguments
  byol-train sft --model meta-llama/Llama-3-8B --dataset alpaca --epochs 3

  # Merge LoRA adapter
  byol-train merge --base-model ./model --adapter ./lora --output ./merged
        """,
    )
    subparsers = parser.add_subparsers(dest="stage", help="Training stage")

    # Common arguments for training stages
    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        """Add common training arguments to a subparser."""
        # Config file
        subparser.add_argument(
            "--config", "-c",
            type=str,
            help="Path to YAML configuration file",
        )

        # Model arguments
        model_group = subparser.add_argument_group("Model")
        model_group.add_argument(
            "--model", "-m",
            type=str,
            help="HuggingFace model ID or local path",
        )
        model_group.add_argument(
            "--name", "-n",
            type=str,
            help="Run name for logging and output directory",
        )
        model_group.add_argument(
            "--template",
            type=str,
            default=None,
            help="Chat template name (default: gemma)",
        )

        # Dataset arguments
        data_group = subparser.add_argument_group("Dataset")
        data_group.add_argument(
            "--dataset", "-d",
            type=str,
            help="Dataset name or comma-separated list",
        )
        data_group.add_argument(
            "--eval-dataset",
            type=str,
            help="Evaluation dataset name (defaults to --dataset if not specified)",
        )
        data_group.add_argument(
            "--cutoff-len",
            type=int,
            default=None,
            help=f"Maximum sequence length (default: {DEFAULT_CUTOFF_LEN})",
        )

        # Training arguments
        train_group = subparser.add_argument_group("Training")
        train_group.add_argument(
            "--epochs", "-e",
            type=int,
            default=None,
            help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
        )
        train_group.add_argument(
            "--batch-size", "-b",
            type=int,
            default=None,
            help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE})",
        )
        train_group.add_argument(
            "--grad-accum",
            type=int,
            default=None,
            help=f"Gradient accumulation steps (default: {DEFAULT_GRADIENT_ACCUMULATION_STEPS})",
        )
        train_group.add_argument(
            "--lr", "--learning-rate",
            type=float,
            default=None,
            dest="learning_rate",
            help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
        )

        # LoRA arguments
        lora_group = subparser.add_argument_group("LoRA")
        lora_group.add_argument(
            "--lora",
            action="store_true",
            help="Enable LoRA fine-tuning",
        )
        lora_group.add_argument(
            "--lora-rank",
            type=int,
            default=DEFAULT_LORA_RANK,
            help=f"LoRA rank (default: {DEFAULT_LORA_RANK})",
        )
        lora_group.add_argument(
            "--lora-alpha",
            type=int,
            default=DEFAULT_LORA_ALPHA,
            help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})",
        )
        lora_group.add_argument(
            "--lora-dropout",
            type=float,
            default=DEFAULT_LORA_DROPOUT,
            help=f"LoRA dropout (default: {DEFAULT_LORA_DROPOUT})",
        )

        # Hardware arguments
        hw_group = subparser.add_argument_group("Hardware")
        hw_group.add_argument(
            "--device", "--gpus", "-g",
            dest="gpus",
            type=str,
            default=None,
            help="Comma-separated GPU device IDs (default: 0)",
        )
        hw_group.add_argument(
            "--bf16/--no-bf16",
            dest="bf16",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use bfloat16 precision (default: True)",
        )

        # Output arguments
        output_group = subparser.add_argument_group("Output")
        output_group.add_argument(
            "--tgt-lang",
            type=str,
            default=None,
            help="Target language code. When set, outputs go to results/<lang>/train/<stage>/",
        )
        output_group.add_argument(
            "--output-dir", "-o",
            type=str,
            default=None,
            help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        output_group.add_argument(
            "--wandb-project",
            type=str,
            help="W&B project name for logging",
        )
        output_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Print config without running training",
        )

        # Overrides
        subparser.add_argument(
            "--override",
            type=str,
            nargs="*",
            metavar="KEY=VALUE",
            help="Override config values (e.g., --override epochs=10 lr=1e-5)",
        )

    # CPT subcommand
    cpt_parser = subparsers.add_parser(
        "cpt",
        help="Continual Pre-Training",
        description="Run continual pre-training on unlabeled text data",
        allow_abbrev=False,
    )
    add_common_args(cpt_parser)

    # SFT subcommand
    sft_parser = subparsers.add_parser(
        "sft",
        help="Supervised Fine-Tuning",
        description="Run supervised fine-tuning on instruction data",
        allow_abbrev=False,
    )
    add_common_args(sft_parser)

    # DPO subcommand
    dpo_parser = subparsers.add_parser(
        "dpo",
        help="Direct Preference Optimization",
        description="Run DPO training on preference data",
        allow_abbrev=False,
    )
    add_common_args(dpo_parser)

    # Merge subcommand
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge LoRA adapter into base model",
        description="Merge a trained LoRA adapter into the base model",
        allow_abbrev=False,
    )
    merge_parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model",
    )
    merge_parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter",
    )
    merge_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    merge_parser.add_argument(
        "--template",
        type=str,
        default="gemma",
        help="Chat template name (default: gemma)",
    )
    merge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running merge",
    )

    return parser


def run_training(args: argparse.Namespace) -> int:
    """Run training job from CLI arguments. Returns exit code."""
    # Determine config file path
    config_path = args.config
    if not config_path:
        # Try to load default config for the stage
        default_configs = {
            "cpt": DEFAULT_CONFIG_CPT,
            "sft": DEFAULT_CONFIG_SFT,
            "dpo": DEFAULT_CONFIG_DPO,
        }
        default_config = default_configs.get(args.stage)
        if default_config:
            # Look for config relative to the repo root
            default_path = REPO_ROOT / default_config
            if default_path.exists():
                config_path = str(default_path)
                logger.info(f"Using default config: {config_path}")
    
    # Load from config file or build from CLI args
    if config_path:
        config = TrainConfig.from_yaml(config_path)
        # Override stage from subcommand
        config.stage = args.stage
        
        # Only override YAML values with CLI args the user explicitly provided
        if args.model:
            config.model_name_or_path = args.model
        if args.dataset:
            config.dataset = args.dataset
        # Set eval_dataset: CLI arg > default to train dataset
        if args.eval_dataset:
            config.eval_dataset = args.eval_dataset
        elif args.dataset:
            # Default eval_dataset to train dataset if not specified
            config.eval_dataset = args.dataset
        # Only apply these when the user explicitly passed them (not None)
        if args.gpus is not None:
            config.gpus = args.gpus
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.grad_accum is not None:
            config.gradient_accumulation_steps = args.grad_accum
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
        if args.cutoff_len is not None:
            config.cutoff_len = args.cutoff_len
        if args.template is not None:
            config.template = args.template
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.wandb_project:
            config.wandb_project = args.wandb_project
        if args.lora:
            config.lora = LoraConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
    else:
        if not args.model:
            logger.error("Either --config or --model must be specified")
            return 1

        # Build LoRA config if enabled
        lora = None
        if args.lora:
            lora = LoraConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )

        config = TrainConfig(
            model_name_or_path=args.model,
            stage=args.stage,
            dataset=args.dataset or "",
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            epochs=args.epochs or DEFAULT_EPOCHS,
            batch_size=args.batch_size or DEFAULT_BATCH_SIZE,
            gradient_accumulation_steps=args.grad_accum or DEFAULT_GRADIENT_ACCUMULATION_STEPS,
            learning_rate=args.learning_rate or DEFAULT_LEARNING_RATE,
            cutoff_len=args.cutoff_len or DEFAULT_CUTOFF_LEN,
            template=args.template or "gemma",
            gpus=args.gpus or "0",
            bf16=args.bf16,
            lora=lora,
            wandb_project=args.wandb_project,
        )

    # Override output_dir when --tgt-lang is provided
    if args.tgt_lang:
        stage_dir = "cpt" if config.stage in ("pt", "cpt") else config.stage
        config.output_dir = f"results/{args.tgt_lang}/train/{stage_dir}"

        # Auto-resolve dataset from tgt_lang if not explicitly provided
        if not args.dataset:
            if config.stage in ("pt", "cpt"):
                config.dataset = f"{args.tgt_lang}_english_cpt"
            elif config.stage == "sft":
                config.dataset = f"{args.tgt_lang}_english_sft"
                if not args.eval_dataset:
                    config.eval_dataset = f"{args.tgt_lang}_sft_test"

        # Auto-resolve dataset_dir if not in overrides
        overrides_raw = parse_overrides(args.override)
        if not any(k.startswith("dataset_dir") for k in overrides_raw):
            base = Path(os.environ.get("BYOL_DATA_DIR", Path("~/byol-data").expanduser()))
            if config.stage in ("pt", "cpt"):
                if config.passthrough_args is None:
                    config.passthrough_args = {}
                config.passthrough_args["dataset_dir"] = str(
                    base / args.tgt_lang / "cpt" / "bilingual_mix"
                )
            elif config.stage == "sft":
                if config.passthrough_args is None:
                    config.passthrough_args = {}
                config.passthrough_args["dataset_dir"] = str(
                    base / args.tgt_lang / "sft" / "bilingual_mix"
                )

    # Apply overrides
    overrides = parse_overrides(args.override)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Override: {key}={value}")
        elif config.passthrough_args and key in config.passthrough_args:
            config.passthrough_args[key] = value
            logger.info(f"Override: {key}={value}")
        else:
            # Add to passthrough_args for LlamaFactory
            if config.passthrough_args is None:
                config.passthrough_args = {}
            config.passthrough_args[key] = value
            logger.info(f"Override (passthrough): {key}={value}")

    # Run training
    runner = TrainingRunner(config, dry_run=args.dry_run)
    result = runner.run()

    return 0 if result.success else 1


def run_merge(args: argparse.Namespace) -> int:
    """Run LoRA merge from CLI arguments. Returns exit code."""
    from .merge import merge_lora

    success = merge_lora(
        base_model=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output,
        template=args.template,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = create_parser()
    parsed = parser.parse_args(args)

    if not parsed.stage:
        parser.print_help()
        return 1

    try:
        if parsed.stage == "merge":
            exit_code = run_merge(parsed)
        else:
            exit_code = run_training(parsed)

        if exit_code == 0:
            logger.info("")
            logger.info(
                "ℹ️  Next step: evaluate with  "
                "python -m byol.eval --model <output_dir> --type base --tgt-lang <lang>"
            )

        return exit_code

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
