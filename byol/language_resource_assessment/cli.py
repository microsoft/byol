# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Command-line interface for language resource assessment.

This module provides the CLI with task dispatch pattern:
- find-best-translator: Evaluate translators via round-trip translation
- find-best-llm: Evaluate LLMs for finetuning (planned)
- language-digital-presence: Analyze online presence (planned)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="byol-language-resource-assessment",
        description=(
            "Evaluate translation quality for low-resource languages via round-trip translation. "
            "Compare different translators (API and local models) to find the best one for your language."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a language by ISO-3 code
  python -m byol.language_resource_assessment --task language-classification --tgt-lang nya

  # Classify a language by name (may show multiple matches if ambiguous)
  python -m byol.language_resource_assessment --task language-classification --tgt-lang Chichewa

  # Basic usage - find best translator for Chichewa
  python -m byol.language_resource_assessment --task find-best-translator --tgt-lang Chichewa

  # Use specific translators
  python -m byol.language_resource_assessment --task find-best-translator --tgt-lang Chichewa --translators gpt4o,nllb-200-3.3B

  # Run only translation step
  python -m byol.language_resource_assessment --task find-best-translator --tgt-lang Maori --step translate

  # Run on custom dataset
  python -m byol.language_resource_assessment --task find-best-translator --tgt-lang Guarani --dataset ./my_data.jsonl

  # List all available translators
  python -m byol.language_resource_assessment --list-translators

Available Tasks:
  find-best-translator       Evaluate translators via round-trip translation (ready)
  find-best-llm              Evaluate LLMs for finetuning (planned)
  language-classification    Classify language resource level (ready)
  language-digital-presence  Analyze online presence of a language (alias for language-classification)
        """,
    )
    
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["find-best-translator", "find-best-llm", "language-classification", "language-digital-presence"],
        default="find-best-translator",
        help="Task to run (default: find-best-translator)",
    )
    
    # Required arguments
    parser.add_argument(
        "--tgt-lang",
        type=str,
        help="Target language name (e.g., Chichewa, Maori, Guarani)",
    )
    
    # Optional - Language
    parser.add_argument(
        "--src-lang",
        type=str,
        default="English",
        help="Source language (default: English)",
    )
    
    # Optional - Translators/Models
    parser.add_argument(
        "--translators",
        type=str,
        default=None,
        help="Comma-separated translator names for find-best-translator task (default: configured defaults)",
    )
    parser.add_argument(
        "--llms",
        type=str,
        default=None,
        help="Comma-separated LLM names for find-best-llm task (default: configured defaults)",
    )
    parser.add_argument(
        "--list-translators",
        action="store_true",
        help="List all available translators/LLMs and exit",
    )
    
    # Optional - Data paths
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to input JSONL dataset (default: built-in RTTBench-Mono)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: ./results/rttbench/<src>_<tgt>)",
    )
    
    # Optional - Execution control
    parser.add_argument(
        "--step",
        choices=["all", "translate", "metrics", "plot"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device(s): single (0), comma-separated for parallel local models (2,3), or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=["openai", "qwen"],
        default="openai",
        help="Embedding model for similarity computation: 'openai' (Azure text-embedding-3-large) or 'qwen' (Qwen3-Embedding-8B) (default: openai)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local models (default: 8)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help="Max concurrent API requests (default: 32)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all). Use for quick prototyping, e.g., --max-samples 16",
    )
    
    # Optional - Flags
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip translators that already have complete results (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute all translations (overrides --skip-existing)",
    )
    parser.add_argument(
        "--exclude-translators",
        type=str,
        default=None,
        help="Comma-separated translators to exclude from plots",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity",
    )
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    # --tgt-lang is required unless --list-translators
    if not args.list_translators and not args.tgt_lang:
        raise ValueError("--tgt-lang is required unless using --list-translators")
    
    # Check dataset file exists if specified
    if args.dataset and not args.dataset.exists():
        raise ValueError(f"Dataset file not found: {args.dataset}")


def get_config_file_for_task(task: str) -> str:
    """
    Get the appropriate config file name for a task.
    
    Args:
        task: The task name (find-best-translator, find-best-llm, etc.)
        
    Returns:
        Config file name to load
    """
    if task == "find-best-llm":
        return "llms.yaml"
    else:
        # Default to translators.yaml for find-best-translator and others
        return "translators.yaml"


def get_selected_models(args: argparse.Namespace, config: dict) -> list[str]:
    """
    Get list of selected models based on args and config.
    
    Uses --llms for find-best-llm task, --translators for find-best-translator task.
    
    Args:
        args: Parsed command-line arguments
        config: Loaded configuration
        
    Returns:
        List of model names to use
    """
    # Determine which flag to use based on task
    if args.task == "find-best-llm":
        user_selection = args.llms
    else:
        user_selection = args.translators
    
    if user_selection:
        # Parse user-specified models
        return [t.strip() for t in user_selection.split(",")]
    else:
        # Use defaults from config
        from .config import DEFAULT_TRANSLATORS
        return config.get("default_translators", DEFAULT_TRANSLATORS)


def get_excluded_models(args: argparse.Namespace, config: dict) -> list[str]:
    """
    Get list of models to exclude from plots.
    
    Args:
        args: Parsed command-line arguments
        config: Loaded configuration
        
    Returns:
        List of model names to exclude
    """
    excluded = config.get("excluded_from_plots", [])
    
    if args.exclude_translators:
        excluded.extend([t.strip() for t in args.exclude_translators.split(",")])
    
    return excluded


def main() -> int:
    """Main entry point for language resource assessment CLI."""
    from dotenv import load_dotenv
    load_dotenv()

    # CRITICAL: Parse args and set CUDA_VISIBLE_DEVICES BEFORE importing torch!
    # torch caches CUDA device info at import time, so this must happen first.
    args = parse_args()
    
    # Parse device specification: "0", "2,3", or "cpu"
    # NOTE: CUDA_VISIBLE_DEVICES remaps physical GPUs to logical indices (0, 1, ...).
    # So --device 2,3 → CUDA_VISIBLE_DEVICES=2,3 → models use cuda:0 and cuda:1.
    device_str = str(args.device).strip()
    if device_str.lower() == "cpu":
        args.device_list = ["cpu"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif "," in device_str:
        physical_devices = [d.strip() for d in device_str.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)
        # Remapped logical indices: 0, 1, 2, ...
        args.device_list = [str(i) for i in range(len(physical_devices))]
    else:
        args.device_list = [device_str]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    
    # ==========================================================================
    # FAST PATH: language-classification task is independent of translator config
    # ==========================================================================
    if args.task in ("language-classification", "language-digital-presence"):
        if not args.tgt_lang:
            print("Error: --tgt-lang is required for language-classification task", file=sys.stderr)
            return 1
        
        from .language_digital_presence import run_language_classification
        return run_language_classification(tgt_lang=args.tgt_lang)
    
    # ==========================================================================
    # TRANSLATOR/LLM TASKS: require config loading and validation
    # ==========================================================================
    
    # NOW import torch-dependent modules (after CUDA device is set)
    from .config import (
        load_config,
        get_translator_configs,
        get_canonical_language_name,
        DEFAULT_TRANSLATORS,
        get_data_dir,
    )
    
    # Load configuration based on task
    config_file = get_config_file_for_task(args.task)
    config = load_config(config_file)
    
    # Normalize language names to canonical form (handles case variations)
    # Keep the ISO-3 code for directory naming (consistent with pipeline runner)
    from .config import resolve_language_code
    tgt_iso3 = None
    src_iso3 = None
    if args.tgt_lang:
        try:
            tgt_iso3 = resolve_language_code(config, args.tgt_lang, "iso3")
            args.tgt_lang = get_canonical_language_name(config, args.tgt_lang)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    if args.src_lang:
        try:
            src_iso3 = resolve_language_code(config, args.src_lang, "iso3")
            args.src_lang = get_canonical_language_name(config, args.src_lang)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Handle --list-translators
    if args.list_translators:
        print("\n" + "=" * 70)
        print("Available Translators")
        print("=" * 70)
        
        api_translators = []
        local_translators = []
        
        for name, cfg in config.get("translators", {}).items():
            if cfg.get("type") == "local":
                local_translators.append((name, cfg))
            else:
                api_translators.append((name, cfg))
        
        print("\n[API-based Translators]")
        for name, cfg in sorted(api_translators):
            params = cfg.get("params", {})
            model = params.get("model_name", "")
            print(f"  {name:45} {model}")
        
        print("\n[Local Models]")
        for name, cfg in sorted(local_translators):
            params = cfg.get("params", {})
            model = params.get("model_name", "")
            print(f"  {name:45} {model}")
        
        print("\n[Default Translators]")
        defaults = config.get("default_translators", DEFAULT_TRANSLATORS)
        print(f"  {', '.join(defaults)}")
        
        print("\n[Supported Languages]")
        for lang in config.get("language_codes", {}).keys():
            print(f"  {lang}")
        
        print()
        return 0
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Get model selection
    selected = get_selected_models(args, config)
    if not selected:
        print("Error: No models selected", file=sys.stderr)
        return 1
    
    model_configs = get_translator_configs(
        config, selected, args.src_lang, args.tgt_lang
    )
    
    if not model_configs:
        print("Error: No valid model configurations found", file=sys.stderr)
        return 1
    
    # Resolve dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = get_data_dir() / "rttbench_mono_dataset" / "RTTBench-Mono.jsonl"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}", file=sys.stderr)
        print("Please specify --dataset or ensure data directory is configured", file=sys.stderr)
        return 1
    
    # Resolve output directory (task-specific)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use separate directories for each task type
        if args.task == "find-best-llm":
            task_subdir = "llm_comparison"
        elif args.task == "find-best-translator":
            task_subdir = "translator_comparison"
        else:
            task_subdir = args.task.replace("-", "_")
        
        output_dir = Path("./results") / (tgt_iso3 or args.tgt_lang.lower()) / "lra" / task_subdir
    
    # Print banner
    print()
    print("=" * 70)
    print("BYOL Language Resource Assessment")
    print("=" * 70)
    print(f"  Task:              {args.task}")
    print(f"  Config File:       {config_file}")
    print(f"  Source Language:   {args.src_lang}")
    print(f"  Target Language:   {args.tgt_lang}")
    print(f"  Models:            {', '.join(model_configs.keys())}")
    print(f"  Dataset:           {dataset_path}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  GPU Device(s):     {args.device}")
    print(f"  Embedding Model:   {args.embedding_model}")
    print(f"  Max Samples:       {args.max_samples if args.max_samples else 'all'}")
    print(f"  Pipeline Step:     {args.step}")
    print("=" * 70)
    print()
    
    # Dispatch to task
    try:
        if args.task in ("find-best-translator", "find-best-llm"):
            from .find_best_model import run_model_evaluation
            
            excluded = get_excluded_models(args, config)
            task_name = "translator" if args.task == "find-best-translator" else "LLM"
            
            run_model_evaluation(
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                model_configs=model_configs,
                dataset_path=dataset_path,
                output_dir=output_dir,
                step=args.step,
                batch_size=args.batch_size,
                max_concurrent=args.max_concurrent,
                max_samples=args.max_samples,
                skip_existing=args.skip_existing and not args.force,
                embedding_model=args.embedding_model,
                excluded_models=excluded,
                task_name=task_name,
                device_list=args.device_list,
            )
        
        print()
        print("=" * 70)
        print("Assessment complete!")
        print("=" * 70)
        print(f"  Results: {output_dir}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be saved.")
        return 130
    except NotImplementedError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose > 0:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
