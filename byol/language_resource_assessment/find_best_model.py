# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Find Best Model Task

Evaluate translation quality for low-resource languages via round-trip translation.
Compare different models (API translators and local LLMs) to find the best one for your language.

This module is used by both:
- find-best-translator: Evaluates translation-focused models (NLLB, Microsoft Translator, etc.)
- find-best-llm: Evaluates general-purpose LLMs (Qwen, Gemma, etc.) for translation capability

The underlying evaluation is identical - only the config file differs:
- translators.yaml for find-best-translator
- llms.yaml for find-best-llm

This module contains:
- Translation pipeline (forward + back translation)
- Result key management for different model types
- Concurrent processing for API and sequential for local models
"""

from __future__ import annotations

import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from byol.translation_backends import (
    translate,
    MODEL_REGISTRY,
    get_model_config,
    get_translator,  # Legacy support for batch processing
)

from .config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    API_MAX_CONCURRENT,
    CORE_KEYS,
    get_data_dir,
)
from .io import load_existing_results, save_results
from .normalize import normalize_text
from byol.common.retry import translate_with_retry
from .metrics import compute_metrics
from .visualize import generate_plots


def _kill_executor_children(executor: ProcessPoolExecutor) -> None:
    """Kill all child processes spawned by a ProcessPoolExecutor.
    
    This ensures GPU memory is freed when the main process is interrupted.
    Called automatically via signal handler on SIGINT/SIGTERM.
    """
    pids = []
    # ProcessPoolExecutor stores worker processes in _processes dict
    if hasattr(executor, "_processes"):
        for pid, proc in executor._processes.items():
            if proc.is_alive():
                pids.append(pid)
    
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    
    # Give them a moment, then force kill
    import time
    time.sleep(1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    
    if pids:
        print(f"\n[cleanup] Terminated {len(pids)} worker process(es)")


def clear_gpu_memory() -> None:
    """
    Clear GPU memory by clearing all model caches and running garbage collection.
    
    This should be called between local model runs to free VRAM.
    Uses the centralized cache registry from _utils.py.
    """
    from byol.translation_backends.local._utils import clear_model_caches
    
    # Clear all registered model caches
    clear_model_caches()
    
    # Additional synchronization for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    print("[cleanup] GPU memory cleared")


def get_result_key_for_model(model_name: str, config: dict) -> str:
    """
    Get the actual key that will be used in results for a model.
    
    For local models with model_name, uses the model's short name.
    Otherwise uses the model name directly.
    """
    if config.get("translator_type") == "local" and "model_name" in config:
        model_path = config.get("model_name", "")
        return model_path.split("/")[-1]
    return model_name


def _run_single_model(
    model_name: str,
    config: dict,
    dataset_path: Path,
    output_file: Path,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_concurrent: int,
    checkpoint_interval: int,
    max_samples: int | None = None,
    device: str | None = None,
    device_queue: multiprocessing.Queue | None = None,
) -> str:
    """
    Run forward/back translations for a single model.
    
    Safe to call concurrently for different models.
    If device_queue is provided, dynamically grabs a free GPU from the queue
    and returns it when done (with GPU memory cleared).
    
    Returns:
        Status message string
    """
    # Dynamic device assignment: grab a free GPU from the queue
    acquired_device = None
    if device_queue is not None:
        acquired_device = device_queue.get()  # blocks until a GPU is free
        device = acquired_device
        print(f"\n[worker:{model_name}] Acquired {device}")
    
    try:
        return _run_single_model_impl(
            model_name, config, dataset_path, output_file,
            src_lang, tgt_lang, batch_size, max_concurrent,
            checkpoint_interval, max_samples, device,
        )
    finally:
        # Always return device to the queue and clear GPU memory
        if device_queue is not None and acquired_device is not None:
            try:
                clear_gpu_memory()
            except Exception:
                pass
            device_queue.put(acquired_device)
            print(f"[worker:{model_name}] Released {acquired_device}")


def _run_single_model_impl(
    model_name: str,
    config: dict,
    dataset_path: Path,
    output_file: Path,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_concurrent: int,
    checkpoint_interval: int,
    max_samples: int | None = None,
    device: str | None = None,
) -> str:
    """Inner implementation of _run_single_model (no device queue logic)."""
    print(f"\n[worker:{model_name}] Starting...")
    
    # Load and normalize dataset
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    normalized = []
    for sample in ds:
        normalized_sample = dict(sample)
        normalized_sample["text"] = normalize_text(sample["text"])
        normalized.append(normalized_sample)
    
    # Initialize local results with core fields
    local_results = []
    for sample in normalized:
        local_results.append({
            "id": sample["id"],
            "domain": sample["domain"],
            "text": sample["text"],
            "source_language": src_lang,
            "target_language": tgt_lang,
        })
    
    # Determine result key and model configuration
    result_key = get_result_key_for_model(model_name, config)
    model_config = config.copy()
    src_lang_code = model_config.pop("src_lang")
    tgt_lang_code = model_config.pop("tgt_lang")
    is_local = model_config.get("translator_type") == "local"
    factory_name = model_config.pop("factory_name", model_name)
    model_config.pop("translator_type", None)
    
    # Inject device for local models
    if is_local and device is not None:
        model_config["device"] = device
    
    # Determine which samples still need processing
    on_disk = load_existing_results(output_file) or []
    completed_ids = set()
    if on_disk:
        for record in on_disk:
            if result_key in record and record[result_key].get("src2tgt"):
                completed_ids.add(record["id"])
    
    pending_indices = [
        i for i, s in enumerate(normalized) 
        if s["id"] not in completed_ids
    ]
    
    if not pending_indices:
        print(f"[worker:{model_name}] Already complete, skipping")
        return f"{model_name}: skipped (already complete)"
    
    print(f"[worker:{model_name}] Processing {len(pending_indices)} samples")
    
    # Initialize translators
    forward_translator = get_translator(
        name=factory_name,
        src_lang=src_lang_code,
        tgt_lang=tgt_lang_code,
        **model_config,
    )
    back_translator = get_translator(
        name=factory_name,
        src_lang=tgt_lang_code,
        tgt_lang=src_lang_code,
        **model_config,
    )
    
    try:
        if is_local:
            _process_local_model(
                normalized=normalized,
                pending_indices=pending_indices,
                local_results=local_results,
                result_key=result_key,
                forward_translator=forward_translator,
                back_translator=back_translator,
                model_name=model_name,
                batch_size=batch_size,
                output_file=output_file,
            )
        else:
            _process_api_model(
                normalized=normalized,
                pending_indices=pending_indices,
                local_results=local_results,
                result_key=result_key,
                forward_translator=forward_translator,
                back_translator=back_translator,
                model_name=model_name,
                max_concurrent=max_concurrent,
                checkpoint_interval=checkpoint_interval,
                output_file=output_file,
            )
    finally:
        # Clean up translator objects and free GPU memory
        try:
            del forward_translator
            del back_translator
        except Exception:
            pass
        
        try:
            clear_gpu_memory()
        except Exception:
            pass
    
    print(f"[worker:{model_name}] Done")
    return f"{model_name}: done"


def _process_local_model(
    normalized: list[dict],
    pending_indices: list[int],
    local_results: list[dict],
    result_key: str,
    forward_translator: Any,
    back_translator: Any,
    model_name: str,
    batch_size: int,
    output_file: Path,
) -> None:
    """Process samples using a local model with batch processing."""
    forward_texts = [normalized[i]["text"] for i in pending_indices]
    
    print(f"[worker:{model_name}] Processing {len(forward_texts)} samples in batches of {batch_size}")
    
    for batch_start in tqdm(
        range(0, len(forward_texts), batch_size),
        desc=f"{model_name} batches",
    ):
        batch_end = min(batch_start + batch_size, len(forward_texts))
        batch_texts = forward_texts[batch_start:batch_end]
        batch_indices = pending_indices[batch_start:batch_end]
        
        # Batch translate
        forward_translations = forward_translator.translate_batch(texts=batch_texts)
        back_translations = back_translator.translate_batch(texts=forward_translations)
        
        # Store results
        for idx, fwd, bwd in zip(batch_indices, forward_translations, back_translations):
            local_results[idx][result_key] = {"src2tgt": fwd, "tgt2src": bwd}
        
        # Checkpoint
        save_results(local_results, output_file)


def _process_api_model(
    normalized: list[dict],
    pending_indices: list[int],
    local_results: list[dict],
    result_key: str,
    forward_translator: Any,
    back_translator: Any,
    model_name: str,
    max_concurrent: int,
    checkpoint_interval: int,
    output_file: Path,
) -> None:
    """Process samples using an API model with concurrent requests."""
    print(
        f"[worker:{model_name}] Processing {len(pending_indices)} samples "
        f"with {max_concurrent} concurrent requests"
    )
    
    def process_single_sample(idx: int) -> tuple[int, dict]:
        """Process a single sample with forward and backward translation."""
        sample = normalized[idx]
        try:
            if not sample["text"] or not sample["text"].strip():
                raise ValueError("Input text empty")
            
            # Forward translation with retry
            fwd = translate_with_retry(forward_translator, sample["text"])
            if not fwd or not fwd.strip():
                raise ValueError("Forward translation empty")
            
            # Backward translation with retry
            bwd = translate_with_retry(back_translator, fwd)
            
            return idx, {"src2tgt": fwd, "tgt2src": bwd}
        except Exception as e:
            return idx, {"error": str(e), "src2tgt": None, "tgt2src": None}
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(process_single_sample, idx): idx 
            for idx in pending_indices
        }
        
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{model_name} samples",
        ):
            idx, result = future.result()
            local_results[idx][result_key] = result
            completed += 1
            
            # Checkpoint periodically
            if completed % checkpoint_interval == 0:
                save_results(local_results, output_file)
    
    # Final save
    save_results(local_results, output_file)


def _run_local_models_sequential(
    local_models: dict[str, dict],
    dataset_path: Path,
    output_file: Path,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_concurrent: int,
    checkpoint_interval: int,
    max_samples: int | None,
    device: str | None,
) -> list[str]:
    """Run local models sequentially on a single GPU (in a separate process)."""
    messages = []
    for name, cfg in local_models.items():
        clear_gpu_memory()
        try:
            msg = _run_single_model(
                name, cfg, dataset_path, output_file,
                src_lang, tgt_lang, batch_size, max_concurrent,
                checkpoint_interval, max_samples, device=device,
            )
            messages.append(msg)
        except Exception as e:
            messages.append(f"{name} failed: {e}")
    return messages


def run_translation(
    dataset_path: Path,
    output_file: Path,
    model_configs: dict[str, dict],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = BATCH_SIZE,
    max_concurrent: int = API_MAX_CONCURRENT,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    skip_existing: bool = True,
    max_samples: int | None = None,
    device_list: list[str] | None = None,
) -> None:
    """
    Run round-trip translations for all configured models.
    
    Args:
        dataset_path: Path to the input JSONL dataset
        output_file: Path to save translation results
        model_configs: Dict mapping model names to their configs
        src_lang: Source language name
        tgt_lang: Target language name
        batch_size: Batch size for local models
        max_concurrent: Max concurrent API requests
        checkpoint_interval: Save checkpoint every N samples
        skip_existing: If True, skip models with existing results
        max_samples: Maximum number of samples to process (None = all)
        device_list: List of GPU device IDs for local models (e.g., ["2", "3"])
    """
    # Initialize baseline results file if it doesn't exist
    if not output_file.exists():
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        baseline_results = []
        for sample in ds:
            baseline_results.append({
                "id": sample["id"],
                "domain": sample["domain"],
                "text": normalize_text(sample["text"]),
                "source_language": src_lang,
                "target_language": tgt_lang,
            })
        save_results(baseline_results, output_file)
        print(f"Initialized baseline results at {output_file}")
    
    # Partition models by type
    api_models = {
        name: cfg for name, cfg in model_configs.items()
        if cfg.get("translator_type") != "local"
    }
    local_models = {
        name: cfg for name, cfg in model_configs.items()
        if cfg.get("translator_type") == "local"
    }
    
    # Launch ALL models concurrently — API models are safe to parallelize,
    # and local models use a device queue for dynamic GPU assignment.
    all_futures: dict = {}
    
    # For multi-GPU: use a device queue so models dynamically grab free GPUs.
    # This ensures that when model A finishes on GPU 1, the next model gets GPU 1
    # (not the GPU where model B is still running).
    multi_gpu = device_list and len(device_list) > 1
    num_gpus = len(device_list) if device_list else 1
    device_queue: multiprocessing.Queue | None = None
    
    if multi_gpu and local_models:
        manager = multiprocessing.Manager()
        device_queue = manager.Queue()
        for dev_id in device_list:
            device_queue.put(f"cuda:{dev_id}")
    
    # Max workers: all API models + local models.
    # For multi-GPU, we submit all local models as separate tasks — each blocks
    # on device_queue.get() until a GPU is free, so at most num_gpus run at once.
    if multi_gpu:
        local_workers = len(local_models) if local_models else 0
    else:
        local_workers = 1 if local_models else 0
    max_workers = len(api_models) + local_workers
    
    if not max_workers:
        print("No models to run.")
        return
    
    print(f"\nLaunching {len(api_models)} API + {len(local_models)} local model(s) concurrently")
    
    executor = ProcessPoolExecutor(max_workers=max_workers)
    
    # Install signal handler to kill child processes on Ctrl+C / SIGTERM
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def _cleanup_handler(signum: int, frame: Any) -> None:
        print(f"\n[main] Received signal {signum}, shutting down workers...")
        _kill_executor_children(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        # Restore original handler and re-raise
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)
    
    try:
        # Submit API models (no GPU needed)
        for name, cfg in api_models.items():
            future = executor.submit(
                _run_single_model,
                name, cfg, dataset_path, output_file,
                src_lang, tgt_lang, batch_size, max_concurrent,
                checkpoint_interval, max_samples,
            )
            all_futures[future] = name
        
        # Submit local models
        if multi_gpu and local_models:
            # Multiple GPUs: each worker grabs a free GPU from the queue.
            # We submit ALL local models — ProcessPoolExecutor limits concurrency
            # to num_gpus workers, and each worker blocks on device_queue.get()
            # until a GPU is free.
            for name, cfg in local_models.items():
                future = executor.submit(
                    _run_single_model,
                    name, cfg, dataset_path, output_file,
                    src_lang, tgt_lang, batch_size, max_concurrent,
                    checkpoint_interval, max_samples,
                    device=None, device_queue=device_queue,
                )
                all_futures[future] = name
        elif local_models:
            # Single GPU: run local models sequentially in a separate process
            single_device = f"cuda:{device_list[0]}" if device_list else None
            future = executor.submit(
                _run_local_models_sequential,
                local_models, dataset_path, output_file,
                src_lang, tgt_lang, batch_size, max_concurrent,
                checkpoint_interval, max_samples, single_device,
            )
            all_futures[future] = "local-models"
        
        # Collect results
        for future in as_completed(all_futures):
            name = all_futures[future]
            try:
                msg = future.result()
                if isinstance(msg, list):
                    for m in msg:
                        print(f"[main] {m}")
                else:
                    print(f"[main] {msg}")
            except Exception as e:
                print(f"[main] {name} failed: {e}")
    except KeyboardInterrupt:
        print("\n[main] Interrupted — killing workers and freeing GPUs...")
        _kill_executor_children(executor)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
    
    # Final cleanup after all local models
    if local_models:
        clear_gpu_memory()
    
    print(f"\nCompleted all translations. Results saved to {output_file}")


def run_model_evaluation(
    src_lang: str,
    tgt_lang: str,
    model_configs: dict[str, dict],
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    step: str = "all",
    batch_size: int = BATCH_SIZE,
    max_concurrent: int = API_MAX_CONCURRENT,
    max_samples: int | None = None,
    skip_existing: bool = True,
    embedding_model: str = "openai",
    excluded_models: list[str] | None = None,
    task_name: str = "model",
    device_list: list[str] | None = None,
) -> None:
    """
    Main entry point for model evaluation task.
    
    This function is used by both find-best-translator and find-best-llm tasks.
    The only difference is which config file is used to populate model_configs.
    
    Runs the full pipeline:
    1. Translation (forward + back)
    2. Metrics computation
    3. Visualization
    
    Args:
        src_lang: Source language name
        tgt_lang: Target language name
        model_configs: Dict mapping model names to their configs
        dataset_path: Path to input JSONL dataset (default: built-in RTTBench-Mono)
        output_dir: Output directory for results
        step: Which step to run - "all", "translate", "metrics", or "plot"
        batch_size: Batch size for local models
        max_concurrent: Max concurrent API requests
        max_samples: Maximum number of samples to process (None = all)
        skip_existing: If True, skip models with existing results
        embedding_model: Embedding model for similarity - "openai" or "qwen"
        excluded_models: Models to exclude from plots
        task_name: Name of the task for display purposes ("translator" or "llm")
    """
    # Resolve paths
    if dataset_path is None:
        dataset_path = get_data_dir() / "rttbench_mono_dataset" / "RTTBench-Mono.jsonl"
    
    if output_dir is None:
        # Default output directory based on task
        if task_name.lower() == "llm":
            task_subdir = "llm_comparison"
        else:
            task_subdir = "translator_comparison"
        output_dir = Path("./results") / tgt_lang.lower() / "lra" / task_subdir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    translation_file = output_dir / f"translations_{src_lang}_{tgt_lang}_on_rttbench_dataset.jsonl"
    metrics_file = translation_file.with_name(
        translation_file.stem + f"_with_metrics_{embedding_model}.jsonl"
    )
    
    # Run pipeline steps
    if step in ("all", "translate"):
        print("\n" + "-" * 50)
        print("[Step 1/3] Running round-trip translations...")
        print("-" * 50)
        
        run_translation(
            dataset_path=dataset_path,
            output_file=translation_file,
            model_configs=model_configs,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            skip_existing=skip_existing,
            max_samples=max_samples,
            device_list=device_list,
        )
    
    if step in ("all", "metrics"):
        print("\n" + "-" * 50)
        print("[Step 2/3] Computing metrics...")
        print("-" * 50)
        
        compute_metrics(
            input_file=translation_file,
            output_file=metrics_file,
            embedding_model=embedding_model,
        )
    
    if step in ("all", "plot"):
        print("\n" + "-" * 50)
        print("[Step 3/3] Generating visualizations...")
        print("-" * 50)
        
        generate_plots(
            input_file=metrics_file,
            output_dir=output_dir,
            excluded_translators=excluded_models or [],
        )
    
    print("\n" + "=" * 50)
    print(f"{task_name.capitalize()} evaluation complete!")
    print("=" * 50)
    print(f"  Results:      {output_dir}")
    print(f"  Translations: {translation_file.name}")
    print(f"  Metrics:      {metrics_file.name}")
    print(f"  Plots:        ranking_*.png")


# Backward compatibility alias
run_translator_evaluation = run_model_evaluation


__all__ = [
    "run_translation",
    "run_model_evaluation",
    "run_translator_evaluation",  # Backward compatibility
]
