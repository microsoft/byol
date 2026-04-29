# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LLM Judge Evaluation - Compare two language models using LLM as judge."""

import gc
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

from .llm_judge_utils import (
    load_huggingface_model,
    cleanup_model,
    get_comparison_models,
    get_judge_config,
    get_evaluation_config,
    initialize_endpoint_clients,
    load_dataset_samples,
    format_prompt,
    generate_response,
    batch_generate_responses,
    generate_endpoint_response,
    judge_responses,
    compute_language_accuracy_stats,
    LANGUAGE_CODE_MAP
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_model_response(
    prompt: str,
    model_config: Dict,
    model: Optional[Any],
    tokenizer: Optional[Any],
    full_model_config: Dict,
    endpoint_clients: Dict
) -> str:
    """
    Generate response from a model (HuggingFace or endpoint) - single sample.
    
    Args:
        prompt: Input prompt
        model_config: Model configuration dict
        model: HuggingFace model (None for endpoints)
        tokenizer: HuggingFace tokenizer (None for endpoints)
        full_model_config: Full model configuration with families
        endpoint_clients: Initialized endpoint clients
    
    Returns:
        Generated response text
    """
    if model_config['type'] == 'huggingface':
        return generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_params=model_config.get('generation', {}),
            use_chat_template=model_config.get('use_chat_template', False)
        )
    else:
        return generate_endpoint_response(
            prompt=prompt,
            model_config=model_config,
            clients=endpoint_clients
        )


def generate_responses_sequential(
    samples: List[Dict],
    dataset_name: str,
    dataset_config: Dict,
    model_config: Dict,
    model: Optional[Any],
    tokenizer: Optional[Any],
    full_model_config: Dict,
    endpoint_clients: Dict,
    model_label: str,
    response_key: str,
    log_interval: int
) -> None:
    """
    Generate responses sequentially (one at a time).
    
    Args:
        samples: List of dataset samples
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        model_config: Model configuration
        model: HuggingFace model or None
        tokenizer: HuggingFace tokenizer or None
        full_model_config: Full model configuration
        endpoint_clients: Initialized endpoint clients
        model_label: Label for logging (e.g., "Model A")
        response_key: Key to store response (e.g., "response_a")
        log_interval: Logging frequency
    """
    total_samples = len(samples)
    
    for i, sample in enumerate(samples, 1):
        if i % log_interval == 0 or i == total_samples:
            logger.info(f"  {model_label}: {i}/{total_samples} ({i*100//total_samples}%)")
        
        prompt = format_prompt(sample, dataset_name, dataset_config)
        response = generate_model_response(
            prompt=prompt,
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            full_model_config=full_model_config,
            endpoint_clients=endpoint_clients
        )
        sample[response_key] = response


def generate_responses_batched(
    samples: List[Dict],
    dataset_name: str,
    dataset_config: Dict,
    model_config: Dict,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    model_label: str,
    response_key: str,
    log_interval: int
) -> None:
    """
    Generate responses in batches for HuggingFace models.
    
    Args:
        samples: List of dataset samples
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        model_config: Model configuration
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        batch_size: Number of samples per batch
        model_label: Label for logging (e.g., "Model A")
        response_key: Key to store response (e.g., "response_a")
        log_interval: Logging frequency
    """
    total_samples = len(samples)
    
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_samples = samples[batch_start:batch_end]
        
        # Log progress
        if (batch_end % log_interval == 0) or (batch_end == total_samples):
            logger.info(f"  {model_label}: {batch_end}/{total_samples} ({batch_end*100//total_samples}%)")
        
        # Prepare batch prompts
        batch_prompts = [
            format_prompt(sample, dataset_name, dataset_config)
            for sample in batch_samples
        ]
        
        # Generate batch responses
        batch_responses = batch_generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            generation_params=model_config.get('generation', {}),
            use_chat_template=model_config.get('use_chat_template', False)
        )
        
        # Store responses
        for sample, response in zip(batch_samples, batch_responses):
            sample[response_key] = response
        
        # Aggressive cleanup after each batch to prevent memory accumulation
        del batch_prompts, batch_responses, batch_samples
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure cache is actually cleared
            torch.cuda.empty_cache()  # Double clear to combat fragmentation
        gc.collect()


def generate_for_one_model(samples, dataset_name, dataset_config, model_config, 
                          model, tokenizer, model_label, response_key, log_interval,
                          full_model_config, endpoint_clients):
    """Generate responses for one model."""
    batch_size = model_config.get('batch_size', 1)
    
    if model_config['type'] == 'huggingface' and batch_size > 1:
        generate_responses_batched(
            samples=samples, dataset_name=dataset_name, dataset_config=dataset_config,
            model_config=model_config, model=model, tokenizer=tokenizer,
            batch_size=batch_size, model_label=model_label, 
            response_key=response_key, log_interval=log_interval
        )
    else:
        generate_responses_sequential(
            samples=samples, dataset_name=dataset_name, dataset_config=dataset_config,
            model_config=model_config, model=model, tokenizer=tokenizer,
            full_model_config=full_model_config, endpoint_clients=endpoint_clients,
            model_label=model_label, response_key=response_key, log_interval=log_interval
        )


def process_dataset(
    dataset_name: str,
    dataset_config: Dict,
    model_a_config: Dict,
    model_b_config: Dict,
    judge_config: Dict,
    eval_config: Dict,
    models: Dict,
    tokenizers: Dict,
    model_config: Dict,
    endpoint_clients: Dict,
    output_path: Path
) -> List[Dict]:
    """
    Process a single dataset and evaluate model responses.
    
    Handles large datasets efficiently:
    - Logs progress every 10% or 10 samples (whichever is larger)
    - Sequential processing (batching can be added in future)
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        model_a_config: Model A configuration
        model_b_config: Model B configuration
        judge_config: Judge model configuration
        eval_config: Evaluation settings (language detection)
        models: Loaded models dict
        tokenizers: Loaded tokenizers dict
        model_config: Full model configuration
        endpoint_clients: Initialized endpoint clients
    
    Returns:
        List of evaluation results
    """
    # Load dataset samples
    target_limit = dataset_config.get('limit', 100)
    
    # Load only the requested number of samples
    # All samples will be evaluated (no filtering/skipping)
    samples = load_dataset_samples(dataset_config)
    total_samples = len(samples)
    logger.info(f"Loaded {total_samples} samples (all will be evaluated)")
    
    # Calculate logging frequency (every 10% or minimum 10 samples)
    log_interval = max(10, total_samples // 10)
    
    # Generate responses sequentially (parallel causes CUDA memory errors)
    logger.info(f"Generating responses sequentially...")
    
    logger.info(f"Model A ({total_samples} samples)...")
    generate_for_one_model(samples, dataset_name, dataset_config, model_a_config,
                          models['model_a'], tokenizers['model_a'], "Model A", 
                          "response_a", log_interval, model_config, endpoint_clients)
    
    logger.info(f"Model B ({total_samples} samples)...")
    generate_for_one_model(samples, dataset_name, dataset_config, model_b_config,
                          models['model_b'], tokenizers['model_b'], "Model B",
                          "response_b", log_interval, model_config, endpoint_clients)
    
    logger.info(f"✅ Sequential generation complete")
    
    # Judge responses and collect results
    all_results = []
    checkpoint_interval = max(10, len(samples) // 10)
    
    # Get language detection config
    detection_config = eval_config.get('language_detection')
    
    logger.info(f"\nJudging {len(samples)} responses...")
    
    for i, sample in enumerate(samples, 1):
        if i % log_interval == 0 or i == len(samples):
            logger.info(f"  Judging: {i}/{len(samples)} ({i*100//len(samples)}%)")
        
        result = judge_responses(
                sample=sample,
                response_a=sample['response_a'],
                response_b=sample['response_b'],
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                judge_config=judge_config,
                clients=endpoint_clients,
                language_names=None,
                detection_config=detection_config
            )
        
        # Add original sample data to result
        result['sample'] = sample
        all_results.append(result)
        
        # Incremental checkpoint save every 10%
        if i % checkpoint_interval == 0 or i == len(samples):
            logger.info(f"💾 Saving checkpoint...")
            save_results(all_results, output_path, dataset_name, model_a_config, model_b_config, 
                        is_final=False)
    
    # Compute statistics if language detection enabled
    language_stats = None
    if detection_config:
        language_stats = compute_language_accuracy_stats(all_results)
        logger.info(f"\nLanguage Detection Statistics:")
        logger.info(f"  Overall Accuracy: {language_stats['overall_accuracy']:.2%}")
        logger.info(f"  Model A Accuracy: {language_stats['model_a_accuracy']:.2%}")
        logger.info(f"  Model B Accuracy: {language_stats['model_b_accuracy']:.2%}")
    
    return all_results, language_stats


def compute_summary_statistics(results: List[Dict]) -> Dict:
    """
    Compute summary statistics from evaluation results.
    Includes ALL samples - no skipping based on language detection.
    
    Args:
        results: List of evaluation results with winner and ratings
        
    Returns:
        Dict with win/loss/tie counts and average ratings for all samples
    """
    stats = {
        'model_a_wins': 0,
        'model_b_wins': 0,
        'ties': 0,
        'total_samples': len(results),
        'model_a_ratings': [],
        'model_b_ratings': []
    }
    
    for result in results:
        winner = result.get('winner', 'tie')
        
        # Count all samples
        if winner == 'model_a':
            stats['model_a_wins'] += 1
        elif winner == 'model_b':
            stats['model_b_wins'] += 1
        else:
            stats['ties'] += 1
        
        stats['model_a_ratings'].append(result.get('rating_a', 0))
        stats['model_b_ratings'].append(result.get('rating_b', 0))
    
    # Calculate percentages and averages on ALL samples
    total = stats['total_samples']
    stats['evaluated_comparisons'] = total
    stats['model_a_win_rate'] = (stats['model_a_wins'] / total) if total > 0 else 0.0
    stats['model_b_win_rate'] = (stats['model_b_wins'] / total) if total > 0 else 0.0
    stats['tie_rate'] = (stats['ties'] / total) if total > 0 else 0.0
    
    stats['model_a_avg_rating'] = sum(stats['model_a_ratings']) / len(stats['model_a_ratings']) if stats['model_a_ratings'] else 0.0
    stats['model_b_avg_rating'] = sum(stats['model_b_ratings']) / len(stats['model_b_ratings']) if stats['model_b_ratings'] else 0.0
    
    # Remove raw rating lists from output (keep summary only)
    del stats['model_a_ratings']
    del stats['model_b_ratings']
    
    return stats


def save_results(
    results: List[Dict],
    output_path: Path,
    dataset_name: str,
    model_a_config: Dict,
    model_b_config: Dict,
    is_final: bool = False,
    language_stats: Optional[Dict] = None
) -> None:
    """
    Save evaluation results to JSON file (single file, overwritten each time).
    
    Args:
        results: List of evaluation results
        output_path: Output directory path
        dataset_name: Name of the dataset
        model_a_config: Model A configuration
        model_b_config: Model B configuration
        is_final: Whether this is the final save
        language_stats: Optional language detection statistics
    """
    from datetime import datetime
    
    # Compute summary statistics (excludes skipped samples)
    summary = compute_summary_statistics(results)
    
    # Create filename with model names, sample count, and date
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_a_id = model_a_config.get('id', model_a_config['name']).replace('/', '_')
    model_b_id = model_b_config.get('id', model_b_config['name']).replace('/', '_')
    num_samples = len(results)
    output_file = output_path / f"{dataset_name}_{model_a_id}_vs_{model_b_id}_{num_samples}samples_{date_str}_results.json"
    
    # Build output with metadata
    output_data = {
        "metadata": {
            "dataset": dataset_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_a": {
                "name": model_a_config['name'],
                "id": model_a_config.get('id', ''),
                "type": model_a_config.get('type', ''),
                "path": model_a_config.get('repo') or model_a_config.get('endpoint', '')
            },
            "model_b": {
                "name": model_b_config['name'],
                "id": model_b_config.get('id', ''),
                "type": model_b_config.get('type', ''),
                "path": model_b_config.get('repo') or model_b_config.get('endpoint', '')
            },
            "num_samples": len(results),
            "is_final": is_final
        },
        "summary": summary,
        "results": results
    }
    
    # Add language detection stats if available
    if language_stats:
        output_data["language_detection_stats"] = language_stats
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if is_final:
        # Log summary statistics only for final save
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Samples: {summary['total_samples']}")
        logger.info(f"Total Comparisons: {summary['evaluated_comparisons']}")
        logger.info(f"\nWin/Loss/Tie:")
        logger.info(f"  Model A Wins: {summary['model_a_wins']} ({summary['model_a_win_rate']:.1%})")
        logger.info(f"  Model B Wins: {summary['model_b_wins']} ({summary['model_b_win_rate']:.1%})")
        logger.info(f"  Ties:         {summary['ties']} ({summary['tie_rate']:.1%})")
        logger.info(f"\nAverage Ratings:")
        logger.info(f"  Model A: {summary['model_a_avg_rating']:.2f}/5.0")
        logger.info(f"  Model B: {summary['model_b_avg_rating']:.2f}/5.0")
        
        # Add language detection stats if available
        if language_stats:
            logger.info(f"\nLanguage Detection Accuracy:")
            logger.info(f"  Model A: {language_stats['model_a_accuracy']:.1%} ({language_stats['model_a_correct']}/{language_stats['model_a_correct']+language_stats['model_a_incorrect']})")
            logger.info(f"  Model B: {language_stats['model_b_accuracy']:.1%} ({language_stats['model_b_correct']}/{language_stats['model_b_correct']+language_stats['model_b_incorrect']})")
            logger.info(f"  Overall:  {language_stats['overall_accuracy']:.1%}")
        
        logger.info(f"{'='*60}\n")
        logger.info(f"✅ Results saved to: {output_file}")


def run_evaluation(
    model_config_path: str,
    dataset_config_path: str,
    output_dir: str = "./results"
):
    """
    Run LLM judge evaluation comparing two models.
    
    Args:
        model_config_path: Path to model configuration YAML
        dataset_config_path: Path to dataset configuration YAML
        output_dir: Directory to save results
    """
    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    
    # Get models to compare (includes device mapping from config)
    model_a_config, model_b_config = get_comparison_models(model_config)
    judge_config = get_judge_config(model_config)
    
    # Extract devices from config
    device_a = model_a_config.get('device', 'cuda:0')
    device_b = model_b_config.get('device', 'cuda:1')
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load models based on type
    models = {}
    tokenizers = {}
    
    # Load Model A
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading Model A: {model_a_config['name']}")
    logger.info(f"{'='*60}")
    
    if model_a_config['type'] == 'huggingface':
        model_path = model_a_config.get('repo') or model_a_config.get('path')
        model_a, tokenizer_a = load_huggingface_model(
            model_path=model_path,
            device=device_a
        )
        models['model_a'] = model_a
        tokenizers['model_a'] = tokenizer_a
    elif model_a_config['type'] == 'endpoint':
        logger.info(f"Model A is endpoint: {model_a_config.get('endpoint', model_a_config.get('path'))}")
        models['model_a'] = None
        tokenizers['model_a'] = None
    
    # Load Model B
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading Model B: {model_b_config['name']}")
    logger.info(f"{'='*60}")
    
    if model_b_config['type'] == 'huggingface':
        model_path = model_b_config.get('repo') or model_b_config.get('path')
        model_b, tokenizer_b = load_huggingface_model(
            model_path=model_path,
            device=device_b
        )
        models['model_b'] = model_b
        tokenizers['model_b'] = tokenizer_b
    elif model_b_config['type'] == 'endpoint':
        logger.info(f"Model B is endpoint: {model_b_config.get('endpoint', model_b_config.get('path'))}")
        models['model_b'] = None
        tokenizers['model_b'] = None
    
    # Get enabled datasets
    enabled_datasets = {
        name: config for name, config in dataset_config.get('datasets', {}).items()
        if config.get('enabled', False)
    }
    
    if not enabled_datasets:
        logger.warning("No datasets enabled in configuration!")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Enabled datasets: {list(enabled_datasets.keys())}")
    logger.info(f"{'='*60}\n")
    
    # Initialize endpoint clients if needed
    endpoint_clients = initialize_endpoint_clients()
    
    # Get evaluation config (language detection settings)
    eval_config = get_evaluation_config(model_config)
    
    # Process each enabled dataset
    all_results = {}
    for dataset_name, dataset_cfg in enabled_datasets.items():
        # Get language info from subset or languages_filter
        lang_code = dataset_cfg.get('subset', '') or (dataset_cfg.get('languages_filter', [None])[0] if dataset_cfg.get('languages_filter') else '')
        lang_name = LANGUAGE_CODE_MAP.get(lang_code, lang_code) if lang_code else ''
        lang_info = f" - Language: {lang_name} ({lang_code})" if lang_code else ""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}{lang_info}")
        logger.info(f"{'='*60}")
        
        process_results = process_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_cfg,
            model_a_config=model_a_config,
            model_b_config=model_b_config,
            judge_config=judge_config,
            eval_config=eval_config,
            models=models,
            tokenizers=tokenizers,
            model_config=model_config,
            endpoint_clients=endpoint_clients,
            output_path=output_path
        )
        
        results, language_stats = process_results
        all_results[dataset_name] = results
        
        # Final save with full results and language stats
        save_results(results, output_path, dataset_name, model_a_config, model_b_config, 
                    is_final=True, language_stats=language_stats)
    
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation complete!")
    logger.info(f"{'='*60}")
    model_a_path = model_a_config.get('repo') or model_a_config.get('endpoint') or model_a_config.get('path')
    model_b_path = model_b_config.get('repo') or model_b_config.get('endpoint') or model_b_config.get('path')
    logger.info(f"Model A: {model_a_config['name']} - {model_a_path} ({model_a_config['type']})")
    logger.info(f"Model B: {model_b_config['name']} - {model_b_path} ({model_b_config['type']})")
    logger.info(f"Judge: {judge_config['model']} ({judge_config['type']})")
    logger.info(f"Datasets: {', '.join(enabled_datasets.keys())}")
    logger.info(f"Results saved to: {output_path}")
    
    # Cleanup
    logger.info("\nCleaning up models...")
    if models['model_a'] is not None:
        cleanup_model(models['model_a'])
    if models['model_b'] is not None:
        cleanup_model(models['model_b'])
    
    logger.info("Evaluation complete!")


def main():
    """Main entry point."""
    # Set PyTorch memory allocator settings to reduce fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    
    # Disable torch.compile to avoid recompilation limit errors with batched generation
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    
    parser = argparse.ArgumentParser(
        description="LLM Judge Evaluation - Compare two language models"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Run evaluation (devices are now specified in model config)
    run_evaluation(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
