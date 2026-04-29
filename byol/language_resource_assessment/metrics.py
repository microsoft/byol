# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Metric computation for translation quality assessment.

Computes:
- src2tgt_similarity_score: Cosine similarity between source and forward translation
- tgt2src_similarity_score: Cosine similarity between source and back translation
- tgt2src_sacreBLEU: BLEU score of back translation vs original
- tgt2src_chrF++: chrF++ score of back translation vs original
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import evaluate
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import REQUIRED_METRICS, METRICS_BATCH_SIZE, CORE_KEYS
from .embedding import EmbeddingClient, EmbeddingModelType
from .io import load_jsonl, save_jsonl
from .normalize import normalize_text


def _check_translator_completeness(entry: dict, translator: str) -> bool:
    """Check if all required metrics are present for a translator."""
    if translator not in entry:
        return False
    
    translator_data = entry[translator]
    return all(metric in translator_data for metric in REQUIRED_METRICS)


def _get_translator_names(results: list[dict]) -> list[str]:
    """Extract translator names from results."""
    if not results:
        return []
    
    return [
        key for key in results[0]
        if key not in CORE_KEYS
    ]


def compute_metrics(
    input_file: Path,
    output_file: Path,
    device: int = 0,
    batch_size: int = METRICS_BATCH_SIZE,
    embedding_model: EmbeddingModelType = "openai",
) -> None:
    """
    Compute translation quality metrics for all translators.
    
    Computes:
    - src2tgt_similarity_score: Cosine similarity between source and forward translation
    - tgt2src_similarity_score: Cosine similarity between source and back translation  
    - tgt2src_sacreBLEU: BLEU score of back translation vs original
    - tgt2src_chrF++: chrF++ score of back translation vs original
    
    Args:
        input_file: Path to JSONL with translations
        output_file: Path to save results with metrics
        device: GPU device number (ignored - CUDA_VISIBLE_DEVICES is set at startup)
        batch_size: Batch size for embedding computation
        embedding_model: Which embedding model to use - "openai" or "qwen"
    """
    # NOTE: Device is always cuda:0 because CUDA_VISIBLE_DEVICES is set at startup
    # to restrict torch to only see the specified GPU
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Use smaller batch size for Qwen to avoid OOM (8B model needs more memory)
    if embedding_model == "qwen":
        batch_size = min(batch_size, 4)  # Max 4 entries per batch for Qwen
        print(f"Using reduced batch size ({batch_size}) for Qwen embedding model")
    
    # Initialize embedding client based on model choice
    print(f"Using embedding model: {embedding_model}")
    embed_client = EmbeddingClient(model_type=embedding_model, device=device_str)
    
    # Load results
    print(f"Loading translations from {input_file}")
    results = load_jsonl(input_file)
    
    if not results:
        raise ValueError(f"No results found in {input_file}")
    
    # Get translator names
    translator_names = _get_translator_names(results)
    print(f"Found {len(translator_names)} translators: {translator_names}")
    
    # Normalize text fields
    for entry in results:
        entry["text"] = normalize_text(entry["text"])
        for translator in translator_names:
            if translator in entry and entry[translator].get("tgt2src"):
                entry[translator]["tgt2src"] = normalize_text(entry[translator]["tgt2src"])
    
    # Load existing results if available
    existing_results: dict[Any, dict] = {}
    if output_file.exists():
        print(f"Loading existing metrics from {output_file}")
        for entry in load_jsonl(output_file):
            existing_results[entry["id"]] = entry
    
    # Initialize evaluation metrics
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    
    # Track statistics
    processed_count = 0
    skipped_count = 0
    checkpoint_interval = 5  # Save every N batches
    batches_since_checkpoint = 0
    
    # Process in batches
    total_batches = (len(results) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(results), batch_size), desc="Computing metrics", total=total_batches):
        batch = results[i : i + batch_size]
        
        # Determine what needs processing
        batch_info = []
        for entry in batch:
            entry_id = entry["id"]
            info = {
                "entry": entry,
                "needs_embedding": False,
                "translators_to_process": [],
            }
            
            # Check existing results
            if entry_id in existing_results:
                existing = existing_results[entry_id]
                for translator in translator_names:
                    if translator in existing and _check_translator_completeness(existing, translator):
                        # Copy existing metrics
                        entry[translator].update({
                            metric: existing[translator][metric]
                            for metric in REQUIRED_METRICS
                            if metric in existing[translator]
                        })
                        skipped_count += 1
            
            # Determine which translators need processing
            for translator in translator_names:
                if not _check_translator_completeness(entry, translator):
                    # Check if we have valid translation data
                    if (translator in entry and 
                        entry[translator].get("src2tgt") and 
                        entry[translator].get("tgt2src")):
                        info["translators_to_process"].append(translator)
                        info["needs_embedding"] = True
                        processed_count += 1
            
            if info["translators_to_process"]:
                batch_info.append(info)
        
        if not batch_info:
            continue
        
        # Collect texts for embedding
        all_texts = []
        text_mapping = []  # (text_type, entry_idx, translator)
        
        for info in batch_info:
            entry = info["entry"]
            entry_idx = results.index(entry)
            
            if info["needs_embedding"]:
                # Original text
                all_texts.append(normalize_text(entry["text"]))
                text_mapping.append(("original", entry_idx, None))
                
                # Translator texts
                for translator in info["translators_to_process"]:
                    all_texts.append(normalize_text(entry[translator]["src2tgt"]))
                    text_mapping.append(("src2tgt", entry_idx, translator))
                    
                    all_texts.append(normalize_text(entry[translator]["tgt2src"]))
                    text_mapping.append(("tgt2src", entry_idx, translator))
        
        if not all_texts:
            continue
        
        # Get embeddings using unified client
        all_embeddings = embed_client.embed(all_texts)
        
        # Build lookup
        embedding_lookup: dict[int, dict] = {}
        for idx, (text_type, entry_idx, translator) in enumerate(text_mapping):
            if entry_idx not in embedding_lookup:
                embedding_lookup[entry_idx] = {}
            
            if text_type == "original":
                embedding_lookup[entry_idx]["original"] = all_embeddings[idx]
            else:
                if translator not in embedding_lookup[entry_idx]:
                    embedding_lookup[entry_idx][translator] = {}
                embedding_lookup[entry_idx][translator][text_type] = all_embeddings[idx]
        
        # Compute metrics
        for info in batch_info:
            entry = info["entry"]
            entry_idx = results.index(entry)
            
            if entry_idx not in embedding_lookup:
                continue
            
            text_embedding = embedding_lookup[entry_idx]["original"].unsqueeze(0)
            
            for translator in info["translators_to_process"]:
                if translator not in embedding_lookup[entry_idx]:
                    continue
                
                src2tgt_emb = embedding_lookup[entry_idx][translator]["src2tgt"].unsqueeze(0)
                tgt2src_emb = embedding_lookup[entry_idx][translator]["tgt2src"].unsqueeze(0)
                
                # Cosine similarities
                entry[translator]["src2tgt_similarity_score"] = round(
                    F.cosine_similarity(text_embedding, src2tgt_emb).item() * 100, 2
                )
                entry[translator]["tgt2src_similarity_score"] = round(
                    F.cosine_similarity(text_embedding, tgt2src_emb).item() * 100, 2
                )
                
                # BLEU and chrF++
                original_text = normalize_text(entry["text"])
                back_translation = normalize_text(entry[translator]["tgt2src"])
                
                bleu_score = bleu.compute(
                    predictions=[back_translation],
                    references=[[original_text]],
                )
                chrf_score = chrf.compute(
                    predictions=[back_translation],
                    references=[[original_text]],
                    word_order=2,
                )
                
                entry[translator]["tgt2src_sacreBLEU"] = round(bleu_score["score"], 2)
                entry[translator]["tgt2src_chrF++"] = round(chrf_score["score"], 2)
        
        # Clear embedding lookup to free memory
        del embedding_lookup, all_embeddings
        
        # Clear GPU cache for Qwen model
        embed_client.clear_cache()
        
        # Incremental checkpoint save
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_interval:
            save_jsonl(results, output_file)
            batches_since_checkpoint = 0
    
    # Final save
    save_jsonl(results, output_file)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total entries: {len(results)}")
    print(f"  Total translators: {len(translator_names)}")
    print(f"  Metrics computed: {processed_count}")
    print(f"  Metrics from cache: {skipped_count}")


__all__ = [
    "compute_metrics",
]
