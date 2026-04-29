# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model merging utilities: LoRA merge, delta merge, and multi-model merge."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger("byol-train")

# Type aliases
DType = Literal["float16", "bfloat16", "float32"]


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class MergeConfig:
    """Configuration for LlamaFactory LoRA export."""
    
    model_name_or_path: str = ""
    adapter_name_or_path: str = ""
    template: str = "gemma"
    export_dir: str = ""
    export_size: int = 2
    export_device: str = "auto"
    export_legacy_format: bool = False
    
    @classmethod
    def from_yaml(cls, path: Path) -> "MergeConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model_name_or_path=data.get("model_name_or_path", ""),
            adapter_name_or_path=data.get("adapter_name_or_path", ""),
            template=data.get("template", "gemma"),
            export_dir=data.get("export_dir", ""),
            export_size=data.get("export_size", 2),
            export_device=data.get("export_device", "auto"),
            export_legacy_format=data.get("export_legacy_format", False),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LlamaFactory."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "adapter_name_or_path": self.adapter_name_or_path,
            "template": self.template,
            "export_dir": self.export_dir,
            "export_size": self.export_size,
            "export_device": self.export_device,
            "export_legacy_format": self.export_legacy_format,
        }


@dataclass
class MergeResult:
    """Result from a merge operation."""
    
    success: bool
    output_dir: str
    merge_type: str
    formula: str
    metadata: Dict[str, Any]
    error: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================

def _get_dtype(dtype: DType) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype, torch.float16)


def _log_merge_banner(title: str, **kwargs: Any) -> None:
    """Log a formatted merge operation banner."""
    logger.info("=" * 60)
    logger.info(title)
    for key, value in kwargs.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def _check_peft_available() -> None:
    """Check if PEFT library is available."""
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA merging. "
            "Install with: pip install peft"
        )


def _load_model(
    model_path: str,
    device: str,
    dtype: torch.dtype,
) -> PreTrainedModel:
    """Load model with specified device and dtype."""
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )


def _load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load tokenizer from model path."""
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _merge_lora_into_model(
    base_model: PreTrainedModel,
    lora_path: str,
    device: str,
) -> PreTrainedModel:
    """Merge LoRA adapter weights into base model."""
    _check_peft_available()
    
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base_model, lora_path, device_map={"": device}
    )
    
    logger.info("Merging LoRA weights into base model...")
    return lora_model.merge_and_unload()


def _save_model_and_metadata(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    output_dir: str,
    merge_info: Dict[str, Any],
    dtype: torch.dtype,
    device: str,
) -> None:
    """Save merged model, tokenizer, and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # Save merge metadata
    metadata = {
        "merge_info": {
            **merge_info,
            "dtype": str(dtype),
            "device": device,
            "output_path": output_dir,
        }
    }
    
    metadata_path = Path(output_dir) / "merge_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")


# =============================================================================
# LlamaFactory Export (Original merge_lora function)
# =============================================================================

def merge_lora_llamafactory(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    template: str = "gemma",
    export_size: int = 2,
    dry_run: bool = False,
) -> bool:
    """Merge LoRA adapter using LlamaFactory CLI."""
    config = MergeConfig(
        model_name_or_path=base_model,
        adapter_name_or_path=adapter_path,
        template=template,
        export_dir=output_dir,
        export_size=export_size,
    )
    
    _log_merge_banner(
        "Merging LoRA adapter (LlamaFactory)",
        **{"Base model": base_model, "Adapter": adapter_path, "Output": output_dir}
    )
    
    # Write temp config
    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="merge_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    if dry_run:
        logger.info("[DRY RUN] Config:")
        logger.info(Path(temp_path).read_text())
        Path(temp_path).unlink()
        return True
    
    try:
        cmd = ["llamafactory-cli", "export", temp_path]
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            logger.info(f"Merged model saved to: {output_dir}")
            return True
        else:
            logger.error(f"Merge failed with exit code {result.returncode}")
            return False
    
    except Exception as e:
        logger.exception(f"Error: {e}")
        return False
    
    finally:
        Path(temp_path).unlink(missing_ok=True)


# Alias for backward compatibility
merge_lora = merge_lora_llamafactory


# =============================================================================
# Simple LoRA Merge
# =============================================================================

def merge_lora_simple(
    base_model: str,
    lora_path: str,
    output_dir: str,
    dtype: DType = "float16",
    gpu: int = 0,
) -> MergeResult:
    """Simple LoRA merge: base + lora_adapter."""
    _check_peft_available()
    
    device = f"cuda:{gpu}"
    torch_dtype = _get_dtype(dtype)
    formula = "merged = base + lora_adapter"
    
    _log_merge_banner(
        "Simple LoRA Merge",
        **{"Base model": base_model, "LoRA adapter": lora_path, "Formula": formula, "Output": output_dir}
    )
    
    try:
        # Load tokenizer and base model
        tokenizer = _load_tokenizer(base_model)
        model = _load_model(base_model, device, torch_dtype)
        
        # Merge LoRA
        merged_model = _merge_lora_into_model(model, lora_path, device)
        
        # Save
        merge_info = {
            "merge_type": "simple_lora",
            "base_model": base_model,
            "lora_adapter": lora_path,
            "formula": formula,
        }
        _save_model_and_metadata(
            merged_model, tokenizer, output_dir, merge_info, torch_dtype, device
        )
        
        logger.info("Simple LoRA merge completed successfully")
        
        return MergeResult(
            success=True,
            output_dir=output_dir,
            merge_type="simple_lora",
            formula=formula,
            metadata=merge_info,
        )
        
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        return MergeResult(
            success=False,
            output_dir=output_dir,
            merge_type="simple_lora",
            formula=formula,
            metadata={},
            error=str(e),
        )


# =============================================================================
# Delta Merge (instruct + alpha * (fine_tuned - base))
# =============================================================================

def delta_merge(
    base: str,
    instruct: str,
    fine_tuned: str,
    alpha: float,
    output_dir: str,
    dtype: DType = "float16",
    gpu: int = 0,
) -> MergeResult:
    """Delta merge: instruct + alpha * (fine_tuned - base)."""
    device = f"cuda:{gpu}"
    torch_dtype = _get_dtype(dtype)
    formula = f"merged = instruct + {alpha} * (fine_tuned - base)"
    
    _log_merge_banner(
        "Delta Merge (Full Models)",
        **{"Base model": base, "Instruct model": instruct, "Fine-tuned model": fine_tuned,
           "Alpha": alpha, "Formula": formula, "Output": output_dir}
    )
    
    try:
        # Load tokenizer from instruct model
        tokenizer = _load_tokenizer(instruct)
        
        # Load all models
        logger.info("Loading instruct model...")
        instruct_model = _load_model(instruct, device, torch_dtype)
        
        logger.info("Loading fine-tuned model...")
        fine_tuned_model = _load_model(fine_tuned, device, torch_dtype)
        
        logger.info("Loading base model...")
        base_model = _load_model(base, device, torch_dtype)
        
        # Get state dictionaries
        instruct_state = instruct_model.state_dict()
        fine_tuned_state = fine_tuned_model.state_dict()
        base_state = base_model.state_dict()
        
        # Perform merge: merged = instruct + alpha * (fine_tuned - base)
        logger.info(f"Performing merge with alpha={alpha}...")
        merged_state = {}
        
        for key in instruct_state.keys():
            if key in fine_tuned_state and key in base_state:
                delta = fine_tuned_state[key] - base_state[key]
                merged_state[key] = instruct_state[key] + alpha * delta
            else:
                merged_state[key] = instruct_state[key]
        
        # Load merged weights
        instruct_model.load_state_dict(merged_state)
        
        # Cleanup
        del fine_tuned_model, base_model, fine_tuned_state, base_state, merged_state
        torch.cuda.empty_cache()
        
        # Save
        merge_info = {
            "merge_type": "delta_merge",
            "base_model": base,
            "instruct_model": instruct,
            "fine_tuned_model": fine_tuned,
            "alpha": alpha,
            "formula": formula,
        }
        _save_model_and_metadata(
            instruct_model, tokenizer, output_dir, merge_info, torch_dtype, device
        )
        
        logger.info("Delta merge completed successfully")
        
        return MergeResult(
            success=True,
            output_dir=output_dir,
            merge_type="delta_merge",
            formula=formula,
            metadata=merge_info,
        )
        
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        return MergeResult(
            success=False,
            output_dir=output_dir,
            merge_type="delta_merge",
            formula=formula,
            metadata={},
            error=str(e),
        )


# =============================================================================
# Delta Merge with LoRA
# =============================================================================

def delta_merge_lora(
    base: str,
    instruct: str,
    lora_path: str,
    alpha: float,
    output_dir: str,
    dtype: DType = "float16",
    gpu: int = 0,
) -> MergeResult:
    """Delta merge with LoRA: instruct + alpha * (lora_merged_base - base)."""
    _check_peft_available()
    
    device = f"cuda:{gpu}"
    torch_dtype = _get_dtype(dtype)
    formula = f"merged = instruct + {alpha} * (lora_merged_base - base)"
    
    _log_merge_banner(
        "Delta Merge (LoRA)",
        **{"Base model": base, "Instruct model": instruct, "LoRA adapter": lora_path,
           "Alpha": alpha, "Formula": formula, "Output": output_dir}
    )
    
    try:
        # Load tokenizer from instruct model
        tokenizer = _load_tokenizer(instruct)
        
        # Load models
        logger.info("Loading instruct model...")
        instruct_model = _load_model(instruct, device, torch_dtype)
        
        logger.info("Loading base model...")
        base_model = _load_model(base, device, torch_dtype)
        
        # Create base model copy and merge LoRA
        logger.info("Loading base model copy for LoRA merge...")
        base_for_lora = _load_model(base, device, torch_dtype)
        lora_merged = _merge_lora_into_model(base_for_lora, lora_path, device)
        
        # Get state dictionaries
        instruct_state = instruct_model.state_dict()
        lora_merged_state = lora_merged.state_dict()
        base_state = base_model.state_dict()
        
        # Perform merge: merged = instruct + alpha * (lora_merged - base)
        logger.info(f"Performing merge with alpha={alpha}...")
        merged_state = {}
        
        for key in instruct_state.keys():
            if key in lora_merged_state and key in base_state:
                delta = lora_merged_state[key] - base_state[key]
                merged_state[key] = instruct_state[key] + alpha * delta
            else:
                merged_state[key] = instruct_state[key]
        
        # Load merged weights
        instruct_model.load_state_dict(merged_state)
        
        # Cleanup
        del base_model, lora_merged, lora_merged_state, base_state, merged_state
        torch.cuda.empty_cache()
        
        # Save
        merge_info = {
            "merge_type": "delta_merge_lora",
            "base_model": base,
            "instruct_model": instruct,
            "lora_adapter": lora_path,
            "alpha": alpha,
            "formula": formula,
        }
        _save_model_and_metadata(
            instruct_model, tokenizer, output_dir, merge_info, torch_dtype, device
        )
        
        logger.info("Delta LoRA merge completed successfully")
        
        return MergeResult(
            success=True,
            output_dir=output_dir,
            merge_type="delta_merge_lora",
            formula=formula,
            metadata=merge_info,
        )
        
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        return MergeResult(
            success=False,
            output_dir=output_dir,
            merge_type="delta_merge_lora",
            formula=formula,
            metadata={},
            error=str(e),
        )


# =============================================================================
# General 4-Model Merge
# =============================================================================

def general_merge(
    model_pt: str,
    model_it: str,
    model_el: str,
    beta: float,
    output_dir: str,
    dtype: DType = "float16",
    gpu: int = 0,
) -> MergeResult:
    """Model merging from BYOL paper (Eq. 2).

    M(α,β) = G_PT + α(G_IT − G_PT) + β(E_ℓ − G_PT)

    With α = 1−β (one-dimensional slice from ablation):
        merged = G_PT + (1−β)*(G_IT − G_PT) + β*(E_ℓ − G_PT)

    Args:
        model_pt: G_PT — pretrained generalist (e.g. google/gemma-3-4b-pt)
        model_it: G_IT — instruction-tuned generalist (e.g. google/gemma-3-4b-it)
        model_el: E_ℓ  — language expert (SFT checkpoint)
        beta: λ — language expert weight (paper default: 0.6)
    """
    device = f"cuda:{gpu}"
    torch_dtype = _get_dtype(dtype)
    alpha = 1 - beta
    formula = f"merged = G_PT + {alpha}*(G_IT - G_PT) + {beta}*(E_l - G_PT)"

    _log_merge_banner(
        "BYOL Model Merge (Eq. 2)",
        **{"G_PT (pretrained)": model_pt, "G_IT (instruct)": model_it,
           "E_l (language expert)": model_el,
           "beta (lambda)": beta, "Formula": formula, "Output": output_dir}
    )

    try:
        # Use instruct tokenizer to preserve chat template
        tokenizer = _load_tokenizer(model_it)

        # Phase 1: Load G_PT, G_IT, apply alpha*(G_IT - G_PT)
        logger.info("Phase 1: Loading G_PT and G_IT...")
        model = _load_model(model_pt, device, torch_dtype)
        model_IT = _load_model(model_it, device, torch_dtype)

        logger.info(f"Applying alpha*(G_IT - G_PT) with alpha={alpha}...")
        params_IT = dict(model_IT.named_parameters())

        for name, param in model.named_parameters():
            if name in params_IT:
                with torch.no_grad():
                    param.data = param.data + alpha * (params_IT[name].data - param.data)

        logger.info("Releasing G_IT...")
        del model_IT, params_IT
        torch.cuda.empty_cache()

        # Phase 2: Load E_l and G_PT_original, apply beta*(E_l - G_PT)
        logger.info("Phase 2: Loading E_l and G_PT (original)...")
        model_EL = _load_model(model_el, device, torch_dtype)
        model_PT_orig = _load_model(model_pt, device, torch_dtype)

        logger.info(f"Applying beta*(E_l - G_PT) with beta={beta}...")
        params_EL = dict(model_EL.named_parameters())
        params_PT = dict(model_PT_orig.named_parameters())

        for name, param in model.named_parameters():
            if name in params_EL and name in params_PT:
                with torch.no_grad():
                    param.data = param.data + beta * (params_EL[name].data - params_PT[name].data)

        logger.info("Releasing E_l and G_PT copy...")
        del model_EL, model_PT_orig, params_EL, params_PT
        torch.cuda.empty_cache()

        # Save
        merge_info = {
            "merge_type": "general_merge",
            "model_pt": model_pt,
            "model_it": model_it,
            "model_el": model_el,
            "beta": beta,
            "alpha": alpha,
            "formula": formula,
        }
        _save_model_and_metadata(
            model, tokenizer, output_dir, merge_info, torch_dtype, device
        )

        logger.info("Model merge completed successfully")

        return MergeResult(
            success=True,
            output_dir=output_dir,
            merge_type="general_merge",
            formula=formula,
            metadata=merge_info,
        )

    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        return MergeResult(
            success=False,
            output_dir=output_dir,
            merge_type="general_merge",
            formula=formula,
            metadata={},
            error=str(e),
        )


# =============================================================================
# General LoRA Merge (A + beta*(B-A) + (1-beta)*lora)
# =============================================================================

def general_merge_lora(
    model_pt: str,
    model_it: str,
    lora_path: str,
    beta: float,
    output_dir: str,
    dtype: DType = "float16",
    gpu: int = 0,
) -> MergeResult:
    """BYOL model merge with LoRA adapter (Eq. 2 variant).

    merged = G_PT + (1−β)*(G_IT − G_PT) + β*(LoRA_merged − G_PT)

    Args:
        model_pt: G_PT — pretrained generalist
        model_it: G_IT — instruction-tuned generalist
        lora_path: LoRA adapter path (language expert trained on G_PT)
        beta: λ — language expert weight (paper default: 0.6)
    """
    _check_peft_available()

    device = f"cuda:{gpu}"
    torch_dtype = _get_dtype(dtype)
    alpha = 1 - beta
    formula = f"merged = G_PT + {alpha}*(G_IT - G_PT) + {beta}*lora_adapter"

    _log_merge_banner(
        "BYOL LoRA Merge (Eq. 2)",
        **{"G_PT (pretrained)": model_pt, "G_IT (instruct)": model_it,
           "LoRA adapter": lora_path,
           "beta (lambda)": beta, "Formula": formula, "Output": output_dir}
    )

    try:
        # Use instruct tokenizer to preserve chat template
        tokenizer = _load_tokenizer(model_it)

        # Phase 1: Load G_PT and G_IT, apply alpha*(G_IT - G_PT)
        logger.info("Phase 1: Loading G_PT and G_IT...")
        model = _load_model(model_pt, device, torch_dtype)
        model_IT = _load_model(model_it, device, torch_dtype)

        # Store original G_PT params for LoRA delta
        logger.info("Storing original G_PT parameters...")
        original_PT_params = {}
        for name, param in model.named_parameters():
            original_PT_params[name] = param.detach().clone()

        # Apply alpha*(G_IT - G_PT)
        logger.info(f"Applying alpha*(G_IT - G_PT) with alpha={alpha}...")
        params_IT = dict(model_IT.named_parameters())

        for name, param in model.named_parameters():
            if name in params_IT:
                with torch.no_grad():
                    param.data = param.data + alpha * (params_IT[name].data - param.data)

        logger.info("Releasing G_IT...")
        del model_IT, params_IT
        torch.cuda.empty_cache()

        # Phase 2: Load G_PT copy, merge LoRA, apply beta*(LoRA_merged - G_PT)
        logger.info("Phase 2: Loading G_PT copy for LoRA merge...")
        model_for_lora = _load_model(model_pt, device, torch_dtype)
        lora_merged = _merge_lora_into_model(model_for_lora, lora_path, device)

        logger.info(f"Applying beta*(LoRA_merged - G_PT) with beta={beta}...")
        params_lora = dict(lora_merged.named_parameters())

        for name, param in model.named_parameters():
            if name in params_lora and name in original_PT_params:
                with torch.no_grad():
                    param.data = param.data + beta * (params_lora[name].data - original_PT_params[name].data)

        logger.info("Releasing LoRA model and original parameters...")
        del lora_merged, params_lora, original_PT_params
        torch.cuda.empty_cache()

        # Save
        merge_info = {
            "merge_type": "general_lora",
            "model_pt": model_pt,
            "model_it": model_it,
            "lora_adapter": lora_path,
            "beta": beta,
            "alpha": alpha,
            "formula": formula,
        }
        _save_model_and_metadata(
            model, tokenizer, output_dir, merge_info, torch_dtype, device
        )

        logger.info("LoRA merge completed successfully")

        return MergeResult(
            success=True,
            output_dir=output_dir,
            merge_type="general_lora",
            formula=formula,
            metadata=merge_info,
        )

    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        return MergeResult(
            success=False,
            output_dir=output_dir,
            merge_type="general_lora",
            formula=formula,
            metadata={},
            error=str(e),
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """CLI entry point for model merging."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Model merging utilities for BYOL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple LoRA merge
  python -m byol.train.merge simple-lora \\
    --base google/gemma-3-4b-pt \\
    --lora path/to/adapter \\
    --output outputs/merged

  # Delta merge (instruct + alpha*(fine_tuned - base))
  python -m byol.train.merge delta \\
    --base google/gemma-3-4b-pt \\
    --instruct google/gemma-3-4b-it \\
    --fine-tuned path/to/cpt-model \\
    --alpha 0.5 \\
    --output outputs/merged

  # BYOL model merge (Eq. 2): G_PT + (1-beta)*(G_IT - G_PT) + beta*(E_l - G_PT)
  python -m byol.train.merge general \\
    --model-pt google/gemma-3-4b-pt \\
    --model-it google/gemma-3-4b-it \\
    --model-el path/to/sft_checkpoint \\
    --beta 0.6 \\
    --output outputs/merged

  # BYOL merge with LoRA adapter
  python -m byol.train.merge general-lora \\
    --model-pt google/gemma-3-4b-pt \\
    --model-it google/gemma-3-4b-it \\
    --lora path/to/lora_adapter \\
    --beta 0.6 \\
    --output outputs/merged
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Merge strategy")
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-o", "--output",
        type=str,
        default="./outputs/merged",
        help="Output directory (default: ./outputs/merged)",
    )
    common.add_argument(
        "--device", "--gpu",
        dest="gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)",
    )
    common.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model precision (default: float16)",
    )
    
    # Simple LoRA merge
    simple_lora = subparsers.add_parser(
        "simple-lora",
        parents=[common],
        help="Simple LoRA merge: base + lora_adapter",
    )
    simple_lora.add_argument("--base", type=str, required=True, help="Base model")
    simple_lora.add_argument("--lora", type=str, required=True, help="LoRA adapter path")
    
    # LlamaFactory export
    llamafactory = subparsers.add_parser(
        "llamafactory",
        parents=[common],
        help="LlamaFactory LoRA export",
    )
    llamafactory.add_argument("--base", type=str, required=True, help="Base model")
    llamafactory.add_argument("--adapter", type=str, required=True, help="Adapter path")
    llamafactory.add_argument("--template", type=str, default="gemma", help="Template")
    llamafactory.add_argument("--dry-run", action="store_true", help="Dry run")
    
    # Delta merge (full models)
    delta = subparsers.add_parser(
        "delta",
        parents=[common],
        help="Delta merge: instruct + alpha*(fine_tuned - base)",
    )
    delta.add_argument("--base", type=str, required=True, help="Base model")
    delta.add_argument("--instruct", type=str, required=True, help="Instruct model")
    delta.add_argument("--fine-tuned", type=str, required=True, help="Fine-tuned model")
    delta.add_argument("--alpha", type=float, required=True, help="Merge weight")
    
    # Delta merge with LoRA
    delta_lora = subparsers.add_parser(
        "delta-lora",
        parents=[common],
        help="Delta merge with LoRA: instruct + alpha*(lora_merged - base)",
    )
    delta_lora.add_argument("--base", type=str, required=True, help="Base model")
    delta_lora.add_argument("--instruct", type=str, required=True, help="Instruct model")
    delta_lora.add_argument("--lora", type=str, required=True, help="LoRA adapter path")
    delta_lora.add_argument("--alpha", type=float, required=True, help="Merge weight")
    
    # General model merge (BYOL Eq. 2)
    general = subparsers.add_parser(
        "general",
        parents=[common],
        help="BYOL merge (Eq. 2): G_PT + (1-beta)*(G_IT - G_PT) + beta*(E_l - G_PT)",
    )
    general.add_argument("--model-pt", type=str, required=True, help="G_PT: pretrained generalist (e.g. google/gemma-3-4b-pt)")
    general.add_argument("--model-it", type=str, required=True, help="G_IT: instruction-tuned generalist (e.g. google/gemma-3-4b-it)")
    general.add_argument("--model-el", type=str, required=True, help="E_l: language expert (SFT checkpoint)")
    general.add_argument("--beta", type=float, default=0.6, help="Lambda: language expert weight (default: 0.6)")
    
    # General LoRA merge
    general_lora_cmd = subparsers.add_parser(
        "general-lora",
        parents=[common],
        help="BYOL LoRA merge: G_PT + (1-beta)*(G_IT - G_PT) + beta*lora",
    )
    general_lora_cmd.add_argument("--model-pt", type=str, required=True, help="G_PT: pretrained generalist")
    general_lora_cmd.add_argument("--model-it", type=str, required=True, help="G_IT: instruction-tuned generalist")
    general_lora_cmd.add_argument("--lora", type=str, required=True, help="LoRA adapter path (trained on G_PT)")
    general_lora_cmd.add_argument("--beta", type=float, default=0.6, help="Lambda: language expert weight (default: 0.6)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if not args.command:
        parser.print_help()
        return
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        exit(1)
    
    num_gpus = torch.cuda.device_count()
    if args.gpu >= num_gpus:
        logger.error(f"GPU {args.gpu} not available. Available: 0 to {num_gpus-1}")
        exit(1)
    
    torch.cuda.set_device(args.gpu)
    
    # Execute command
    if args.command == "simple-lora":
        result = merge_lora_simple(
            base_model=args.base,
            lora_path=args.lora,
            output_dir=args.output,
            dtype=args.dtype,
            gpu=args.gpu,
        )
    elif args.command == "llamafactory":
        success = merge_lora_llamafactory(
            base_model=args.base,
            adapter_path=args.adapter,
            output_dir=args.output,
            template=args.template,
            dry_run=args.dry_run,
        )
        exit(0 if success else 1)
    elif args.command == "delta":
        result = delta_merge(
            base=args.base,
            instruct=args.instruct,
            fine_tuned=args.fine_tuned,
            alpha=args.alpha,
            output_dir=args.output,
            dtype=args.dtype,
            gpu=args.gpu,
        )
    elif args.command == "delta-lora":
        result = delta_merge_lora(
            base=args.base,
            instruct=args.instruct,
            lora_path=args.lora,
            alpha=args.alpha,
            output_dir=args.output,
            dtype=args.dtype,
            gpu=args.gpu,
        )
    elif args.command == "general":
        result = general_merge(
            model_pt=args.model_pt,
            model_it=args.model_it,
            model_el=args.model_el,
            beta=args.beta,
            output_dir=args.output,
            dtype=args.dtype,
            gpu=args.gpu,
        )
    elif args.command == "general-lora":
        result = general_merge_lora(
            model_pt=args.model_pt,
            model_it=args.model_it,
            lora_path=args.lora,
            beta=args.beta,
            output_dir=args.output,
            dtype=args.dtype,
            gpu=args.gpu,
        )
    else:
        parser.print_help()
        return
    
    # Print result
    if result.success:
        logger.info("=" * 60)
        logger.info(f"{result.merge_type.upper()} COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {result.output_dir}")
        logger.info(f"Formula: {result.formula}")
        logger.info("=" * 60)
    else:
        logger.error(f"Merge failed: {result.error}")
        exit(1)


if __name__ == "__main__":
    main()
