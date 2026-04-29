# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Constants and default values for BYOL Data Preparation."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# =============================================================================
# Package / Repo Paths
# =============================================================================
DATA_PREP_PACKAGE_DIR: _Path = _Path(__file__).resolve().parent
REPO_ROOT: _Path = DATA_PREP_PACKAGE_DIR.parent.parent  # byol/data_prep → repo root

DEFAULT_CONFIGS_DIR: str = "configs/data_prep"
_DATA_ROOT: str = _os.environ.get("BYOL_DATA_DIR", str(_Path("~/byol-data").expanduser()))
DEFAULT_OUTPUT_DIR: str = _DATA_ROOT
DEFAULT_SHARED_DIR: str = str(_Path(_DATA_ROOT) / "_shared")


def ensure_data_symlink(tgt_lang_code: str, output_dir: str) -> None:
    """Ensure eval data is accessible at data/<lang>/eval/.

    For shipped languages (nya, mri) the eval/ directory already exists in the
    repo with real files.  For new languages, if data/<lang>/eval/ doesn't exist
    we symlink it to the external output directory.

    Also creates data/<lang>/ as a real directory if it doesn't exist (no
    language-level symlinks — only subfolder symlinks for cpt/sft).
    """
    import logging
    _logger = logging.getLogger("byol-data-prep")

    lang_dir = REPO_ROOT / "data" / tgt_lang_code
    lang_dir.mkdir(parents=True, exist_ok=True)

    eval_link = lang_dir / "eval"
    if eval_link.exists() or eval_link.is_symlink():
        return

    # Resolve external eval directory
    out = _Path(output_dir).resolve()
    if out.name == "eval":
        target = out
    elif out.name == tgt_lang_code:
        target = out / "eval"
    else:
        target = out / tgt_lang_code / "eval"

    target.mkdir(parents=True, exist_ok=True)
    eval_link.symlink_to(target)
    _logger.info(f"Created symlink: {eval_link} -> {target}")

# Dataset tags used in the bilingual mix JSONL (matches existing training data)
# -- CPT tags --
DATASET_TAG_REFINED_TGT_LANG: str = "train_refined_{lang}"
DATASET_TAG_REFINED_ENG: str = "train_refined_edu_english"
DATASET_TAG_TRANSLATED: str = "train_refined_translated_edu_english"

# -- SFT tags --
DATASET_TAG_SFT_SMOLTALK2_ENG: str = "smoltalk2_subset"
DATASET_TAG_SFT_SMOLTALK2_TRANSLATED: str = "smoltalk2_subset_translated_to_{lang}_using_gpt5"
DATASET_TAG_SFT_AYA_TRANSLATED_TRAIN: str = "aya_dataset_translated_to_{lang}_train"
DATASET_TAG_SFT_AYA_NATIVE_TRAIN: str = "aya_dataset_native_{lang}_train"

# =============================================================================
# Supported Stages
# =============================================================================
SUPPORTED_STAGES: tuple[str, ...] = ("cpt", "sft", "eval")

# =============================================================================
# Language Metadata
# =============================================================================
# Maps ISO 639-3 codes → language names (extended as needed)
LANG_NAMES: dict[str, str] = {
    "nya": "Chichewa",
    "mri": "Māori",
    "eng": "English",
    "fra": "French",
    "deu": "German",
    "nld": "Dutch",
    "spa": "Spanish",
    "ita": "Italian",
    "gug": "Guaraní",
}

# =============================================================================
# SFT Data Source Defaults
# =============================================================================
# smoltalk2 dataset subsets configuration
SMOLTALK2_DATASETS_CONFIG: list[dict] = [
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "smoltalk_smollm3_explore_instruct_rewriting_no_think",
        "default_source": "smoltalk_smollm3_explore_instruct_rewriting_no_think",
        "default_chat_template_kwargs": "explore_instruct_rewriting",
        "sample_size": None,  # all samples (~30.4k)
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "smoltalk_smollm3_everyday_conversations_no_think",
        "default_source": "smoltalk_smollm3_everyday_conversations_no_think",
        "default_chat_template_kwargs": "everyday_conversations",
        "sample_size": None,  # all samples (~2.26k)
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "default_source": "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "default_chat_template_kwargs": "smol_magpie_ultra",
        "sample_size": 50000,
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "smoltalk_smollm3_smol_rewrite_no_think",
        "default_source": "smoltalk_smollm3_smol_rewrite_no_think",
        "default_chat_template_kwargs": "smol_rewrite",
        "sample_size": 50000,
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "smoltalk_smollm3_smol_summarize_no_think",
        "default_source": "smoltalk_smollm3_smol_summarize_no_think",
        "default_chat_template_kwargs": "smol_summarize",
        "sample_size": 50000,
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "smoltalk_smollm3_systemchats_30k_no_think",
        "default_source": "smoltalk_smollm3_systemchats_30k_no_think",
        "default_chat_template_kwargs": "systemchats_30k",
        "sample_size": None,  # all samples
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk",
        "subset": "smol-constraints",
        "split": None,
        "default_source": "smol-constraints",
        "default_chat_template_kwargs": "",
        "sample_size": None,  # all samples
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "OpenThoughts3_1.2M_no_think_no_think",
        "default_source": "OpenThoughts3_1.2M_no_think_no_think",
        "default_chat_template_kwargs": "openthoughts",
        "sample_size": 50000,
    },
    {
        "dataset_name": "HuggingFaceTB/smoltalk2",
        "subset": "SFT",
        "split": "OpenHermes_2.5_no_think",
        "default_source": "OpenHermes_2.5_no_think",
        "default_chat_template_kwargs": "openhermes",
        "sample_size": 50000,
    },
]

# AYA dataset configuration
AYA_DATASET_REPO_ID: str = "CohereForAI/aya_dataset"
AYA_SOURCE_LANGUAGE_CODES: list[str] = ["eng", "fra", "deu", "spa", "ita"]

# SFT Translation defaults
DEFAULT_SFT_TRANSLATE_BATCH_SIZE: int = 8
DEFAULT_SFT_TRANSLATE_CONCURRENCY: int = 64
DEFAULT_SFT_TRANSLATE_CHECKPOINT_EVERY: int = 50
DEFAULT_SFT_TRANSLATE_TOKEN_BUDGET: int = 2048

# =============================================================================
# FineWeb-2 Defaults
# =============================================================================
FINEWEB2_REPO_ID: str = "HuggingFaceFW/fineweb-2"
FINEWEB2_SPLITS: tuple[str, ...] = ("train",)

# =============================================================================
# FineWeb-Edu Defaults
# =============================================================================
FINEWEBEDU_REPO_ID: str = "HuggingFaceFW/fineweb-edu"
FINEWEBEDU_NUM_SHARDS: int = 14
FINEWEBEDU_SHARD_TEMPLATE: str = "sample/10BT/{index:03d}_00000.parquet"
DEFAULT_MAX_TOKEN_COUNT: int = 1024
DEFAULT_SEED: int = 42
DEFAULT_TIKTOKEN_ENCODING: str = "o200k_base"

# =============================================================================
# LLM Processing Defaults
# =============================================================================
DEFAULT_MODEL_NAME: str = "gpt-5"
DEFAULT_REFINE_ENGLISH_MODEL: str = "gpt-5-mini"
DEFAULT_API_VERSION: str = "2024-08-01-preview"

# Batch processing
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_CONCURRENCY: int = 32
DEFAULT_CHECKPOINT_EVERY: int = 50
DEFAULT_TOKEN_BUDGET_PER_ITEM: int = 2048

# English refinement specifics
DEFAULT_ENGLISH_BATCH_SIZE: int = 16
DEFAULT_ENGLISH_CONCURRENCY: int = 128
DEFAULT_ENGLISH_CHECKPOINT_EVERY: int = 1000
DEFAULT_ENGLISH_TOKEN_BUDGET: int = 1024

# Translation specifics
DEFAULT_TRANSLATE_BATCH_SIZE: int = 16
DEFAULT_TRANSLATE_CONCURRENCY: int = 96
DEFAULT_TRANSLATE_CHECKPOINT_EVERY: int = 5000
DEFAULT_TRANSLATE_TOKEN_BUDGET: int = 1536

# =============================================================================
# Default Config Paths (relative to repo root)
# =============================================================================
DEFAULT_CONFIG_CPT: str = "configs/data_prep/cpt"
DEFAULT_CONFIG_SFT: str = "configs/data_prep/sft"
DEFAULT_CONFIG_EVAL: str = "configs/data_prep/eval"


# =============================================================================
# Eval Data Preparation Defaults
# =============================================================================

# Default output root for eval translated data (goes under ~/byol-data/<lang>/eval/)
DEFAULT_EVAL_OUTPUT_DIR: str = DEFAULT_OUTPUT_DIR

# Default concurrency settings for eval translation
DEFAULT_EVAL_MAX_WORKERS: int = 8
DEFAULT_EVAL_BATCH_SIZE: int = 32

# Benchmark source files (shipped with repo under data/benchmarks/sources/)
BENCHMARK_SOURCE_DIR: str = str(REPO_ROOT / "data" / "benchmarks" / "sources")

# Translator suffix mapping: model name → short suffix for output filename
TRANSLATOR_SUFFIX_MAP: dict[str, str] = {
    "microsoft-translator": "microsoft",
    "google-translator": "google",
    "gpt-5": "gpt5",
    "gpt-5-mini": "gpt5mini",
    "gpt-4o": "gpt4o",
    "gpt-4o-mini": "gpt4omini",
    "deepseek-r1": "deepseekr1",
}

# Default translator model name for eval benchmarks
DEFAULT_EVAL_TRANSLATOR: str = "microsoft-translator"

# Default max_tokens for LLM-based translators
DEFAULT_EVAL_LLM_MAX_TOKENS: int = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Eval Benchmark Defaults
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry describes a single benchmark dataset and how to load / translate it.
#
#   dataset_path   : HuggingFace repo ID or None for local files
#   dataset_config : HF config name (e.g. "ARC-Challenge") or None
#   splits         : list of splits to translate
#   fields         : list of field specifiers to translate.
#                    Plain string → direct field, e.g. "question"
#                    "parent.child" → nested dict access, e.g. "choices.text"
#   translator     : default translator model name
#   local_file_template : for local datasets, filename template (in BENCHMARK_SOURCE_DIR)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re


def _load_mgsm_tsv(rows: list[dict]) -> list[dict]:
    """Post-process: set answer=None and parse answer_number for MGSM test rows."""
    for row in rows:
        if "answer_number" not in row:
            row["answer_number"] = row.pop("answer_num", None)
        row.setdefault("answer", None)
        row.setdefault("equation_solution", None)
    return rows


ROW_TRANSFORMS: dict[str, callable] = {
    "_mgsm_post_process": _load_mgsm_tsv,
}


EVAL_BENCHMARK_DEFAULTS: dict[str, dict] = {
    "copa": {
        "dataset_path": None,  # local file
        "dataset_config": None,
        "splits": ["test", "validation"],
        "fields": ["premise", "choice1", "choice2"],
        "translator": "microsoft-translator",
        "local_file_template": "copa_{split}_fixed.jsonl",
    },
    "ai2_arc_hard": {
        "dataset_path": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "splits": ["validation", "test"],
        "fields": ["question", "choices.text"],
        "translator": "microsoft-translator",
    },
    "ai2_arc_easy": {
        "dataset_path": "allenai/ai2_arc",
        "dataset_config": "ARC-Easy",
        "splits": ["validation", "test"],
        "fields": ["question", "choices.text"],
        "translator": "microsoft-translator",
    },
    "hellaswag": {
        "dataset_path": "Rowan/hellaswag",
        "dataset_config": None,
        "splits": ["validation"],
        "fields": ["activity_label", "ctx_a", "ctx_b", "ctx", "endings"],
        "translator": "microsoft-translator",
    },
    "piqa": {
        "dataset_path": "baber/piqa",
        "dataset_config": None,
        "splits": ["train", "validation"],
        "fields": ["goal", "sol1", "sol2"],
        "translator": "microsoft-translator",
    },
    "xnli": {
        "dataset_path": "facebook/xnli",
        "dataset_config": "en",
        "splits": ["validation", "test"],
        "fields": ["premise", "hypothesis"],
        "translator": "microsoft-translator",
    },
    "xstory_cloze": {
        "dataset_path": "juletxara/xstory_cloze",
        "dataset_config": "en",
        "splits": ["train", "eval"],
        "fields": [
            "input_sentence_1",
            "input_sentence_2",
            "input_sentence_3",
            "input_sentence_4",
            "sentence_quiz1",
            "sentence_quiz2",
        ],
        "translator": "microsoft-translator",
    },
    "mgsm": {
        "dataset_path": "juletxara/mgsm",
        "dataset_config": "en",
        "splits": ["train", "test"],
        "fields": ["question"],
        "translator": "gpt-5",
        "hf_loader": "_mgsm",
        "row_transform": "_mgsm_post_process",
    },
    "HiTZ-truthfulqa-multi": {
        "dataset_path": "HiTZ/truthfulqa-multi",
        "dataset_config": "en",
        "splits": ["train", "validation"],
        "fields": [
            "question",
            "mc1_targets.choices",
            "mc2_targets.choices",
            "best_answer",
            "correct_answers",
            "incorrect_answers",
        ],
        "translator": "microsoft-translator",
    },
    "mmlu_lite": {
        "dataset_path": "CohereForAI/Global-MMLU-Lite",
        "dataset_config": "en",
        "splits": ["dev", "test"],
        "fields": ["question", "option_a", "option_b", "option_c", "option_d"],
        "translator": "gpt-5",
    },
    "xwinograd": {
        "dataset_path": None,  # local file
        "dataset_config": None,
        "splits": ["test"],
        "fields": ["sentence", "option1", "option2"],
        "translator": "microsoft-translator",
        "local_file_template": "xwinograd_aligned_english_1000.jsonl",
        "output_filename_template": "xwinograd_aligned_{lang}_1000.jsonl",
    },
}
