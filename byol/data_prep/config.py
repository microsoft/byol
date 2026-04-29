# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Configuration dataclasses for data preparation pipelines."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_CONCURRENCY,
    DEFAULT_ENGLISH_BATCH_SIZE,
    DEFAULT_ENGLISH_CHECKPOINT_EVERY,
    DEFAULT_ENGLISH_CONCURRENCY,
    DEFAULT_ENGLISH_TOKEN_BUDGET,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_LLM_MAX_TOKENS,
    DEFAULT_EVAL_MAX_WORKERS,
    DEFAULT_EVAL_OUTPUT_DIR,
    DEFAULT_EVAL_TRANSLATOR,
    DEFAULT_MAX_TOKEN_COUNT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINE_ENGLISH_MODEL,
    DEFAULT_SEED,
    DEFAULT_SFT_TRANSLATE_BATCH_SIZE,
    DEFAULT_SFT_TRANSLATE_CHECKPOINT_EVERY,
    DEFAULT_SFT_TRANSLATE_CONCURRENCY,
    DEFAULT_SFT_TRANSLATE_TOKEN_BUDGET,
    DEFAULT_SHARED_DIR,
    DEFAULT_TOKEN_BUDGET_PER_ITEM,
    DEFAULT_TRANSLATE_BATCH_SIZE,
    DEFAULT_TRANSLATE_CHECKPOINT_EVERY,
    DEFAULT_TRANSLATE_CONCURRENCY,
    DEFAULT_TRANSLATE_TOKEN_BUDGET,
    AYA_SOURCE_LANGUAGE_CODES,
    EVAL_BENCHMARK_DEFAULTS,
    LANG_NAMES,
    SMOLTALK2_DATASETS_CONFIG,
    SUPPORTED_STAGES,
)


# Shared serialization helpers

_LLM_CONFIG_EXCLUDE_KEYS = {"api_version", "max_completion_tokens"}


def _llm_config_to_dict(cfg: "LLMStepConfig") -> Dict[str, Any]:
    """Serialize an LLMStepConfig to a clean dict for YAML output."""
    d = dataclasses.asdict(cfg)
    return {k: v for k, v in d.items() if v is not None and k not in _LLM_CONFIG_EXCLUDE_KEYS}


def _write_yaml(data: Dict[str, Any], path: Path, header: str) -> None:
    """Write a config dict to YAML with a header comment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")


# Known script suffixes for FineWeb-2 subsets (non-Latin languages)
_FINEWEB2_SCRIPT_OVERRIDES: dict[str, str] = {
    "bho": "Deva", "hin": "Deva", "mar": "Deva", "nep": "Deva", "san": "Deva",
    "mai": "Deva", "doi": "Deva", "kok": "Deva", "brx": "Deva",
    "ara": "Arab", "urd": "Arab", "fas": "Arab", "snd": "Arab", "uig": "Arab",
    "ben": "Beng", "asm": "Beng",
    "tam": "Taml", "tel": "Telu", "kan": "Knda", "mal": "Mlym", "guj": "Gujr",
    "pan": "Guru", "ori": "Orya", "sin": "Sinh", "mya": "Mymr",
    "tha": "Thai", "lao": "Laoo", "khm": "Khmr", "bod": "Tibt",
    "kat": "Geor", "hye": "Armn", "ell": "Grek",
    "zho": "Hans", "jpn": "Jpan", "kor": "Hang",
    "heb": "Hebr", "amh": "Ethi", "tir": "Ethi",
    "rus": "Cyrl", "ukr": "Cyrl", "bul": "Cyrl", "srp": "Cyrl", "mkd": "Cyrl",
    "bel": "Cyrl", "mon": "Cyrl", "kaz": "Cyrl", "kir": "Cyrl", "tgk": "Cyrl",
}


def _guess_fineweb2_subset(lang_code: str) -> str:
    """Guess the FineWeb-2 HuggingFace subset name for a language code.
    
    Most languages use {iso3}_Latn; non-Latin scripts have specific suffixes.
    """
    script = _FINEWEB2_SCRIPT_OVERRIDES.get(lang_code, "Latn")
    return f"{lang_code}_{script}"


# ──────────────────────────────────────────────────────────────────────────────
# Extra user-supplied data sources
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXTRA_SOURCE_FORMATS = ("text", "sharegpt", "aya")


@dataclass
class ExtraSource:
    """An additional user-supplied JSONL file to include in the bilingual mix.

    Configure via YAML::

        extra_sources:
          - path: /data/my_corpus.jsonl
            dataset_tag: my_corpus
            format: text          # CPT: one ``text`` field per row
          - path: /data/my_chat.jsonl
            dataset_tag: my_chat
            format: sharegpt      # SFT: ``messages`` list per row

    Attributes:
        path: Absolute or relative path to a JSONL file.
        dataset_tag: Short identifier used to tag rows (e.g. ``"my_corpus"``).
        format: Data format — ``"text"`` (CPT), ``"sharegpt"`` or ``"aya"`` (SFT).
    """

    path: str = ""
    dataset_tag: str = ""
    format: str = "text"  # "text" | "sharegpt" | "aya"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("ExtraSource.path is required")
        if not self.dataset_tag:
            raise ValueError("ExtraSource.dataset_tag is required")
        if self.format not in VALID_EXTRA_SOURCE_FORMATS:
            raise ValueError(
                f"ExtraSource.format must be one of {VALID_EXTRA_SOURCE_FORMATS}, "
                f"got {self.format!r}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs for individual processing steps
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMStepConfig:
    """Parameters shared by all LLM-based processing steps (refine / translate)."""

    model_name: str = DEFAULT_MODEL_NAME
    reasoning_effort: str = "low"
    api_version: str = "2024-08-01-preview"
    batch_size: int = DEFAULT_BATCH_SIZE
    concurrency: int = DEFAULT_CONCURRENCY
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    token_budget_per_item: int = DEFAULT_TOKEN_BUDGET_PER_ITEM
    max_completion_tokens: Optional[int] = None


@dataclass
class FineWeb2Config:
    """Configuration for the FineWeb-2 download step."""

    dataset_name: Optional[str] = None  # auto-derived from tgt_lang_code if None
    splits: List[str] = field(default_factory=lambda: ["train"])


@dataclass
class FineWebEduConfig:
    """Configuration for the FineWeb-Edu download step."""

    num_shards: int = 14
    parquet_dir: Optional[str] = None  # where raw parquets live / will be downloaded


@dataclass
class SubsetConfig:
    """Configuration for the FineWeb-Edu subset extraction step.

    ``target_tokens`` is computed automatically from the target-language
    file using tiktoken (o200k_base).  Set a positive value here to
    override the automatic count.
    """

    target_tokens: Optional[int] = None  # None = auto-match tgt_lang token count
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT
    seed: int = DEFAULT_SEED


# ──────────────────────────────────────────────────────────────────────────────
# Top-level CPT data prep config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CPTDataPrepConfig:
    """Full CPT data preparation configuration.

    Controls which pipeline steps run and their parameters.

    Example YAML::

        tgt_lang_code: nya
        tgt_lang_name: Chichewa

        steps:
          download_tgt_lang_fineweb2: true
          refine_tgt_lang: true
          download_eng_finewebedu: true
          refine_eng: true
          translate_eng_to_tgt_lang: true
    """

    # ── Language ─────────────────────────────────────────────────────────
    tgt_lang_code: str = ""
    tgt_lang_name: str = ""  # auto-resolved from code if blank

    # ── Overwrite existing outputs ───────────────────────────────────────
    overwrite: bool = False

    # ── Pipeline step flags ──────────────────────────────────────────────
    download_tgt_lang_fineweb2: bool = True
    refine_tgt_lang: bool = True
    download_eng_finewebedu: bool = True
    refine_eng: bool = True
    translate_eng_to_tgt_lang: bool = True

    # ── Sampling limit (for quick testing) ──────────────────────
    max_samples: Optional[int] = None  # if set, cap rows per step

    # ── Extra user-supplied data sources ─────────────────────────────────
    extra_sources: List[ExtraSource] = field(default_factory=list)

    # ── Sub-configs ──────────────────────────────────────────────────────
    fineweb2: FineWeb2Config = field(default_factory=FineWeb2Config)
    finewebedu: FineWebEduConfig = field(default_factory=FineWebEduConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)

    refine_tgt_lang_config: LLMStepConfig = field(
        default_factory=lambda: LLMStepConfig(
            model_name=DEFAULT_MODEL_NAME,
            reasoning_effort="low",
            batch_size=DEFAULT_BATCH_SIZE,
            concurrency=DEFAULT_CONCURRENCY,
            checkpoint_every=DEFAULT_CHECKPOINT_EVERY,
            token_budget_per_item=DEFAULT_TOKEN_BUDGET_PER_ITEM,
        )
    )
    refine_eng_config: LLMStepConfig = field(
        default_factory=lambda: LLMStepConfig(
            model_name=DEFAULT_REFINE_ENGLISH_MODEL,
            reasoning_effort="minimal",
            batch_size=DEFAULT_ENGLISH_BATCH_SIZE,
            concurrency=DEFAULT_ENGLISH_CONCURRENCY,
            checkpoint_every=DEFAULT_ENGLISH_CHECKPOINT_EVERY,
            token_budget_per_item=DEFAULT_ENGLISH_TOKEN_BUDGET,
        )
    )
    translate_config: LLMStepConfig = field(
        default_factory=lambda: LLMStepConfig(
            model_name=DEFAULT_MODEL_NAME,
            reasoning_effort="low",
            batch_size=DEFAULT_TRANSLATE_BATCH_SIZE,
            concurrency=DEFAULT_TRANSLATE_CONCURRENCY,
            checkpoint_every=DEFAULT_TRANSLATE_CHECKPOINT_EVERY,
            token_budget_per_item=DEFAULT_TRANSLATE_TOKEN_BUDGET,
        )
    )

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        if not self.tgt_lang_code:
            raise ValueError("tgt_lang_code is required")
        # Auto-resolve language name
        if not self.tgt_lang_name:
            self.tgt_lang_name = LANG_NAMES.get(self.tgt_lang_code, self.tgt_lang_code)
        # Expand ~ in paths
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())

    # ── Derived paths ────────────────────────────────────────────────────

    @property
    def lang_data_dir(self) -> str:
        """Per-language root directory: ``output_dir / tgt_lang_code``."""
        return str(Path(self.output_dir) / self.tgt_lang_code)

    @property
    def fineweb2_raw_dir(self) -> str:
        return str(Path(self.lang_data_dir) / "cpt" / "raw")

    @property
    def fineweb2_raw_jsonl(self) -> str:
        return str(Path(self.fineweb2_raw_dir) / f"{self.tgt_lang_code}_train.jsonl")

    @property
    def fineweb2_refined_dir(self) -> str:
        return str(Path(self.lang_data_dir) / "cpt" / "refined")

    @property
    def fineweb2_refined_jsonl(self) -> str:
        return str(Path(self.fineweb2_refined_dir) / f"{self.tgt_lang_code}_train_refined.jsonl")

    @property
    def finewebedu_parquet_dir(self) -> str:
        return self.finewebedu.parquet_dir or str(
            Path(self.output_dir) / "_shared" / "eng_fineweb_edu_raw"
        )

    @property
    def finewebedu_subset_dir(self) -> str:
        return str(Path(self.output_dir) / "_shared" / "eng_fineweb_edu_raw")

    @property
    def finewebedu_subset_jsonl(self) -> str:
        return str(Path(self.finewebedu_subset_dir) / "eng_train.jsonl")

    @property
    def finewebedu_refined_dir(self) -> str:
        return str(Path(self.output_dir) / "_shared" / "eng_fineweb_edu_refined")

    @property
    def finewebedu_refined_jsonl(self) -> str:
        return str(Path(self.finewebedu_refined_dir) / "eng_train_refined.jsonl")

    @property
    def finewebedu_translated_dir(self) -> str:
        return str(Path(self.lang_data_dir) / "cpt" / "translated")

    @property
    def finewebedu_translated_jsonl(self) -> str:
        return str(Path(self.finewebedu_translated_dir) / f"eng2{self.tgt_lang_code}_translated.jsonl")

    # ── Bilingual mix paths ──────────────────────────────────────────────

    @property
    def _lang_alias(self) -> str:
        """ISO 639-3 code used in filenames."""
        return self.tgt_lang_code

    @property
    def bilingual_mix_dir(self) -> str:
        """Directory for the bilingual mix output."""
        return str(Path(self.lang_data_dir) / "cpt" / "bilingual_mix")

    @property
    def bilingual_mix_jsonl(self) -> str:
        """Path to the bilingual mix JSONL file."""
        return str(
            Path(self.bilingual_mix_dir)
            / f"{self.tgt_lang_code}_english_cpt.jsonl"
        )

    @property
    def bilingual_mix_dataset_info(self) -> str:
        """Path to the dataset_info.json inside bilingual_mix/."""
        return str(Path(self.bilingual_mix_dir) / "dataset_info.json")

    @property
    def bilingual_mix_dataset_name(self) -> str:
        """LlamaFactory dataset name for the bilingual mix."""
        return f"{self.tgt_lang_code}_english_cpt"

    # ── Serialisation ────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CPTDataPrepConfig":
        """Load from a YAML file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CPTDataPrepConfig":
        """Create from a flat or nested dictionary."""
        # Extract step flags from a nested "steps" key, if present
        steps = data.pop("steps", {})

        # Build sub-configs
        fineweb2 = FineWeb2Config(**data.pop("fineweb2", {}))
        finewebedu = FineWebEduConfig(**data.pop("finewebedu", {}))
        subset = SubsetConfig(**data.pop("subset", {}))

        refine_tgt = data.pop("refine_tgt_lang_config", {})
        refine_eng = data.pop("refine_eng_config", {})
        translate = data.pop("translate_config", {})

        # Parse extra sources
        raw_extras = data.pop("extra_sources", [])
        extra_sources = [ExtraSource(**e) for e in raw_extras] if raw_extras else []

        # Strip snapshot_percents if present (no longer used)
        for sub in (refine_tgt, refine_eng, translate):
            sub.pop("snapshot_percents", None)

        # Merge step flags into top-level
        merged = {**data, **steps}

        return cls(
            tgt_lang_code=merged.get("tgt_lang_code", ""),
            tgt_lang_name=merged.get("tgt_lang_name", ""),
            overwrite=merged.get("overwrite", False),
            download_tgt_lang_fineweb2=merged.get("download_tgt_lang_fineweb2", True),
            refine_tgt_lang=merged.get("refine_tgt_lang", True),
            download_eng_finewebedu=merged.get("download_eng_finewebedu", True),
            refine_eng=merged.get("refine_eng", True),
            translate_eng_to_tgt_lang=merged.get("translate_eng_to_tgt_lang", True),
            max_samples=merged.get("max_samples", None),
            extra_sources=extra_sources,
            fineweb2=fineweb2,
            finewebedu=finewebedu,
            subset=subset,
            refine_tgt_lang_config=LLMStepConfig(**refine_tgt) if refine_tgt else LLMStepConfig(),
            refine_eng_config=LLMStepConfig(**refine_eng) if refine_eng else LLMStepConfig(
                model_name=DEFAULT_REFINE_ENGLISH_MODEL,
                reasoning_effort="minimal",
                batch_size=DEFAULT_ENGLISH_BATCH_SIZE,
                concurrency=DEFAULT_ENGLISH_CONCURRENCY,
                checkpoint_every=DEFAULT_ENGLISH_CHECKPOINT_EVERY,
                token_budget_per_item=DEFAULT_ENGLISH_TOKEN_BUDGET,
            ),
            translate_config=LLMStepConfig(**translate) if translate else LLMStepConfig(
                batch_size=DEFAULT_TRANSLATE_BATCH_SIZE,
                concurrency=DEFAULT_TRANSLATE_CONCURRENCY,
                checkpoint_every=DEFAULT_TRANSLATE_CHECKPOINT_EVERY,
                token_budget_per_item=DEFAULT_TRANSLATE_TOKEN_BUDGET,
            ),
            output_dir=merged.get("output_dir", DEFAULT_OUTPUT_DIR),
        )

    # ── Output serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict matching the expected YAML structure."""
        home = str(Path.home())
        output_dir_short = str(Path(self.output_dir)).replace(home, "~")

        d: Dict[str, Any] = {
            "tgt_lang_code": self.tgt_lang_code,
            "tgt_lang_name": self.tgt_lang_name,
            "output_dir": output_dir_short,
            "overwrite": self.overwrite,
            "steps": {
                "download_tgt_lang_fineweb2": self.download_tgt_lang_fineweb2,
                "refine_tgt_lang": self.refine_tgt_lang,
                "download_eng_finewebedu": self.download_eng_finewebedu,
                "refine_eng": self.refine_eng,
                "translate_eng_to_tgt_lang": self.translate_eng_to_tgt_lang,
            },
            "fineweb2": dataclasses.asdict(self.fineweb2),
            "finewebedu": {
                k: v
                for k, v in dataclasses.asdict(self.finewebedu).items()
                if v is not None
            },
            "subset": {
                k: v
                for k, v in dataclasses.asdict(self.subset).items()
                if v is not None
            },
            "refine_tgt_lang_config": _llm_config_to_dict(self.refine_tgt_lang_config),
            "refine_eng_config": _llm_config_to_dict(self.refine_eng_config),
            "translate_config": _llm_config_to_dict(self.translate_config),
        }

        # Resolve auto-derived dataset_name so it's visible in the config file
        if d["fineweb2"].get("dataset_name") is None:
            d["fineweb2"]["dataset_name"] = _guess_fineweb2_subset(self.tgt_lang_code)

        return d

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write config to a YAML file with a descriptive header comment."""
        header = (
            f"# BYOL CPT Data Preparation — "
            f"{self.tgt_lang_name} ({self.tgt_lang_code})\n"
        )
        _write_yaml(self.to_dict(), Path(path), header)


# ──────────────────────────────────────────────────────────────────────────────
# SFT Data Preparation Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SmolTalk2Config:
    """Configuration for which SmolTalk2 subsets to download."""

    datasets: List[Dict[str, Any]] = field(
        default_factory=lambda: list(SMOLTALK2_DATASETS_CONFIG)
    )


@dataclass
class AyaConfig:
    """Configuration for AYA dataset download and filtering."""

    source_language_codes: List[str] = field(
        default_factory=lambda: list(AYA_SOURCE_LANGUAGE_CODES)
    )
    cache_dir: str = ""  # auto-derived if blank


@dataclass
class SFTDataPrepConfig:
    """Full SFT data preparation configuration.

    Controls which pipeline steps run and their parameters.

    Example YAML::

        tgt_lang_code: nya
        tgt_lang_name: Chichewa

        steps:
          download_smoltalk2: true
          translate_smoltalk2: true
          download_aya: true
          translate_aya: true
    """

    # ── Language ─────────────────────────────────────────────────────────
    tgt_lang_code: str = ""
    tgt_lang_name: str = ""  # auto-resolved from code if blank

    # ── Overwrite existing outputs ───────────────────────────────────────
    overwrite: bool = False

    # ── Pipeline step flags ──────────────────────────────────────────────
    download_smoltalk2: bool = True
    translate_smoltalk2: bool = True
    download_aya: bool = True
    translate_aya: bool = True

    # ── Sampling limit (for quick testing) ──────────────────────
    max_samples: Optional[int] = None  # if set, cap rows per step

    # ── Extra user-supplied data sources ─────────────────────────────────
    extra_sources: List[ExtraSource] = field(default_factory=list)

    # ── Sub-configs ──────────────────────────────────────────────────────
    smoltalk2: SmolTalk2Config = field(default_factory=SmolTalk2Config)
    aya: AyaConfig = field(default_factory=AyaConfig)

    translate_smoltalk2_config: LLMStepConfig = field(
        default_factory=lambda: LLMStepConfig(
            model_name=DEFAULT_MODEL_NAME,
            reasoning_effort="low",
            batch_size=DEFAULT_SFT_TRANSLATE_BATCH_SIZE,
            concurrency=DEFAULT_SFT_TRANSLATE_CONCURRENCY,
            checkpoint_every=DEFAULT_SFT_TRANSLATE_CHECKPOINT_EVERY,
            token_budget_per_item=DEFAULT_SFT_TRANSLATE_TOKEN_BUDGET,
        )
    )
    translate_aya_config: LLMStepConfig = field(
        default_factory=lambda: LLMStepConfig(
            model_name=DEFAULT_MODEL_NAME,
            reasoning_effort="low",
            batch_size=DEFAULT_SFT_TRANSLATE_BATCH_SIZE,
            concurrency=DEFAULT_SFT_TRANSLATE_CONCURRENCY,
            checkpoint_every=DEFAULT_SFT_TRANSLATE_CHECKPOINT_EVERY,
            token_budget_per_item=DEFAULT_SFT_TRANSLATE_TOKEN_BUDGET,
        )
    )

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not self.tgt_lang_code:
            raise ValueError("tgt_lang_code is required")
        if not self.tgt_lang_name:
            self.tgt_lang_name = LANG_NAMES.get(self.tgt_lang_code, self.tgt_lang_code)
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        if not self.aya.cache_dir:
            self.aya.cache_dir = str(Path(self.output_dir) / "_shared" / "aya_cache")

    # ── Derived paths ────────────────────────────────────────────────────

    @property
    def sft_data_dir(self) -> str:
        """Per-language SFT working directory: ``output_dir / tgt_lang_code / sft``."""
        return str(Path(self.output_dir) / self.tgt_lang_code / "sft")

    # -- SmolTalk2 paths --

    @property
    def smoltalk2_raw_dir(self) -> str:
        return str(Path(self.sft_data_dir) / "smoltalk2_english")

    @property
    def smoltalk2_combined_jsonl(self) -> str:
        return str(Path(self.smoltalk2_raw_dir) / "smoltalk_combined.jsonl")

    @property
    def smoltalk2_translated_dir(self) -> str:
        return str(Path(self.sft_data_dir) / "smoltalk2_translated")

    @property
    def smoltalk2_translated_jsonl(self) -> str:
        return str(
            Path(self.smoltalk2_translated_dir)
            / f"smoltalk2_subset_translated_to_{self.tgt_lang_code}_using_gpt5.jsonl"
        )

    # -- AYA paths --

    @property
    def aya_raw_dir(self) -> str:
        return str(Path(self.sft_data_dir) / "aya_dataset")

    @property
    def aya_filtered_train_jsonl(self) -> str:
        return str(Path(self.aya_raw_dir) / "aya_filtered_train.jsonl")

    @property
    def aya_filtered_test_jsonl(self) -> str:
        return str(Path(self.aya_raw_dir) / "aya_filtered_test.jsonl")

    @property
    def aya_translated_dir(self) -> str:
        return str(Path(self.sft_data_dir) / "aya_translated")

    @property
    def aya_translated_train_jsonl(self) -> str:
        return str(
            Path(self.aya_translated_dir)
            / f"aya_dataset_translated_to_{self.tgt_lang_code}_train.jsonl"
        )

    @property
    def aya_translated_test_jsonl(self) -> str:
        return str(
            Path(self.aya_translated_dir)
            / f"aya_dataset_translated_to_{self.tgt_lang_code}_test.jsonl"
        )

    # -- Bilingual mix paths --

    @property
    def _lang_alias(self) -> str:
        """ISO 639-3 code used in filenames."""
        return self.tgt_lang_code

    @property
    def bilingual_mix_dir(self) -> str:
        """Directory for the bilingual mix output."""
        return str(Path(self.sft_data_dir) / "bilingual_mix")

    @property
    def bilingual_mix_jsonl(self) -> str:
        """Path to the bilingual mix JSONL file for SFT training."""
        return str(
            Path(self.bilingual_mix_dir)
            / f"{self.tgt_lang_code}_english_sft.jsonl"
        )

    @property
    def bilingual_mix_test_jsonl(self) -> str:
        """Path to the SFT test JSONL file."""
        return str(
            Path(self.bilingual_mix_dir)
            / f"{self.tgt_lang_code}_sft_test.jsonl"
        )

    @property
    def bilingual_mix_dataset_info(self) -> str:
        """Path to the dataset_info.json inside bilingual_mix/."""
        return str(Path(self.bilingual_mix_dir) / "dataset_info.json")

    @property
    def bilingual_mix_dataset_name(self) -> str:
        """LlamaFactory dataset name for the SFT bilingual mix."""
        return f"{self.tgt_lang_code}_english_sft"

    @property
    def bilingual_mix_test_dataset_name(self) -> str:
        """LlamaFactory dataset name for the SFT test set."""
        return f"{self.tgt_lang_code}_sft_test"

    # ── Serialisation ────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SFTDataPrepConfig":
        """Load from a YAML file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTDataPrepConfig":
        """Create from a flat or nested dictionary."""
        steps = data.pop("steps", {})

        # Build sub-configs
        smoltalk2_data = data.pop("smoltalk2", {})
        aya_data = data.pop("aya", {})
        translate_smoltalk2 = data.pop("translate_smoltalk2_config", {})
        translate_aya = data.pop("translate_aya_config", {})

        # Parse extra sources
        raw_extras = data.pop("extra_sources", [])
        extra_sources = [ExtraSource(**e) for e in raw_extras] if raw_extras else []

        # Strip unsupported keys
        for sub in (translate_smoltalk2, translate_aya):
            sub.pop("snapshot_percents", None)

        merged = {**data, **steps}

        smoltalk2_cfg = SmolTalk2Config(**smoltalk2_data) if smoltalk2_data else SmolTalk2Config()
        aya_cfg = AyaConfig(**aya_data) if aya_data else AyaConfig()

        return cls(
            tgt_lang_code=merged.get("tgt_lang_code", ""),
            tgt_lang_name=merged.get("tgt_lang_name", ""),
            overwrite=merged.get("overwrite", False),
            download_smoltalk2=merged.get("download_smoltalk2", True),
            translate_smoltalk2=merged.get("translate_smoltalk2", True),
            download_aya=merged.get("download_aya", True),
            translate_aya=merged.get("translate_aya", True),
            max_samples=merged.get("max_samples", None),
            extra_sources=extra_sources,
            smoltalk2=smoltalk2_cfg,
            aya=aya_cfg,
            translate_smoltalk2_config=(
                LLMStepConfig(**translate_smoltalk2)
                if translate_smoltalk2
                else LLMStepConfig(
                    batch_size=DEFAULT_SFT_TRANSLATE_BATCH_SIZE,
                    concurrency=DEFAULT_SFT_TRANSLATE_CONCURRENCY,
                    checkpoint_every=DEFAULT_SFT_TRANSLATE_CHECKPOINT_EVERY,
                    token_budget_per_item=DEFAULT_SFT_TRANSLATE_TOKEN_BUDGET,
                )
            ),
            translate_aya_config=(
                LLMStepConfig(**translate_aya)
                if translate_aya
                else LLMStepConfig(
                    batch_size=DEFAULT_SFT_TRANSLATE_BATCH_SIZE,
                    concurrency=DEFAULT_SFT_TRANSLATE_CONCURRENCY,
                    checkpoint_every=DEFAULT_SFT_TRANSLATE_CHECKPOINT_EVERY,
                    token_budget_per_item=DEFAULT_SFT_TRANSLATE_TOKEN_BUDGET,
                )
            ),
            output_dir=merged.get("output_dir", DEFAULT_OUTPUT_DIR),
            seed=merged.get("seed", DEFAULT_SEED),
        )

    # ── Output serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict matching the expected YAML structure."""
        home = str(Path.home())
        output_dir_short = str(Path(self.output_dir)).replace(home, "~")

        d: Dict[str, Any] = {
            "tgt_lang_code": self.tgt_lang_code,
            "tgt_lang_name": self.tgt_lang_name,
            "output_dir": output_dir_short,
            "overwrite": self.overwrite,
            "steps": {
                "download_smoltalk2": self.download_smoltalk2,
                "translate_smoltalk2": self.translate_smoltalk2,
                "download_aya": self.download_aya,
                "translate_aya": self.translate_aya,
            },
            "translate_smoltalk2_config": _llm_config_to_dict(self.translate_smoltalk2_config),
            "translate_aya_config": _llm_config_to_dict(self.translate_aya_config),
            "seed": self.seed,
        }

        return d

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write config to a YAML file with a descriptive header comment."""
        header = (
            f"# BYOL SFT Data Preparation — "
            f"{self.tgt_lang_name} ({self.tgt_lang_code})\n"
        )
        _write_yaml(self.to_dict(), Path(path), header)


# ──────────────────────────────────────────────────────────────────────────────
# Eval Data Preparation Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EvalBenchmarkConfig:
    """Configuration for a single eval benchmark to translate.

    Attributes:
        name: Canonical benchmark name (matches keys in ``EVAL_BENCHMARK_DEFAULTS``).
        enabled: Whether to translate this benchmark (default ``True``).
        translator: Override the translator model for this benchmark.
        max_tokens: Max tokens for LLM translators (ignored for API translators).
        splits: Override splits to process.  ``None`` → use defaults.
        fields: Override fields to translate.  ``None`` → use defaults.
    """

    name: str = ""
    enabled: bool = True
    translator: Optional[str] = None  # None → inherit from benchmark defaults
    max_tokens: Optional[int] = None  # None → inherit from global config
    splits: Optional[List[str]] = None  # None → use benchmark defaults
    fields: Optional[List[str]] = None  # None → use benchmark defaults


@dataclass
class EvalDataPrepConfig:
    """Full eval data preparation configuration.

    Controls translation of evaluation benchmarks from English to a target language.

    Example YAML::

        tgt_lang_code: nya
        tgt_lang_name: Chichewa

        benchmarks:
          - name: copa
            enabled: true
          - name: ai2_arc_hard
            enabled: true
          - name: mmlu_lite
            translator: gpt-5
            max_tokens: 2048
    """

    # ── Language ─────────────────────────────────────────────────────────
    tgt_lang_code: str = ""
    tgt_lang_name: str = ""  # auto-resolved from code if blank

    # ── Overwrite existing outputs ───────────────────────────────────────
    overwrite: bool = False

    # ── Sampling limit (for quick testing) ──────────────────────
    max_samples: Optional[int] = None  # if set, cap rows per benchmark split

    # ── Benchmark list ───────────────────────────────────────────────────
    #    If empty, all benchmarks from EVAL_BENCHMARK_DEFAULTS are enabled.
    benchmarks: List[EvalBenchmarkConfig] = field(default_factory=list)

    # ── Translation parameters ───────────────────────────────────────────
    max_workers: int = DEFAULT_EVAL_MAX_WORKERS
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    default_translator: str = DEFAULT_EVAL_TRANSLATOR
    default_llm_max_tokens: int = DEFAULT_EVAL_LLM_MAX_TOKENS

    # ── Device (GPU) ─────────────────────────────────────────────────────
    device: Optional[str] = None  # e.g. "3" or "2,3" for multi-GPU

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = ""  # auto-derived if blank: DEFAULT_EVAL_OUTPUT_DIR / {tgt_lang_name}

    def __post_init__(self) -> None:
        if not self.tgt_lang_code:
            raise ValueError("tgt_lang_code is required")
        if not self.tgt_lang_name:
            self.tgt_lang_name = LANG_NAMES.get(self.tgt_lang_code, self.tgt_lang_code)
        # Auto-derive output dir: output_dir / tgt_lang_code / eval
        if not self.output_dir:
            self.output_dir = str(
                Path(DEFAULT_OUTPUT_DIR) / self.tgt_lang_code / "eval"
            )
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        # If no benchmarks specified, enable all defaults
        if not self.benchmarks:
            self.benchmarks = [
                EvalBenchmarkConfig(name=name)
                for name in EVAL_BENCHMARK_DEFAULTS
            ]

    @property
    def _lang_alias(self) -> str:
        """ISO 639-3 code used in filenames."""
        return self.tgt_lang_code

    def get_output_filename(
        self, benchmark_name: str, split: str, translator_suffix: str
    ) -> str:
        """Build the output JSONL filename for a translated benchmark split.

        Convention (matches existing eval data):
            ``{benchmark}_{split}_english2{lang_code}_{suffix}_translated.jsonl``
        """
        return (
            f"{benchmark_name}_{split}_english2{self._lang_alias}"
            f"_{translator_suffix}_translated.jsonl"
        )

    # ── Serialisation ────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvalDataPrepConfig":
        """Load from a YAML file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalDataPrepConfig":
        """Create from a flat or nested dictionary."""
        # Parse benchmark list
        raw_benchmarks = data.pop("benchmarks", [])
        benchmarks = [
            EvalBenchmarkConfig(**b) if isinstance(b, dict) else b
            for b in raw_benchmarks
        ]

        return cls(
            tgt_lang_code=data.get("tgt_lang_code", ""),
            tgt_lang_name=data.get("tgt_lang_name", ""),
            overwrite=data.get("overwrite", False),
            max_samples=data.get("max_samples", None),
            benchmarks=benchmarks,
            max_workers=data.get("max_workers", DEFAULT_EVAL_MAX_WORKERS),
            batch_size=data.get("batch_size", DEFAULT_EVAL_BATCH_SIZE),
            default_translator=data.get("default_translator", DEFAULT_EVAL_TRANSLATOR),
            default_llm_max_tokens=data.get(
                "default_llm_max_tokens", DEFAULT_EVAL_LLM_MAX_TOKENS
            ),
            output_dir=data.get("output_dir", ""),
        )

    # ── Output serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict matching the expected YAML structure."""
        d: Dict[str, Any] = {
            "tgt_lang_code": self.tgt_lang_code,
            "tgt_lang_name": self.tgt_lang_name,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "default_translator": self.default_translator,
            "default_llm_max_tokens": self.default_llm_max_tokens,
        }

        benchmarks_list: List[Dict[str, Any]] = []
        for bm in self.benchmarks:
            bm_d: Dict[str, Any] = {"name": bm.name, "enabled": bm.enabled}
            if bm.translator is not None:
                bm_d["translator"] = bm.translator
            if bm.max_tokens is not None:
                bm_d["max_tokens"] = bm.max_tokens
            if bm.splits is not None:
                bm_d["splits"] = bm.splits
            if bm.fields is not None:
                bm_d["fields"] = bm.fields
            benchmarks_list.append(bm_d)

        d["benchmarks"] = benchmarks_list
        return d

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write config to a YAML file with a descriptive header comment."""
        header = (
            f"# BYOL Eval Data Preparation — "
            f"{self.tgt_lang_name} ({self.tgt_lang_code})\n"
        )
        _write_yaml(self.to_dict(), Path(path), header)
