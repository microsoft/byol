# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CPT data-preparation pipeline runner.

Orchestrates the CPT data-preparation steps in sequence, respecting
the boolean flags in :class:`CPTDataPrepConfig`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CPTDataPrepConfig
from .prompts import REFINE_ENG_PROMPT, get_refine_tgt_lang_prompt, get_translate_prompt

logger = logging.getLogger("byol-data-prep")


@dataclass
class CPTDataPrepResult:
    """Result of a CPT data-preparation run."""

    success: bool
    stage: str
    tgt_lang_code: str
    output_files: dict[str, str]
    error: Optional[str] = None
    duration_seconds: float = 0.0



class CPTDataPrepRunner:
    """Run the CPT data-preparation pipeline.

    Steps (each gated by config flags):

    1. ``download_tgt_lang_fineweb2`` — download FineWeb-2 target-language data
    2. ``refine_tgt_lang``         — refine target-language text via LLM
    3. ``download_eng_finewebedu`` — download FineWeb-Edu, extract token-matched subset
    4. ``refine_eng``              — clean/refine English text via LLM
    5. ``translate_eng_to_tgt_lang`` — translate refined English → target language
    6. ``create_bilingual_mix``    — concatenate, shuffle, and output training-ready JSONL
    """

    def __init__(self, config: CPTDataPrepConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._output_files: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> CPTDataPrepResult:
        """Execute the full pipeline and return a result summary."""
        start = datetime.now()
        cfg = self.config

        logger.info("=" * 60)
        logger.info("BYOL DATA PREPARATION — CPT")
        logger.info(f"  Target language : {cfg.tgt_lang_name} ({cfg.tgt_lang_code})")
        logger.info(f"  Output dir      : {cfg.output_dir}")
        logger.info(f"  Lang data dir   : {cfg.lang_data_dir}")
        logger.info(f"  Dry run         : {self.dry_run}")
        if cfg.max_samples is not None:
            logger.info(f"  Max samples     : {cfg.max_samples}")
        logger.info(f"  Overwrite       : {cfg.overwrite}")
        if cfg.extra_sources:
            logger.info(f"  Extra sources   : {len(cfg.extra_sources)}")
        logger.info("  Steps enabled:")
        for step_name in (
            "download_tgt_lang_fineweb2", "refine_tgt_lang",
            "download_eng_finewebedu",
            "refine_eng", "translate_eng_to_tgt_lang",
        ):
            flag = getattr(cfg, step_name)
            logger.info(f"    {step_name:30s} {'ON' if flag else 'OFF'}")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("[DRY RUN] Would execute the steps above. Exiting.")
            return CPTDataPrepResult(
                success=True, stage="cpt",
                tgt_lang_code=cfg.tgt_lang_code,
                output_files={},
            )

        try:
            self._run_pipeline()
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            logger.exception("Pipeline failed")
            return CPTDataPrepResult(
                success=False, stage="cpt",
                tgt_lang_code=cfg.tgt_lang_code,
                output_files=self._output_files,
                error=str(e), duration_seconds=duration,
            )

        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Pipeline complete in {duration:.1f}s")
        self._print_summary()

        return CPTDataPrepResult(
            success=True, stage="cpt",
            tgt_lang_code=cfg.tgt_lang_code,
            output_files=self._output_files,
            duration_seconds=duration,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline orchestration
    # ──────────────────────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        cfg = self.config

        # Ensure data/<lang> symlink exists
        from .constants import ensure_data_symlink
        ensure_data_symlink(cfg.tgt_lang_code, cfg.output_dir)

        # ── 1. Download FineWeb-2 target-language data ───────────────────
        if cfg.download_tgt_lang_fineweb2:
            self._step_download_tgt_lang_fineweb2()

        # ── 2. Refine target-language text ───────────────────────────────
        if cfg.refine_tgt_lang:
            self._step_refine_tgt_lang()

        # ── 3. Download FineWeb-Edu English data + extract subset ────
        if cfg.download_eng_finewebedu:
            self._step_download_eng_finewebedu()

        # ── 4. Refine English text ───────────────────────────────────────
        if cfg.refine_eng:
            self._step_refine_eng()

        # ── 5. Translate refined English → target language ───────────────
        if cfg.translate_eng_to_tgt_lang:
            self._step_translate_eng_to_tgt_lang()

        # ── 6. Create bilingual mix (always runs) ────────────────────────
        self._step_create_bilingual_mix()

    # ──────────────────────────────────────────────────────────────────────
    # Individual steps
    # ──────────────────────────────────────────────────────────────────────

    def _step_download_tgt_lang_fineweb2(self) -> None:
        logger.info("─── Step 1: Download FineWeb-2 ───")
        from .steps.download_tgt_lang_fineweb2 import download_tgt_lang_fineweb2

        cfg = self.config
        if not cfg.overwrite and os.path.exists(cfg.fineweb2_raw_jsonl):
            logger.info(
                f"SKIPPED: Output already exists at {cfg.fineweb2_raw_dir}. "
                "Use --overwrite to re-download."
            )
            self._output_files["fineweb2_raw"] = cfg.fineweb2_raw_dir
            return

        out = download_tgt_lang_fineweb2(
            lang_code=cfg.tgt_lang_code,
            output_dir=cfg.fineweb2_raw_dir,
            dataset_name=cfg.fineweb2.dataset_name,
            splits=tuple(cfg.fineweb2.splits),
            max_samples=cfg.max_samples,
        )
        self._output_files["fineweb2_raw"] = out

    def _step_refine_tgt_lang(self) -> None:
        logger.info("─── Step 2: Refine target-language text ───")
        from .steps.refine import run_refine

        cfg = self.config
        sc = cfg.refine_tgt_lang_config
        prompt = get_refine_tgt_lang_prompt(cfg.tgt_lang_name, cfg.tgt_lang_code)

        if not cfg.overwrite and os.path.exists(cfg.fineweb2_refined_jsonl):
            logger.info(
                f"SKIPPED: Output already exists at {cfg.fineweb2_refined_dir}. "
                "Use --overwrite to re-run."
            )
            self._output_files["fineweb2_refined"] = cfg.fineweb2_refined_jsonl
            return

        input_file = cfg.fineweb2_raw_jsonl
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Cannot refine: input not found at {input_file}. "
                "Run with download_tgt_lang_fineweb2=true first."
            )

        out = run_refine(
            mode="tgt_lang",
            input_file=input_file,
            output_file=cfg.fineweb2_refined_jsonl,
            system_prompt=prompt,
            model_name=sc.model_name,
            reasoning_effort=sc.reasoning_effort,
            api_version=sc.api_version,
            batch_size=sc.batch_size,
            concurrency=sc.concurrency,
            checkpoint_every=sc.checkpoint_every,
            token_budget_per_item=sc.token_budget_per_item,
            max_completion_tokens=sc.max_completion_tokens,
            max_samples=cfg.max_samples,
        )
        self._output_files["fineweb2_refined"] = out

    def _step_download_eng_finewebedu(self) -> None:
        logger.info("─── Step 3: Download FineWeb-Edu + extract subset ───")
        from .steps.download_finewebedu import download_and_extract_finewebedu

        cfg = self.config

        if not cfg.overwrite and os.path.exists(cfg.finewebedu_subset_jsonl):
            logger.info(
                f"SKIPPED: Subset already exists at {cfg.finewebedu_subset_jsonl}. "
                "Use --overwrite to re-download/re-extract."
            )
            self._output_files["finewebedu_subset"] = cfg.finewebedu_subset_jsonl
            return

        # Need the tgt_lang file for token counting
        tgt_lang_jsonl = cfg.fineweb2_raw_jsonl
        if not os.path.exists(tgt_lang_jsonl):
            raise FileNotFoundError(
                f"Cannot count target-language tokens: {tgt_lang_jsonl} not found. "
                "Run with download_tgt_lang_fineweb2=true first."
            )

        out = download_and_extract_finewebedu(
            parquet_dir=cfg.finewebedu_parquet_dir,
            output_jsonl=cfg.finewebedu_subset_jsonl,
            tgt_lang_jsonl=tgt_lang_jsonl,
            num_shards=cfg.finewebedu.num_shards,
            target_tokens_override=cfg.subset.target_tokens,
            max_token_count=cfg.subset.max_token_count,
            seed=cfg.subset.seed,
            max_samples=cfg.max_samples,
        )
        self._output_files["finewebedu_subset"] = out

    def _step_refine_eng(self) -> None:
        logger.info("─── Step 4: Refine English text ───")
        from .steps.refine import run_refine

        cfg = self.config
        sc = cfg.refine_eng_config

        if not cfg.overwrite and os.path.exists(cfg.finewebedu_refined_jsonl):
            logger.info(
                f"SKIPPED: Output already exists at {cfg.finewebedu_refined_dir}. "
                "Use --overwrite to re-run."
            )
            self._output_files["finewebedu_refined"] = cfg.finewebedu_refined_jsonl
            return

        input_file = cfg.finewebedu_subset_jsonl
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Cannot refine English: subset not found at {input_file}. "
                "Run with download_eng_finewebedu=true first."
            )

        out = run_refine(
            mode="eng",
            input_file=input_file,
            output_file=cfg.finewebedu_refined_jsonl,
            system_prompt=REFINE_ENG_PROMPT,
            model_name=sc.model_name,
            reasoning_effort=sc.reasoning_effort,
            api_version=sc.api_version,
            batch_size=sc.batch_size,
            concurrency=sc.concurrency,
            checkpoint_every=sc.checkpoint_every,
            token_budget_per_item=sc.token_budget_per_item,
            max_completion_tokens=sc.max_completion_tokens,
            max_samples=cfg.max_samples,
        )
        self._output_files["finewebedu_refined"] = out

    def _step_translate_eng_to_tgt_lang(self) -> None:
        logger.info("─── Step 5: Translate English → target language ───")
        from .steps.translate import run_translate

        cfg = self.config
        sc = cfg.translate_config
        prompt = get_translate_prompt(cfg.tgt_lang_name, cfg.tgt_lang_code)

        if not cfg.overwrite and os.path.exists(cfg.finewebedu_translated_jsonl):
            logger.info(
                f"SKIPPED: Output already exists at {cfg.finewebedu_translated_dir}. "
                "Use --overwrite to re-run."
            )
            self._output_files["finewebedu_translated"] = cfg.finewebedu_translated_jsonl
            return

        # Prefer refined English; fall back to subset
        input_file = cfg.finewebedu_refined_jsonl
        if not os.path.exists(input_file):
            input_file = cfg.finewebedu_subset_jsonl
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Cannot translate: no English input found. "
                "Run with refine_eng=true or download_eng_finewebedu=true first."
            )

        out = run_translate(
            input_file=input_file,
            output_file=cfg.finewebedu_translated_jsonl,
            system_prompt=prompt,
            model_name=sc.model_name,
            reasoning_effort=sc.reasoning_effort,
            api_version=sc.api_version,
            batch_size=sc.batch_size,
            concurrency=sc.concurrency,
            checkpoint_every=sc.checkpoint_every,
            token_budget_per_item=sc.token_budget_per_item,
            max_completion_tokens=sc.max_completion_tokens,
            max_samples=cfg.max_samples,
        )
        self._output_files["finewebedu_translated"] = out

    def _step_create_bilingual_mix(self) -> None:
        logger.info("─── Step 6: Create bilingual mix ───")
        from .steps.cpt_bilingual_mix import create_bilingual_mix

        cfg = self.config

        # Prefer refined files; fall back to raw.
        tgt_lang_jsonl = (
            cfg.fineweb2_refined_jsonl
            if os.path.exists(cfg.fineweb2_refined_jsonl)
            else cfg.fineweb2_raw_jsonl
        )
        eng_jsonl = (
            cfg.finewebedu_refined_jsonl
            if os.path.exists(cfg.finewebedu_refined_jsonl)
            else cfg.finewebedu_subset_jsonl
        )
        translated_jsonl = cfg.finewebedu_translated_jsonl

        out = create_bilingual_mix(
            tgt_lang_jsonl=tgt_lang_jsonl,
            eng_jsonl=eng_jsonl,
            translated_jsonl=translated_jsonl,
            output_jsonl=cfg.bilingual_mix_jsonl,
            dataset_info_path=cfg.bilingual_mix_dataset_info,
            dataset_name=cfg.bilingual_mix_dataset_name,
            lang_code=cfg._lang_alias,
            seed=cfg.subset.seed,
            max_samples=cfg.max_samples,
            overwrite=cfg.overwrite,
            extra_sources=cfg.extra_sources,
        )
        self._output_files["bilingual_mix"] = out

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        cfg = self.config
        logger.info("=" * 60)
        logger.info("CPT DATA PREP — SUMMARY")
        logger.info(f"  Language: {cfg.tgt_lang_name} ({cfg.tgt_lang_code})")
        logger.info("  Output files:")
        for key, path in self._output_files.items():
            exists = os.path.exists(path) if path else False
            logger.info(f"    {key:30s} {'✓' if exists else '✗'}  {path}")
        logger.info("")
        logger.info("  The CPT corpus consists of:")
        logger.info("    (1) Refined real target-language text from FineWeb-2")
        logger.info("    (2) Synthetic target-language data translated from FineWeb-Edu")
        logger.info("    (3) Refined real English data from FineWeb-Edu")
        logger.info("")
        logger.info("  Bilingual mix (training-ready):")
        logger.info(f"    JSONL         : {cfg.bilingual_mix_jsonl}")
        logger.info(f"    dataset_info  : {cfg.bilingual_mix_dataset_info}")
        logger.info(f"    dataset name  : {cfg.bilingual_mix_dataset_name}")
        logger.info("")
        logger.info("  Next step → python -m byol.train cpt \\")
        logger.info(f"      --dataset {cfg.bilingual_mix_dataset_name} \\")
        logger.info(f"      --override dataset_dir={cfg.bilingual_mix_dir}")
        logger.info("=" * 60)


