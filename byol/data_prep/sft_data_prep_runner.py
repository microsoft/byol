# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SFT data-preparation pipeline runner.

Orchestrates the SFT data-preparation steps in sequence, respecting
the boolean flags in :class:`SFTDataPrepConfig`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import SFTDataPrepConfig
from .prompts import get_translate_aya_prompt, get_translate_smoltalk2_prompt

logger = logging.getLogger("byol-data-prep")


@dataclass
class SFTDataPrepResult:
    """Result of an SFT data-preparation run."""

    success: bool
    stage: str
    tgt_lang_code: str
    output_files: dict[str, str]
    error: Optional[str] = None
    duration_seconds: float = 0.0


class SFTDataPrepRunner:
    """Run the SFT data-preparation pipeline.

    Steps (each gated by config flags):

    1. ``download_smoltalk2`` — download & combine SmolTalk2 subsets
    2. ``translate_smoltalk2``  — translate SmolTalk2 conversations → target language
    3. ``download_aya``       — download & filter AYA dataset
    4. ``translate_aya``      — translate AYA entries → target language
    5. ``create_sft_mix``     — combine all sources into training-ready JSONL
    """

    def __init__(self, config: SFTDataPrepConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._output_files: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> SFTDataPrepResult:
        """Execute the full SFT pipeline and return a result summary."""
        start = datetime.now()
        cfg = self.config

        logger.info("=" * 60)
        logger.info("BYOL DATA PREPARATION — SFT")
        logger.info(f"  Target language : {cfg.tgt_lang_name} ({cfg.tgt_lang_code})")
        logger.info(f"  Output dir      : {cfg.output_dir}")
        logger.info(f"  SFT data dir    : {cfg.sft_data_dir}")
        logger.info(f"  Dry run         : {self.dry_run}")
        if cfg.max_samples is not None:
            logger.info(f"  Max samples     : {cfg.max_samples}")
        logger.info(f"  Overwrite       : {cfg.overwrite}")
        if cfg.extra_sources:
            logger.info(f"  Extra sources   : {len(cfg.extra_sources)}")
            for es in cfg.extra_sources:
                logger.info(f"    {es.dataset_tag:20s} ({es.format}) → {es.path}")
        logger.info("  Steps enabled:")
        for step_name in (
            "download_smoltalk2",
            "translate_smoltalk2",
            "download_aya",
            "translate_aya",
        ):
            flag = getattr(cfg, step_name)
            logger.info(f"    {step_name:30s} {'ON' if flag else 'OFF'}")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("[DRY RUN] Would execute the steps above. Exiting.")
            return SFTDataPrepResult(
                success=True,
                stage="sft",
                tgt_lang_code=cfg.tgt_lang_code,
                output_files={},
            )

        try:
            self._run_pipeline()
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            logger.exception("SFT pipeline failed")
            return SFTDataPrepResult(
                success=False,
                stage="sft",
                tgt_lang_code=cfg.tgt_lang_code,
                output_files=self._output_files,
                error=str(e),
                duration_seconds=duration,
            )

        duration = (datetime.now() - start).total_seconds()
        logger.info(f"SFT pipeline complete in {duration:.1f}s")
        self._print_summary()

        return SFTDataPrepResult(
            success=True,
            stage="sft",
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

        # ── 1. Download & combine SmolTalk2 subsets ──────────────────────
        if cfg.download_smoltalk2:
            self._step_download_smoltalk2()

        # ── 2. Translate SmolTalk2 → target language ─────────────────────
        if cfg.translate_smoltalk2:
            self._step_translate_smoltalk2()

        # ── 3. Download & filter AYA dataset ─────────────────────────────
        if cfg.download_aya:
            self._step_download_aya()

        # ── 4. Translate AYA → target language ───────────────────────────
        if cfg.translate_aya:
            self._step_translate_aya()

        # ── 5. Create SFT bilingual mix (always runs) ───────────────────
        self._step_create_sft_mix()

    # ──────────────────────────────────────────────────────────────────────
    # Individual steps
    # ──────────────────────────────────────────────────────────────────────

    def _step_download_smoltalk2(self) -> None:
        logger.info("─── Step 1: Download & combine SmolTalk2 subsets ───")
        from .steps.download_smoltalk2 import download_smoltalk2

        cfg = self.config

        if not cfg.overwrite and os.path.exists(cfg.smoltalk2_combined_jsonl):
            logger.info(
                f"SKIPPED: Combined SmolTalk2 already exists at {cfg.smoltalk2_combined_jsonl}. "
                "Use --overwrite to re-download."
            )
            self._output_files["smoltalk2_combined"] = cfg.smoltalk2_combined_jsonl
            return

        out = download_smoltalk2(
            output_dir=cfg.smoltalk2_raw_dir,
            output_jsonl=cfg.smoltalk2_combined_jsonl,
            datasets_config=cfg.smoltalk2.datasets,
            seed=cfg.seed,
            max_samples=cfg.max_samples,
        )
        self._output_files["smoltalk2_combined"] = out

    def _step_translate_smoltalk2(self) -> None:
        logger.info("─── Step 2: Translate SmolTalk2 → target language ───")
        from .steps.translate_smoltalk2 import run_translate_smoltalk2

        cfg = self.config
        sc = cfg.translate_smoltalk2_config
        prompt = get_translate_smoltalk2_prompt(cfg.tgt_lang_name, cfg.tgt_lang_code)

        if not cfg.overwrite and os.path.exists(cfg.smoltalk2_translated_jsonl):
            logger.info(
                f"SKIPPED: Translated SmolTalk2 already exists at {cfg.smoltalk2_translated_jsonl}. "
                "Use --overwrite to re-run."
            )
            self._output_files["smoltalk2_translated"] = cfg.smoltalk2_translated_jsonl
            return

        input_file = cfg.smoltalk2_combined_jsonl
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Cannot translate SmolTalk2: input not found at {input_file}. "
                "Run with download_smoltalk2=true first."
            )

        out = run_translate_smoltalk2(
            input_file=input_file,
            output_file=cfg.smoltalk2_translated_jsonl,
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
        self._output_files["smoltalk2_translated"] = out

    def _step_download_aya(self) -> None:
        logger.info("─── Step 3: Download & filter AYA dataset ───")
        from .steps.download_aya import download_aya

        cfg = self.config

        if (
            not cfg.overwrite
            and os.path.exists(cfg.aya_filtered_train_jsonl)
            and os.path.exists(cfg.aya_filtered_test_jsonl)
        ):
            logger.info(
                f"SKIPPED: AYA filtered data already exists at {cfg.aya_raw_dir}. "
                "Use --overwrite to re-download."
            )
            self._output_files["aya_filtered_train"] = cfg.aya_filtered_train_jsonl
            self._output_files["aya_filtered_test"] = cfg.aya_filtered_test_jsonl
            return

        train_out, test_out = download_aya(
            tgt_lang_code=cfg.tgt_lang_code,
            output_dir=cfg.aya_raw_dir,
            train_jsonl=cfg.aya_filtered_train_jsonl,
            test_jsonl=cfg.aya_filtered_test_jsonl,
            source_language_codes=cfg.aya.source_language_codes,
            cache_dir=cfg.aya.cache_dir,
            max_samples=cfg.max_samples,
        )
        self._output_files["aya_filtered_train"] = train_out
        self._output_files["aya_filtered_test"] = test_out

    def _step_translate_aya(self) -> None:
        logger.info("─── Step 4: Translate AYA → target language ───")
        from .steps.translate_aya import run_translate_aya

        cfg = self.config
        sc = cfg.translate_aya_config
        prompt = get_translate_aya_prompt(cfg.tgt_lang_name, cfg.tgt_lang_code)

        if (
            not cfg.overwrite
            and os.path.exists(cfg.aya_translated_train_jsonl)
            and os.path.exists(cfg.aya_translated_test_jsonl)
        ):
            logger.info(
                f"SKIPPED: AYA translated data already exists at {cfg.aya_translated_dir}. "
                "Use --overwrite to re-run."
            )
            self._output_files["aya_translated_train"] = cfg.aya_translated_train_jsonl
            self._output_files["aya_translated_test"] = cfg.aya_translated_test_jsonl
            return

        train_input = cfg.aya_filtered_train_jsonl
        test_input = cfg.aya_filtered_test_jsonl
        if not os.path.exists(train_input):
            raise FileNotFoundError(
                f"Cannot translate AYA: filtered train not found at {train_input}. "
                "Run with download_aya=true first."
            )

        train_out, test_out = run_translate_aya(
            train_input_file=train_input,
            test_input_file=test_input,
            train_output_file=cfg.aya_translated_train_jsonl,
            test_output_file=cfg.aya_translated_test_jsonl,
            tgt_lang_code=cfg.tgt_lang_code,
            system_prompt=prompt,
            model_name=sc.model_name,
            reasoning_effort=sc.reasoning_effort,
            api_version=sc.api_version,
            batch_size=sc.batch_size,
            concurrency=sc.concurrency,
            checkpoint_every=sc.checkpoint_every,
            token_budget_per_item=sc.token_budget_per_item,
            max_completion_tokens=sc.max_completion_tokens,
            source_language_codes=cfg.aya.source_language_codes,
            max_samples=cfg.max_samples,
        )
        self._output_files["aya_translated_train"] = train_out
        self._output_files["aya_translated_test"] = test_out

    def _step_create_sft_mix(self) -> None:
        logger.info("─── Step 5: Create SFT bilingual mix ───")
        from .steps.sft_bilingual_mix import create_sft_bilingual_mix

        cfg = self.config

        out = create_sft_bilingual_mix(
            smoltalk2_eng_jsonl=cfg.smoltalk2_combined_jsonl,
            smoltalk2_translated_jsonl=cfg.smoltalk2_translated_jsonl,
            aya_translated_train_jsonl=cfg.aya_translated_train_jsonl,
            aya_translated_test_jsonl=cfg.aya_translated_test_jsonl,
            output_jsonl=cfg.bilingual_mix_jsonl,
            test_output_jsonl=cfg.bilingual_mix_test_jsonl,
            dataset_info_path=cfg.bilingual_mix_dataset_info,
            dataset_name=cfg.bilingual_mix_dataset_name,
            test_dataset_name=cfg.bilingual_mix_test_dataset_name,
            lang_code=cfg._lang_alias,
            tgt_lang_code=cfg.tgt_lang_code,
            seed=cfg.seed,
            max_samples=cfg.max_samples,
            overwrite=cfg.overwrite,
            extra_sources=cfg.extra_sources,
        )
        self._output_files["sft_bilingual_mix"] = out

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        cfg = self.config
        logger.info("=" * 60)
        logger.info("SFT DATA PREP — SUMMARY")
        logger.info(f"  Language: {cfg.tgt_lang_name} ({cfg.tgt_lang_code})")
        logger.info("  Output files:")
        for key, path in self._output_files.items():
            exists = os.path.exists(path) if path else False
            logger.info(f"    {key:30s} {'✓' if exists else '✗'}  {path}")
        logger.info("")
        logger.info("  The SFT corpus consists of:")
        logger.info("    (1) English SmolTalk2 instruction conversations")
        logger.info("    (2) SmolTalk2 conversations translated to target language")
        logger.info("    (3) AYA dataset entries (native + translated) converted to messages")
        logger.info("")
        logger.info("  Bilingual mix (training-ready):")
        logger.info(f"    Train JSONL    : {cfg.bilingual_mix_jsonl}")
        logger.info(f"    Test JSONL     : {cfg.bilingual_mix_test_jsonl}")
        logger.info(f"    dataset_info   : {cfg.bilingual_mix_dataset_info}")
        logger.info(f"    dataset name   : {cfg.bilingual_mix_dataset_name}")
        logger.info(f"    test dataset   : {cfg.bilingual_mix_test_dataset_name}")
        logger.info("")
        logger.info("  Next step → python -m byol.train sft \\")
        logger.info(f"      --dataset {cfg.bilingual_mix_dataset_name} \\")
        logger.info(f"      --eval-dataset {cfg.bilingual_mix_test_dataset_name} \\")
        logger.info(f"      --override dataset_dir={cfg.bilingual_mix_dir}")
        logger.info("=" * 60)
