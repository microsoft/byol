# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Eval data-preparation pipeline runner.

Orchestrates translation of evaluation benchmarks from English to a
target language, using the ``byol.translation_backends`` package.

Each benchmark's data is loaded (from HuggingFace Hub or a local JSONL),
translated field-by-field, and written to a JSONL file that the
``byol.eval`` subpackage already expects.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import EvalBenchmarkConfig, EvalDataPrepConfig
from .constants import (
    BENCHMARK_SOURCE_DIR,
    EVAL_BENCHMARK_DEFAULTS,
    TRANSLATOR_SUFFIX_MAP,
)
from .prompts import get_eval_translate_prompt

from byol.common.translator_support import (
    is_language_supported,
    get_supported_translators,
)
from byol.common.exceptions import LanguageNotSupportedError

logger = logging.getLogger("byol-data-prep")

# Models that accept a system_prompt kwarg (LLM-based translators).
_LLM_TRANSLATOR_PREFIXES = ("gpt-", "deepseek-")


def _is_llm_translator(model: str) -> bool:
    """Return True for LLM-based translators that support system_prompt."""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _LLM_TRANSLATOR_PREFIXES)


def _translator_suffix(model: str) -> str:
    """Map a translator model name to a short suffix for filenames."""
    return TRANSLATOR_SUFFIX_MAP.get(model, model.replace("-", ""))


def _translate_benchmark_on_device(
    config: EvalDataPrepConfig,
    benchmark: EvalBenchmarkConfig,
    device_queue: "multiprocessing.Queue[str]",
) -> dict[str, str]:
    """Translate a single benchmark on a dynamically assigned GPU.

    This is a module-level function so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    import multiprocessing  # noqa: F811 – needed in subprocess

    device_id = device_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    logger.info("[worker:%s] Acquired GPU %s", benchmark.name, device_id)

    try:
        runner = EvalDataPrepRunner(config, dry_run=False)
        runner._translate_benchmark(benchmark)
        return runner._output_files
    finally:
        device_queue.put(device_id)
        logger.info("[worker:%s] Released GPU %s", benchmark.name, device_id)


@dataclass
class EvalDataPrepResult:
    """Result of an eval data-preparation run."""

    success: bool
    stage: str
    tgt_lang_code: str
    output_files: dict[str, str]
    error: Optional[str] = None
    duration_seconds: float = 0.0


class EvalDataPrepRunner:
    """Run the eval data-preparation pipeline.

    For each enabled benchmark:

    1. Load source data (HuggingFace or local JSONL).
    2. Translate the configured fields English → target language.
    3. Write the translated JSONL.
    """

    def __init__(self, config: EvalDataPrepConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._output_files: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> EvalDataPrepResult:
        """Execute the full eval pipeline and return a result summary."""
        start = datetime.now()
        cfg = self.config

        # Print plan
        logger.info("=" * 60)
        logger.info("BYOL DATA PREPARATION — EVAL")
        logger.info(f"  Target language : {cfg.tgt_lang_name} ({cfg.tgt_lang_code})")
        logger.info(f"  Output dir      : {cfg.output_dir}")
        logger.info(f"  Dry run         : {self.dry_run}")
        logger.info(f"  Max workers     : {cfg.max_workers}")
        logger.info(f"  Overwrite       : {cfg.overwrite}")
        if cfg.max_samples is not None:
            logger.info(f"  Max samples     : {cfg.max_samples}")
        logger.info("  Benchmarks:")
        for bm in cfg.benchmarks:
            status = "ON" if bm.enabled else "OFF"
            translator = bm.translator or self._resolve_translator(bm.name)
            logger.info(f"    {bm.name:30s} {status:3s}  (translator={translator})")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("[DRY RUN] Would execute the steps above. Exiting.")
            return EvalDataPrepResult(
                success=True,
                stage="eval",
                tgt_lang_code=cfg.tgt_lang_code,
                output_files={},
            )

        try:
            self._run_pipeline()
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            logger.exception("Eval pipeline failed")
            return EvalDataPrepResult(
                success=False,
                stage="eval",
                tgt_lang_code=cfg.tgt_lang_code,
                output_files=self._output_files,
                error=str(e),
                duration_seconds=duration,
            )

        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Eval pipeline complete in {duration:.1f}s")
        self._print_summary()

        return EvalDataPrepResult(
            success=True,
            stage="eval",
            tgt_lang_code=cfg.tgt_lang_code,
            output_files=self._output_files,
            duration_seconds=duration,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline orchestration
    # ──────────────────────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        cfg = self.config
        enabled = [bm for bm in cfg.benchmarks if bm.enabled]

        # Ensure data/<lang> symlink exists
        from .constants import ensure_data_symlink
        ensure_data_symlink(cfg.tgt_lang_code, cfg.output_dir)

        # ── Fail-fast: validate translator/language support ──
        unsupported: list[tuple[str, str, list[str]]] = []
        for bm in enabled:
            translator_model = self._resolve_translator(bm.name, bm)
            lang = cfg.tgt_lang_code
            if not is_language_supported(translator_model, lang):
                alternatives = get_supported_translators(lang)
                unsupported.append((bm.name, translator_model, alternatives))

        if unsupported:
            lines = [
                "The following benchmarks use translators that do not support "
                f"'{cfg.tgt_lang_name}' ({cfg.tgt_lang_code}):"
            ]
            for bm_name, translator, alts in unsupported:
                alt_text = ", ".join(alts) if alts else "none known"
                lines.append(
                    f"  • {bm_name} → {translator}  (try: {alt_text})"
                )
            lines.append(
                "\nFix: change the translator in your config or remove the "
                "unsupported benchmarks."
            )
            msg = "\n".join(lines)
            logger.error(msg)
            raise LanguageNotSupportedError(
                language=cfg.tgt_lang_code,
                translator=", ".join(t for _, t, _ in unsupported),
            )

        # ── Decide sequential vs. multi-GPU parallel ──
        device_str = cfg.device or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        device_list = [d.strip() for d in device_str.split(",") if d.strip()]
        multi_gpu = len(device_list) > 1

        is_local = False
        if multi_gpu and enabled:
            from byol.translation_backends.registry import MODEL_REGISTRY

            sample_translator = self._resolve_translator(
                enabled[0].name, enabled[0]
            )
            model_cfg = MODEL_REGISTRY.get(sample_translator)
            is_local = model_cfg is not None and model_cfg.model_type == "local"

        if multi_gpu and is_local:
            self._run_benchmarks_parallel(enabled, device_list)
        else:
            for idx, bm in enumerate(enabled, 1):
                logger.info(
                    "─── Benchmark %d/%d: %s ───", idx, len(enabled), bm.name
                )
                self._translate_benchmark(bm)

    # ──────────────────────────────────────────────────────────────────────
    # Multi-GPU parallel execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_benchmarks_parallel(
        self,
        benchmarks: list[EvalBenchmarkConfig],
        device_list: list[str],
    ) -> None:
        """Run benchmarks in parallel across multiple GPUs."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        num_gpus = len(device_list)
        logger.info(
            "Running %d benchmarks in parallel across %d GPUs: %s",
            len(benchmarks),
            num_gpus,
            ", ".join(device_list),
        )

        manager = multiprocessing.Manager()
        device_queue: multiprocessing.Queue[str] = manager.Queue()
        for dev_id in device_list:
            device_queue.put(dev_id)

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for bm in benchmarks:
                future = executor.submit(
                    _translate_benchmark_on_device,
                    config=self.config,
                    benchmark=bm,
                    device_queue=device_queue,
                )
                futures[future] = bm.name

            for future in as_completed(futures):
                bm_name = futures[future]
                try:
                    output_files = future.result()
                    self._output_files.update(output_files)
                    logger.info("[main] %s: done", bm_name)
                except Exception:
                    logger.exception("[main] %s failed", bm_name)

    # ──────────────────────────────────────────────────────────────────────
    # Per-benchmark translation
    # ──────────────────────────────────────────────────────────────────────

    def _translate_benchmark(self, bm: EvalBenchmarkConfig) -> None:
        """Translate all splits of a single benchmark."""
        from .steps.eval_translate_benchmark import (
            load_hf_dataset,
            load_local_jsonl,
            translate_benchmark_split,
        )

        cfg = self.config
        defaults = EVAL_BENCHMARK_DEFAULTS.get(bm.name, {})
        if not defaults:
            logger.warning(
                "No default config for benchmark '%s' — skipping.", bm.name
            )
            return

        translator_model = self._resolve_translator(bm.name, bm)
        suffix = _translator_suffix(translator_model)
        splits = bm.splits or defaults.get("splits", [])
        fields = bm.fields or defaults.get("fields", [])
        dataset_path = defaults.get("dataset_path")
        dataset_config = defaults.get("dataset_config")

        # Determine tgt_lang to pass to the translator
        # LLM translators → language name (used in prompt)
        # API translators → language code (each backend resolves its own format)
        if _is_llm_translator(translator_model):
            tgt_lang = cfg.tgt_lang_name
        else:
            tgt_lang = cfg.tgt_lang_code

        # System prompt for LLM translators
        system_prompt: Optional[str] = None
        if _is_llm_translator(translator_model):
            system_prompt = get_eval_translate_prompt(
                cfg.tgt_lang_name, cfg.tgt_lang_code
            )

        # max_tokens
        max_tokens: Optional[int] = None
        if _is_llm_translator(translator_model):
            max_tokens = bm.max_tokens or cfg.default_llm_max_tokens

        for split in splits:
            # Build output path
            output_template = defaults.get("output_filename_template")
            if output_template:
                out_filename = output_template.format(lang=cfg.tgt_lang_code)
            else:
                out_filename = cfg.get_output_filename(bm.name, split, suffix)
            out_path = str(Path(cfg.output_dir) / out_filename)

            # Skip if a human-translated version exists for this benchmark/split
            human_file = self._find_human_translated(cfg.output_dir, bm.name, split)
            if human_file:
                logger.info(
                    "SKIPPED: Human-translated file exists: %s",
                    human_file,
                )
                self._output_files[f"{bm.name}_{split}"] = human_file
                continue

            # Skip if already exists
            if not cfg.overwrite and os.path.exists(out_path):
                logger.info(
                    "SKIPPED: %s already exists. Use --overwrite to re-translate.",
                    out_path,
                )
                self._output_files[f"{bm.name}_{split}"] = out_path
                continue

            # Load data
            if dataset_path is None:
                # Local file (e.g. COPA)
                template = defaults.get(
                    "local_file_template", f"{bm.name}_{split}.jsonl"
                )
                local_path = str(
                    Path(BENCHMARK_SOURCE_DIR) / template.format(split=split)
                )
                if not os.path.exists(local_path):
                    logger.error(
                        "Local file not found: %s — skipping %s/%s",
                        local_path,
                        bm.name,
                        split,
                    )
                    continue
                rows = load_local_jsonl(local_path)
            else:
                try:
                    hf_loader = defaults.get("hf_loader")
                    rows = load_hf_dataset(dataset_path, dataset_config, split, hf_loader=hf_loader)
                except Exception:
                    logger.exception(
                        "Failed to load HF dataset %s/%s — skipping",
                        dataset_path,
                        split,
                    )
                    continue

            # Apply row transform if configured
            transform_name = defaults.get("row_transform")
            if transform_name:
                from .constants import ROW_TRANSFORMS
                transform_fn = ROW_TRANSFORMS.get(transform_name)
                if transform_fn:
                    rows = transform_fn(rows)

            # Translate
            translate_benchmark_split(
                benchmark_name=bm.name,
                split=split,
                rows=rows,
                fields=fields,
                translator_model=translator_model,
                tgt_lang=tgt_lang,
                output_path=out_path,
                src_lang="en",
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                max_workers=cfg.max_workers,
                batch_size=cfg.batch_size,
                max_samples=cfg.max_samples,
            )
            self._output_files[f"{bm.name}_{split}"] = out_path

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_translator(
        self,
        benchmark_name: str,
        bm: Optional[EvalBenchmarkConfig] = None,
    ) -> str:
        """Resolve the translator model for a benchmark.

        Priority: benchmark config override > benchmark defaults > global config.
        """
        if bm and bm.translator:
            return bm.translator
        defaults = EVAL_BENCHMARK_DEFAULTS.get(benchmark_name, {})
        return defaults.get("translator", self.config.default_translator)

    @staticmethod
    def _find_human_translated(output_dir: str, benchmark_name: str, split: str) -> str:
        """Check if a human-translated file exists for this benchmark/split.

        Returns the path if found, empty string otherwise.
        Human-translated files contain 'human_translated' in the filename.
        """
        out = Path(output_dir)
        if not out.exists():
            return ""
        # Map benchmark names to filename prefixes
        prefix_map = {"mmlu_lite": "mmlu_lite"}
        prefix = prefix_map.get(benchmark_name, benchmark_name)
        pattern = f"{prefix}_{split}_*human_translated*"
        matches = list(out.glob(pattern))
        if not matches:
            # Also check without split in name
            pattern2 = f"{prefix}*{split}*human_translated*"
            matches = list(out.glob(pattern2))
        return str(matches[0]) if matches else ""

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        cfg = self.config
        logger.info("=" * 60)
        logger.info("EVAL DATA PREP — SUMMARY")
        logger.info(f"  Language: {cfg.tgt_lang_name} ({cfg.tgt_lang_code})")
        logger.info(f"  Output dir: {cfg.output_dir}")
        logger.info("  Output files:")
        for key, path in sorted(self._output_files.items()):
            exists = os.path.exists(path) if path else False
            logger.info(f"    {key:40s} {'✓' if exists else '✗'}  {path}")
        logger.info("")
        logger.info(
            "  Next step → configure eval task YAMLs to point to these files"
        )
        logger.info("  Then run: python -m byol.eval --config configs/eval/...")
        logger.info("=" * 60)
