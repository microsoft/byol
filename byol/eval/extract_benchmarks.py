#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Extract benchmark results from lm-evaluation-harness log files.

Parses lm-eval log files and extracts benchmark metrics for both base (CPT)
and instruct model evaluations.  Supports multiple output formats and handles
language-specific task name variations.

It can accept:
    - A single log file
    - Multiple log files
    - A directory (auto-discovers all ``lm_eval.log`` files recursively)

All parsed metrics are merged, so you get one combined table across tasks.

Supported benchmarks:
    - General & STEM: Global MMLU-Lite, ARC, MGSM, BBH, GPQA
    - Commonsense: XCOPA/COPA, XStoryCloze/StoryCloze, PIQA, HellaSwag
    - NLI & Reading: XNLI, XWinograd/Winogrande, Belebele
    - Translation: FLORES (BLEU and chrF metrics)
    - Instruct-only: IFEval, TruthfulQA, HumanEval

Usage:
    python -m byol.eval.extract_benchmarks results/ --type instruct --tgt-lang eng
    python -m byol.eval.extract_benchmarks results/ --type base --tgt-lang nya --csv
    python -m byol.eval.extract_benchmarks results/ --debug
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "0.1.0"


# =============================================================================
# Language Configuration — extend here to add new languages
# =============================================================================

SUPPORTED_LANGUAGES: dict[str, dict] = {
    "eng": {
        "code": "eng",
        "name": "English",
        "aliases": ["eng", "en", "english"],
    },
    "mri": {
        "code": "mri",
        "name": "Māori",
        "aliases": ["mri", "maori"],
    },
    "nya": {
        "code": "nya",
        "name": "Chichewa",
        "aliases": ["nya", "ny", "nyanja", "chichewa"],
    },
    "gug": {
        "code": "gug",
        "name": "Guaraní",
        "aliases": ["gug", "gn", "guarani"],
    },
}

VALID_LANGS: frozenset[str] = frozenset(SUPPORTED_LANGUAGES.keys())
VALID_TYPES: frozenset[str] = frozenset({"base", "instruct"})
VALID_TYPES_WITH_AUTO: frozenset[str] = frozenset({"base", "instruct", "auto"})

# =============================================================================
# Benchmark Categories and Metrics
# =============================================================================

CATEGORY_GENERAL = "General & STEM"
CATEGORY_COMMONSENSE = "Commonsense"
CATEGORY_NLI_READING = "NLI & Reading"
CATEGORY_READING_QA = "Reading & QA"
CATEGORY_TRANSLATION = "Translation"
CATEGORY_INSTRUCTION = "Instruction Following"
CATEGORY_TRUTHFULNESS = "Truthfulness"
CATEGORY_CODE = "Code"

METRIC_ACC = "acc"
METRIC_ACC_NORM = "acc_norm"
METRIC_EXACT_MATCH = "exact_match"
METRIC_BLEU = "bleu"
METRIC_CHRF = "chrf"
METRIC_BLEU_ACC = "bleu_acc"
METRIC_PASS_AT_1 = "pass@1"
METRIC_PROMPT_LOOSE = "prompt_level_loose_acc"
METRIC_PROMPT_STRICT = "prompt_level_strict_acc"


# =============================================================================
# Benchmark Definitions (data-driven)
# =============================================================================


@dataclass(frozen=True)
class BenchmarkDef:
    """Definition of a single benchmark and how to look up its scores.

    Attributes:
        name: Human-readable benchmark name.
        category: Benchmark category for grouping in output.
        metric: Primary metric to extract.
        fallback_metrics: Alternative metrics to try (priority order).
        tgt_task: Task name template for the target language (``{lang}`` placeholder).
        eng_task: Task name for the English reference score.
        eng_fallback_tasks: Fallback English task names to try.
        eng_only: If ``True``, benchmark only applies when ``lang == eng``.
        non_eng_only: If ``True``, benchmark only applies when ``lang != eng``.
    """

    name: str
    category: str
    metric: str
    fallback_metrics: tuple[str, ...] = ()
    tgt_task: str | None = None
    eng_task: str | None = None
    eng_fallback_tasks: tuple[str, ...] = ()
    eng_only: bool = False
    non_eng_only: bool = False


# --- Base model benchmarks ---------------------------------------------------

BASE_BENCHMARKS: list[BenchmarkDef] = [
    # General & STEM
    BenchmarkDef("Global MMLU-Lite", CATEGORY_GENERAL, METRIC_ACC,
                 tgt_task="global_mmlu_{lang}", eng_task="global_mmlu_en"),
    BenchmarkDef("ARC-Easy", CATEGORY_GENERAL, METRIC_ACC_NORM,
                 fallback_metrics=(METRIC_ACC,),
                 tgt_task="arc_easy_{lang}", eng_task="arc_easy"),
    BenchmarkDef("ARC-Challenge", CATEGORY_GENERAL, METRIC_ACC_NORM,
                 fallback_metrics=(METRIC_ACC,),
                 tgt_task="arc_challenge_{lang}", eng_task="arc_challenge"),
    BenchmarkDef("MGSM", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                 tgt_task="mgsm_direct_{lang}", eng_task="mgsm_direct_en"),
    BenchmarkDef("BBH", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                 eng_task="bbh_fewshot"),
    BenchmarkDef("GPQA Diamond", CATEGORY_GENERAL, METRIC_ACC_NORM,
                 fallback_metrics=(METRIC_ACC,),
                 eng_task="gpqa_diamond_n_shot"),
    # Commonsense
    BenchmarkDef("XCOPA", CATEGORY_COMMONSENSE, METRIC_ACC,
                 tgt_task="xcopa_{lang}", eng_task="copa", non_eng_only=True),
    BenchmarkDef("COPA", CATEGORY_COMMONSENSE, METRIC_ACC,
                 eng_task="copa", eng_only=True),
    BenchmarkDef("XStoryCloze", CATEGORY_COMMONSENSE, METRIC_ACC,
                 tgt_task="xstorycloze_{lang}",
                 eng_task="xstorycloze_en", eng_fallback_tasks=("storycloze",),
                 non_eng_only=True),
    BenchmarkDef("StoryCloze", CATEGORY_COMMONSENSE, METRIC_ACC,
                 eng_task="xstorycloze_en", eng_fallback_tasks=("storycloze",),
                 eng_only=True),
    BenchmarkDef("PIQA", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                 fallback_metrics=(METRIC_ACC,),
                 tgt_task="piqa_{lang}", eng_task="piqa"),
    BenchmarkDef("HellaSwag", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                 fallback_metrics=(METRIC_ACC,),
                 tgt_task="hellaswag_{lang}", eng_task="hellaswag"),
    # NLI & Reading
    BenchmarkDef("XNLI 2.0", CATEGORY_NLI_READING, METRIC_ACC,
                 tgt_task="xnli_{lang}",
                 eng_task="xnli_en", eng_fallback_tasks=("xnli",)),
    BenchmarkDef("XWinograd", CATEGORY_NLI_READING, METRIC_ACC,
                 tgt_task="xwinograd_{lang}", eng_task="winogrande",
                 non_eng_only=True),
    BenchmarkDef("Winogrande", CATEGORY_NLI_READING, METRIC_ACC,
                 eng_task="winogrande", eng_only=True),
    BenchmarkDef("Belebele", CATEGORY_NLI_READING, METRIC_ACC_NORM,
                 tgt_task="belebele_{lang}_Latn", eng_task="belebele_eng_Latn"),
    # Translation (non-English only)
    BenchmarkDef("FLORES (->EN)", CATEGORY_TRANSLATION, METRIC_BLEU,
                 tgt_task="flores_{lang}_en", non_eng_only=True),
    BenchmarkDef("FLORES (->EN) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                 tgt_task="flores_{lang}_en", non_eng_only=True),
    BenchmarkDef("FLORES (EN->)", CATEGORY_TRANSLATION, METRIC_BLEU,
                 tgt_task="flores_en_{lang}", non_eng_only=True),
    BenchmarkDef("FLORES (EN->) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                 tgt_task="flores_en_{lang}", non_eng_only=True),
]

# --- Instruct model benchmarks -----------------------------------------------

INSTRUCT_BENCHMARKS: list[BenchmarkDef] = [
    # General & STEM
    BenchmarkDef("Global MMLU-Lite", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                 tgt_task="global_mmlu_{lang}_gen_0shot",
                 eng_task="global_mmlu_en_gen_0shot"),
    BenchmarkDef("ARC Challenge (chat)", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                 tgt_task="arc_challenge_chat_{lang}",
                 eng_task="arc_challenge_chat"),
    BenchmarkDef("MGSM", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                 tgt_task="mgsm_direct_{lang}", eng_task="mgsm_direct_en"),
    BenchmarkDef("BBH", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                 eng_task="bbh_zeroshot", eng_fallback_tasks=("bbh",)),
    BenchmarkDef("GPQA Diamond", CATEGORY_GENERAL, METRIC_ACC_NORM,
                 eng_task="gpqa_diamond_zeroshot",
                 eng_fallback_tasks=("gpqa_diamond",)),
    # Commonsense
    BenchmarkDef("XCOPA", CATEGORY_COMMONSENSE, METRIC_ACC,
                 tgt_task="xcopa_{lang}", eng_task="copa", non_eng_only=True),
    BenchmarkDef("COPA", CATEGORY_COMMONSENSE, METRIC_ACC,
                 eng_task="copa", eng_only=True),
    BenchmarkDef("XStoryCloze", CATEGORY_COMMONSENSE, METRIC_ACC,
                 tgt_task="xstorycloze_{lang}", eng_task="xstorycloze_en",
                 non_eng_only=True),
    BenchmarkDef("StoryCloze", CATEGORY_COMMONSENSE, METRIC_ACC,
                 eng_task="xstorycloze_en", eng_only=True),
    BenchmarkDef("PIQA", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                 tgt_task="piqa_{lang}", eng_task="piqa"),
    BenchmarkDef("HellaSwag", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                 tgt_task="hellaswag_{lang}", eng_task="hellaswag"),
    # Reading & QA
    BenchmarkDef("XNLI 2.0", CATEGORY_READING_QA, METRIC_ACC,
                 tgt_task="xnli_{lang}", eng_task="xnli_en"),
    BenchmarkDef("XWinograd", CATEGORY_READING_QA, METRIC_ACC,
                 tgt_task="xwinograd_{lang}", eng_task="winogrande",
                 non_eng_only=True),
    BenchmarkDef("Winogrande", CATEGORY_READING_QA, METRIC_ACC,
                 eng_task="winogrande", eng_only=True),
    BenchmarkDef("Belebele", CATEGORY_READING_QA, METRIC_ACC_NORM,
                 tgt_task="belebele_{lang}_Latn", eng_task="belebele_eng_Latn"),
    # Instruction Following
    BenchmarkDef("IFEval", CATEGORY_INSTRUCTION, METRIC_PROMPT_LOOSE,
                 fallback_metrics=(METRIC_PROMPT_STRICT,),
                 eng_task="ifeval"),
    # Truthfulness
    BenchmarkDef("TruthfulQA", CATEGORY_TRUTHFULNESS, METRIC_BLEU_ACC,
                 tgt_task="truthfulqa-multi_gen_{lang}",
                 eng_task="truthfulqa-multi_gen_en"),
    # Code
    BenchmarkDef("HumanEval", CATEGORY_CODE, METRIC_PASS_AT_1,
                 eng_task="humaneval_instruct",
                 eng_fallback_tasks=("humaneval",)),
    # Translation (non-English only)
    BenchmarkDef("FLORES (->EN)", CATEGORY_TRANSLATION, METRIC_BLEU,
                 tgt_task="flores_{lang}_en", non_eng_only=True),
    BenchmarkDef("FLORES (->EN) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                 tgt_task="flores_{lang}_en", non_eng_only=True),
    BenchmarkDef("FLORES (EN->)", CATEGORY_TRANSLATION, METRIC_BLEU,
                 tgt_task="flores_en_{lang}", non_eng_only=True),
    BenchmarkDef("FLORES (EN->) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                 tgt_task="flores_en_{lang}", non_eng_only=True),
]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BenchmarkResult:
    """Single benchmark result with target and reference scores."""

    name: str
    category: str
    metric: str
    target: float | None = None
    english: float | None = None


@dataclass
class MultiLangBenchmarkResult:
    """Benchmark result with scores for multiple languages."""

    name: str
    category: str
    metric: str
    scores: dict[str, float | None] = field(default_factory=dict)


@dataclass
class ParsedMetrics:
    """Container for all parsed metrics from log files."""

    data: dict[str, dict[str, float]] = field(default_factory=dict)

    def get(self, task: str, metric: str) -> float | None:
        """Get metric value for a task."""
        return self.data.get(task, {}).get(metric)

    def get_with_lang_variants(
        self, template: str, lang: str, metric: str,
    ) -> float | None:
        """Get metric trying all language code variants."""
        lang_config = SUPPORTED_LANGUAGES.get(lang, {})
        variants = lang_config.get("aliases", [lang])
        for variant in variants:
            value = self.get(template.format(lang=variant), metric)
            if value is not None:
                return value
        return None

    def get_first_available(self, task: str, metrics: list[str]) -> float | None:
        """Get first available metric from a priority list."""
        for metric in metrics:
            value = self.get(task, metric)
            if value is not None:
                return value
        return None

    def merge(self, other: ParsedMetrics) -> None:
        """Merge another ``ParsedMetrics`` into this one, keeping max values."""
        for task, task_metrics in other.data.items():
            if task not in self.data:
                self.data[task] = {}
            for metric, value in task_metrics.items():
                current = self.data[task].get(metric, 0)
                self.data[task][metric] = max(value, current)


# =============================================================================
# Log Parser
# =============================================================================


class LogParser:
    """Parse lm-evaluation-harness log files to extract benchmark metrics.

    The parser handles the table format output from lm-eval::

        |task_name|version|filter|n-shot|metric|dir|value|+/-|stderr|

    Accepts a single file, multiple files, or directories (auto-discovers
    ``lm_eval.log`` files recursively).
    """

    _TABLE_PATTERN = re.compile(
        r"^\|([\w_-]*)\s*\|"
        r"[^|]*\|"
        r"[^|]*\|"
        r"[^|]*\|"
        r"([\w_@]+)\s*\|"
        r"[^|]*\|"
        r"\s*([\d.]+)\s*\|"
    )

    def __init__(self, paths: list[Path]) -> None:
        self.metrics = ParsedMetrics()
        self.files_parsed: list[Path] = []
        log_files = self._resolve_paths(paths)
        if not log_files:
            print("Warning: No lm_eval.log files found.", file=sys.stderr)
        for log_file in log_files:
            single = self._parse_file(log_file)
            self.metrics.merge(single)
            self.files_parsed.append(log_file)

    @staticmethod
    def _resolve_paths(paths: list[Path]) -> list[Path]:
        """Resolve paths to a flat list of log files."""
        log_files: list[Path] = []
        for p in paths:
            if p.is_file():
                log_files.append(p)
            elif p.is_dir():
                log_files.extend(sorted(p.rglob("lm_eval.log")))
            else:
                print(f"Warning: Skipping {p} (not a file or directory).", file=sys.stderr)
        return log_files

    def _parse_file(self, log_file: Path) -> ParsedMetrics:
        """Parse a single log file and return its metrics."""
        metrics = ParsedMetrics()
        content = log_file.read_text(encoding="utf-8")
        current_task: str | None = None
        for line in content.split("\n"):
            if line.strip().startswith("| -") or not line.strip():
                continue
            match = self._TABLE_PATTERN.match(line)
            if not match:
                continue
            task = match.group(1).strip() or current_task
            metric = match.group(2).strip()
            value = float(match.group(3))
            current_task = task
            if task:
                if task not in metrics.data:
                    metrics.data[task] = {}
                current = metrics.data[task].get(metric, 0)
                metrics.data[task][metric] = max(value, current)
        return metrics


# =============================================================================
# Language Detection
# =============================================================================


def _get_known_langs() -> set[str]:
    """Get all known language codes and aliases."""
    known: set[str] = set()
    for lang_config in SUPPORTED_LANGUAGES.values():
        known.update(lang_config.get("aliases", []))
        known.add(lang_config["code"])
    return known


def _normalize_lang_code(name: str) -> str | None:
    """Normalise a language name/alias to its canonical code."""
    name = name.lower()
    for canonical, lang_config in SUPPORTED_LANGUAGES.items():
        if name in lang_config.get("aliases", []):
            return canonical
    if name in SUPPORTED_LANGUAGES:
        return name
    return None


# =============================================================================
# Type Detection (score-based, matches original behavior)
# =============================================================================


def detect_eval_type(metrics: ParsedMetrics, languages: list[str]) -> tuple[str, str]:
    """Pick base vs instruct by comparing how many scores resolve.

    Tries both benchmark sets and counts non-None targets across languages.
    Chooses the type with more hits (ties default to base).
    Returns (type, reason).
    """

    best_type = "base"
    best_count = -1
    best_reason = ""

    for eval_type in ("base", "instruct"):
        count = 0
        for lang in languages:
            extractor = BenchmarkExtractor(metrics, lang)
            results = extractor.extract(eval_type)
            count += sum(1 for r in results if r.target is not None)
        reason = f"resolved {count} benchmark scores"
        if count > best_count:
            best_count = count
            best_type = eval_type
            best_reason = reason

    return best_type, best_reason


def detect_single_language(paths: list[Path]) -> str | None:
    """Check whether the path itself is a language folder."""
    if len(paths) != 1:
        return None
    p = paths[0]
    if not p.is_dir():
        return None
    return _normalize_lang_code(p.name)


def detect_languages(paths: list[Path]) -> list[str]:
    """Detect language folders from result directories.

    Returns:
        Sorted list of canonical language codes (``eng`` first).
    """
    known_langs = _get_known_langs()
    detected: set[str] = set()
    for p in paths:
        if p.is_dir():
            for subdir in p.iterdir():
                if subdir.is_dir():
                    name = subdir.name.lower()
                    if name in known_langs:
                        canonical = _normalize_lang_code(name)
                        if canonical:
                            detected.add(canonical)
    return sorted(detected, key=lambda x: (x != "eng", x))


# =============================================================================
# Benchmark Extractor
# =============================================================================


class BenchmarkExtractor:
    """Extract benchmark results from parsed metrics using benchmark definitions."""

    def __init__(self, metrics: ParsedMetrics, lang: str) -> None:
        self._m = metrics
        self._lang = lang
        self._is_eng = lang == "eng"

    def extract(self, mode: str) -> list[BenchmarkResult]:
        """Extract benchmarks for the given mode (``base`` or ``instruct``).

        Args:
            mode: ``"base"`` or ``"instruct"``.

        Returns:
            List of ``BenchmarkResult`` with scores filled in.
        """
        defs = BASE_BENCHMARKS if mode == "base" else INSTRUCT_BENCHMARKS
        results: list[BenchmarkResult] = []
        for bdef in defs:
            if bdef.eng_only and not self._is_eng:
                continue
            if bdef.non_eng_only and self._is_eng:
                continue
            target = self._resolve_target(bdef)
            english = self._resolve_english(bdef)
            results.append(BenchmarkResult(
                name=bdef.name, category=bdef.category, metric=bdef.metric,
                target=target, english=english,
            ))
        return results

    def _resolve_target(self, bdef: BenchmarkDef) -> float | None:
        all_metrics = [bdef.metric, *bdef.fallback_metrics]
        if self._is_eng:
            return self._lookup_eng_task(bdef, all_metrics)
        if bdef.tgt_task:
            for metric in all_metrics:
                value = self._m.get_with_lang_variants(bdef.tgt_task, self._lang, metric)
                if value is not None:
                    return value
        return None

    def _resolve_english(self, bdef: BenchmarkDef) -> float | None:
        if self._is_eng:
            return None
        all_metrics = [bdef.metric, *bdef.fallback_metrics]
        return self._lookup_eng_task(bdef, all_metrics)

    def _lookup_eng_task(
        self, bdef: BenchmarkDef, metrics: list[str],
    ) -> float | None:
        if bdef.eng_task is None:
            return None
        tasks = [bdef.eng_task, *bdef.eng_fallback_tasks]
        for task in tasks:
            for metric in metrics:
                value = self._m.get(task, metric)
                if value is not None:
                    return value
        return None


class MultiLangBenchmarkExtractor:
    """Extract benchmark results for multiple languages at once."""

    def __init__(self, metrics: ParsedMetrics, languages: list[str]) -> None:
        self._m = metrics
        self._languages = languages

    def extract(self, mode: str) -> list[MultiLangBenchmarkResult]:
        """Extract benchmarks for all languages."""
        defs = BASE_BENCHMARKS if mode == "base" else INSTRUCT_BENCHMARKS
        results: list[MultiLangBenchmarkResult] = []
        seen_benchmarks: set[str] = set()
        for bdef in defs:
            if bdef.name in seen_benchmarks:
                continue
            seen_benchmarks.add(bdef.name)
            scores: dict[str, float | None] = {}
            for lang in self._languages:
                is_eng = lang == "eng"
                if bdef.eng_only and not is_eng:
                    scores[lang] = None
                    continue
                if bdef.non_eng_only and is_eng:
                    scores[lang] = None
                    continue
                extractor = BenchmarkExtractor(self._m, lang)
                scores[lang] = extractor._resolve_target(bdef)
            results.append(MultiLangBenchmarkResult(
                name=bdef.name, category=bdef.category, metric=bdef.metric,
                scores=scores,
            ))
        return results


# =============================================================================
# Output Formatters
# =============================================================================


class OutputFormatter:
    """Format benchmark results for display or export."""

    @staticmethod
    def format_value(value: float | None, width: int = 10) -> str:
        if value is not None:
            return f"{value:>{width}.4f}"
        return f"{'-':>{width}}"

    @classmethod
    def print_single_lang_table(cls, results: list[BenchmarkResult], lang: str) -> None:
        print(f"{'Benchmark':<24} {lang.upper():>10}")
        print("-" * 36)
        current_category: str | None = None
        for result in results:
            if result.target is None:
                continue
            if result.category != current_category:
                if current_category is not None:
                    print()
                print(f"# {result.category}")
                current_category = result.category
            print(f"{result.name:<24} {cls.format_value(result.target)}")

    @staticmethod
    def print_single_lang_csv(results: list[BenchmarkResult], lang: str) -> None:
        print(f"category,benchmark,metric,{lang}")
        for r in results:
            if r.target is None:
                continue
            print(f"{r.category},{r.name},{r.metric},{r.target:.4f}")

    @classmethod
    def print_multi_lang_table(
        cls, results: list[MultiLangBenchmarkResult], languages: list[str],
    ) -> None:
        col_width = 10
        name_width = 24
        header = f"{'Benchmark':<{name_width}}"
        for lang in languages:
            header += f" {lang.upper():>{col_width}}"
        print(header)
        print("-" * (name_width + (col_width + 1) * len(languages)))
        current_category: str | None = None
        for result in results:
            if not any(s is not None for s in result.scores.values()):
                continue
            if result.category != current_category:
                if current_category is not None:
                    print()
                print(f"# {result.category}")
                current_category = result.category
            row = f"{result.name:<{name_width}}"
            for lang in languages:
                row += f" {cls.format_value(result.scores.get(lang), col_width)}"
            print(row)

    @classmethod
    def print_multi_lang_csv(
        cls, results: list[MultiLangBenchmarkResult], languages: list[str],
    ) -> None:
        print("category,benchmark,metric," + ",".join(languages))
        for r in results:
            if not any(s is not None for s in r.scores.values()):
                continue
            scores_str = ",".join(
                f"{r.scores.get(lang):.4f}" if r.scores.get(lang) is not None else ""
                for lang in languages
            )
            print(f"{r.category},{r.name},{r.metric},{scores_str}")

    @staticmethod
    def print_debug(metrics: ParsedMetrics) -> None:
        print("=" * 60)
        print("PARSED TASKS")
        print("=" * 60)
        for task in sorted(metrics.data.keys()):
            task_metrics = metrics.data[task]
            print(f"\n{task}:")
            for metric, value in sorted(task_metrics.items()):
                print(f"  {metric}: {value:.4f}")


# =============================================================================
# CLI
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for extract_benchmarks."""
    parser = argparse.ArgumentParser(
        description="Extract benchmark results from lm-evaluation-harness log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s results/ --type auto --tgt-lang eng
    %(prog)s results/run1/lm_eval.log results/run2/lm_eval.log --type base --tgt-lang nya
    %(prog)s results/ --type instruct --tgt-lang mri --csv
  %(prog)s results/ --debug
        """,
    )
    parser.add_argument(
        "log_paths", type=Path, nargs="+",
        help="Log file(s) or directory(ies) containing lm_eval.log files",
    )
    parser.add_argument(
        "--type", type=str, choices=sorted(VALID_TYPES_WITH_AUTO), default="auto",
        help="Evaluation type: 'base', 'instruct', or 'auto' to infer from tasks (default: auto)",
    )
    parser.add_argument(
        "--tgt-lang", type=str, default=None,
        help="Target language code (e.g. eng, mri, nya, gug). Auto-detects from result folders if omitted.",
    )
    parser.add_argument("--csv", action="store_true", help="Output results in CSV format")
    parser.add_argument("--debug", action="store_true", help="Show all parsed tasks")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args()

    for p in args.log_paths:
        if not p.exists():
            print(f"Error: Path not found: {p}", file=sys.stderr)
            return 1

    try:
        log_parser = LogParser(args.log_paths)
    except Exception as e:
        print(f"Error parsing log files: {e}", file=sys.stderr)
        return 1

    # ── Determine active languages ───────────────────────────────────────
    single_lang = args.tgt_lang
    if single_lang is None:
        single_lang = detect_single_language(args.log_paths)
    active_langs = [single_lang] if single_lang is not None else detect_languages(args.log_paths)

    eval_type = args.type
    detect_reason = ""
    if eval_type == "auto":
        eval_type, detect_reason = detect_eval_type(log_parser.metrics, active_langs)
        print(f"Auto-detected eval type: {eval_type} ({detect_reason})", file=sys.stderr)

    all_lang_codes = _get_known_langs()

    # ── Print relevant parsed files ──────────────────────────────────────
    if log_parser.files_parsed:
        relevant_files: list[Path] = []
        for f in log_parser.files_parsed:
            path_parts = [p.lower() for p in f.parts]
            file_lang: str | None = None
            for part in path_parts:
                if part in all_lang_codes:
                    file_lang = _normalize_lang_code(part)
                    break
            if file_lang and file_lang in active_langs:
                relevant_files.append(f)
        if relevant_files:
            print(f"Parsed {len(relevant_files)} log file(s):", file=sys.stderr)
            for f in relevant_files:
                print(f"  {f}", file=sys.stderr)
            print(file=sys.stderr)

    if args.debug:
        OutputFormatter.print_debug(log_parser.metrics)
        print()

    # ── Output ───────────────────────────────────────────────────────────
    if single_lang is not None:
        extractor = BenchmarkExtractor(log_parser.metrics, single_lang)
        results = extractor.extract(eval_type)
        if args.csv:
            OutputFormatter.print_single_lang_csv(results, single_lang)
        else:
            OutputFormatter.print_single_lang_table(results, single_lang)
    else:
        detected_langs = detect_languages(args.log_paths)
        if not detected_langs:
            print(
                "Error: No language folders detected. Use --tgt-lang to specify.",
                file=sys.stderr,
            )
            return 1
        print(
            f"Detected languages: {', '.join(lang.upper() for lang in detected_langs)}",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        ml_extractor = MultiLangBenchmarkExtractor(log_parser.metrics, detected_langs)
        ml_results = ml_extractor.extract(eval_type)
        if args.csv:
            OutputFormatter.print_multi_lang_csv(ml_results, detected_langs)
        else:
            OutputFormatter.print_multi_lang_table(ml_results, detected_langs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
