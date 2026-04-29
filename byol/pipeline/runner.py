# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pipeline runner — detects completion status and executes pipeline steps.

Each step knows:
- How to check if it's already done (output files exist)
- The command to run it
- What it depends on
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("byol-pipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# Step Definition
# =============================================================================

STATUS_DONE = "done"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

STATUS_ICONS = {
    STATUS_DONE: "✅",
    STATUS_PENDING: "⬚",
    STATUS_RUNNING: "🔄",
    STATUS_FAILED: "❌",
    STATUS_SKIPPED: "⏭️",
}


@dataclass
class StepResult:
    success: bool
    message: str = ""


@dataclass
class PipelineStep:
    """A single step in the BYOL pipeline."""

    id: str
    name: str
    description: str
    depends_on: list[str] = field(default_factory=list)

    # Callables set by PipelineRunner
    _check_done: object = None  # () -> bool
    _build_cmd: object = None  # () -> list[str]
    _is_optional: bool = False

    def is_done(self) -> bool:
        if self._check_done:
            return self._check_done()
        return False

    def build_command(self) -> list[str]:
        if self._build_cmd:
            return self._build_cmd()
        return []


# =============================================================================
# Pipeline Runner
# =============================================================================


class PipelineRunner:
    """Orchestrates the BYOL pipeline for a target language.

    Usage::

        runner = PipelineRunner("gug", device="3")
        runner.status()          # print what's done / pending
        runner.run_next()        # run the next pending step
        runner.run_all()         # run all pending steps sequentially
    """

    def __init__(
        self,
        tgt_lang: str,
        *,
        device: str = "0",
        max_samples: int | None = None,
        model: str = "google/gemma-3-4b-pt",
        instruct_model: str | None = None,
        quick_test: bool = False,
        translators: str = "",
        llms: str = "",
    ) -> None:
        self.tgt_lang = tgt_lang
        self.device = device
        self.max_samples = max_samples
        self.model = model
        self.instruct_model = instruct_model  # None if not provided; merge step will error
        self.quick_test = quick_test
        self.translators = translators
        self.llms = llms
        self.python = sys.executable

        # Resolve language display name
        self._lang_name = self._resolve_name()

        # Build steps
        self.steps = self._build_steps()

    # ── Language Resolution ──────────────────────────────────────────────

    def _resolve_name(self) -> str:
        """Get display name for the language."""
        try:
            from byol.data_prep.constants import LANG_NAMES

            return LANG_NAMES.get(self.tgt_lang, self.tgt_lang)
        except ImportError:
            return self.tgt_lang

    def _is_known_language(self) -> bool:
        """Check if the language is recognized by the classification system."""
        try:
            from byol.language_resource_assessment.language_digital_presence import (
                LanguageClassifier,
                LanguageInfo,
            )

            classifier = LanguageClassifier()
            result = classifier.lookup(self.tgt_lang)
            return isinstance(result, LanguageInfo)
        except (ValueError, ImportError, Exception):
            return False

    # ── Path Helpers ─────────────────────────────────────────────────────

    @property
    def _byol_data(self) -> Path:
        return Path(os.environ.get("BYOL_DATA_DIR", Path.home() / "byol-data"))

    @property
    def _lang_data(self) -> Path:
        """Per-language data root: ~/byol-data/<lang>"""
        return self._byol_data / self.tgt_lang

    @property
    def _cpt_jsonl(self) -> Path:
        return self._lang_data / "cpt" / "bilingual_mix" / f"{self.tgt_lang}_english_cpt.jsonl"

    @property
    def _sft_jsonl(self) -> Path:
        return self._lang_data / "sft" / "bilingual_mix" / f"{self.tgt_lang}_english_sft.jsonl"

    @property
    def _sft_test_jsonl(self) -> Path:
        return self._lang_data / "sft" / "bilingual_mix" / f"{self.tgt_lang}_sft_test.jsonl"

    @property
    def _eval_data_dir(self) -> Path:
        return self._lang_data / "eval"

    @property
    def _results_dir(self) -> Path:
        """Per-language results root: results/<lang>"""
        return REPO_ROOT / "results" / self.tgt_lang

    @property
    def _lra_results(self) -> Path:
        return self._results_dir / "lra"

    @property
    def _train_cpt_dir(self) -> Path:
        return self._results_dir / "train" / "cpt"

    @property
    def _train_sft_dir(self) -> Path:
        return self._results_dir / "train" / "sft"

    @property
    def _train_merged_dir(self) -> Path:
        return self._results_dir / "train" / "merged"

    def _find_latest_checkpoint(self, base_dir: Path, stage: str) -> str | None:
        """Find the latest checkpoint directory for this language."""
        if not base_dir.exists():
            return None
        # Look for dirs matching the model pattern
        candidates = sorted(base_dir.iterdir(), key=lambda p: p.name, reverse=True)
        for d in candidates:
            if d.is_dir() and stage in d.name:
                # Check it has model files (full or LoRA adapter)
                if (
                    (d / "config.json").exists()
                    or (d / "model.safetensors.index.json").exists()
                    or (d / "adapter_config.json").exists()
                ):
                    return str(d)
        return None

    def _ensure_data_symlink(self) -> None:
        """Create data/<lang>/{cpt,sft} symlinks to ~/byol-data/<lang>/{cpt,sft}.

        The eval/ subdirectory is kept as real files in the repo (shipped for
        paper languages). Only cpt/ and sft/ are symlinked to the external
        data directory where data_prep generates training data.
        """
        lang_dir = REPO_ROOT / "data" / self.tgt_lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ("cpt", "sft"):
            link = lang_dir / subdir
            target = self._lang_data / subdir
            if link.exists() or link.is_symlink():
                continue
            target.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
            logger.info(f"Created symlink: {link} -> {target}")

        # Also ensure eval/ dir exists (as real dir for new languages,
        # or already present for shipped languages like nya/mri)
        eval_dir = lang_dir / "eval"
        if not eval_dir.exists():
            ext_eval = self._lang_data / "eval"
            if ext_eval.exists():
                eval_dir.symlink_to(ext_eval)
                logger.info(f"Created symlink: {eval_dir} -> {ext_eval}")
            else:
                eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Step Definitions ─────────────────────────────────────────────────

    def _build_steps(self) -> list[PipelineStep]:
        lang = self.tgt_lang
        dev = self.device
        samples_args = [f"--max-samples {self.max_samples}"] if self.max_samples else []
        samples_flag = f" --max-samples {self.max_samples}" if self.max_samples else ""

        steps = []

        # 1. Language Classification
        s = PipelineStep(
            id="classify",
            name="Language Classification",
            description=f"Classify {self._lang_name} ({lang}) by digital resource level",
        )
        s._check_done = lambda: self._is_known_language()
        s._build_cmd = lambda: [
            self.python, "-m", "byol.language_resource_assessment",
            "--task", "language-classification", "--tgt-lang", lang,
        ]
        steps.append(s)

        # 2. Find Best Translator
        translators_arg = f" --translators {self.translators}" if self.translators else ""
        s = PipelineStep(
            id="find-translator",
            name="Find Best Translator",
            description="Benchmark translation models on RTTBench-Mono",
            depends_on=["classify"],
        )
        s._check_done = lambda: (
            self._lra_results / "translator_comparison"
        ).exists()
        s._build_cmd = lambda: [
            self.python, "-m", "byol.language_resource_assessment",
            "--task", "find-best-translator", "--tgt-lang", lang,
            "--output-dir", str(self._lra_results / "translator_comparison"),
            *(["--translators", self.translators] if self.translators else []),
            *(["--max-samples", str(self.max_samples)] if self.max_samples else []),
        ]
        steps.append(s)

        # 3. Find Best LLM (optional)
        s = PipelineStep(
            id="find-llm",
            name="Find Best LLM",
            description="Benchmark open-weight LLMs for language adaptation",
            depends_on=["classify"],
        )
        s._is_optional = True
        s._check_done = lambda: (
            self._lra_results / "llm_comparison"
        ).exists()
        s._build_cmd = lambda: [
            self.python, "-m", "byol.language_resource_assessment",
            "--task", "find-best-llm", "--tgt-lang", lang,
            "--output-dir", str(self._lra_results / "llm_comparison"),
            *(["--llms", self.llms] if self.llms else []),
            *(["--max-samples", str(self.max_samples)] if self.max_samples else []),
            "--device", dev,
        ]
        steps.append(s)

        # 4. Data Prep — CPT
        s = PipelineStep(
            id="data-prep-cpt",
            name="Prepare CPT Data",
            description="Download, refine, translate → bilingual CPT training data",
            depends_on=["classify"],
        )
        s._check_done = lambda: self._cpt_jsonl.exists()
        s._build_cmd = lambda: [
            self.python, "-m", "byol.data_prep",
            "--stage", "cpt", "--tgt-lang", lang,
            *(["--max-samples", str(self.max_samples)] if self.max_samples else []),
        ]
        steps.append(s)

        # 5. Data Prep — SFT
        s = PipelineStep(
            id="data-prep-sft",
            name="Prepare SFT Data",
            description="Download, translate → bilingual SFT instruction data",
            depends_on=["classify"],
        )
        s._check_done = lambda: self._sft_jsonl.exists()
        s._build_cmd = lambda: [
            self.python, "-m", "byol.data_prep",
            "--stage", "sft", "--tgt-lang", lang,
            *(["--max-samples", str(self.max_samples)] if self.max_samples else []),
        ]
        steps.append(s)

        # 6. Data Prep — Eval
        s = PipelineStep(
            id="data-prep-eval",
            name="Prepare Eval Data",
            description="Translate English benchmarks → target language",
            depends_on=["classify"],
        )
        s._check_done = lambda: (
            self._eval_data_dir.exists()
            and len(list(self._eval_data_dir.glob("*.jsonl"))) >= 5
        )
        s._build_cmd = lambda: [
            self.python, "-m", "byol.data_prep",
            "--stage", "eval", "--tgt-lang", lang,
            *(["--max-samples", str(self.max_samples)] if self.max_samples else []),
        ]
        steps.append(s)

        # 7. Add Eval Tasks
        s = PipelineStep(
            id="add-eval-tasks",
            name="Generate Eval Task Configs",
            description="Create lm-eval YAML task definitions for the language",
            depends_on=["classify"],
        )
        s._check_done = lambda: (
            REPO_ROOT / "configs" / "eval" / f"benchmark_base_{self.tgt_lang}.yaml"
        ).exists()
        def _add_eval_cmd():
            self._ensure_data_symlink()
            return [
                self.python, "-m", "byol.eval",
                "add-language", "--lang", lang, "--name", self._lang_name,
            ]
        s._build_cmd = _add_eval_cmd
        steps.append(s)

        # 8. Train CPT
        s = PipelineStep(
            id="train-cpt",
            name="Continual Pre-Training",
            description=f"Train {self.model} on bilingual CPT data",
            depends_on=["data-prep-cpt"],
        )
        s._check_done = lambda: (
            self._cpt_jsonl.exists()
            and self._find_latest_checkpoint(self._train_cpt_dir, "cpt") is not None
        )
        def _cpt_cmd():
            cmd = [
                self.python, "-m", "byol.train", "cpt",
                "--model", self.model,
                "--dataset", f"{lang}_english_cpt",
                "--tgt-lang", lang,
                "--override", f"dataset_dir={self._lang_data / 'cpt' / 'bilingual_mix'}",
            ]
            if self.quick_test:
                cmd.extend([
                    "gradient_accumulation_steps=1",
                    "per_device_train_batch_size=1",
                    "cutoff_len=512",
                    "deepspeed=",
                    "--epochs", "1",
                ])
            cmd.extend(["--device", dev])
            return cmd
        s._build_cmd = _cpt_cmd
        steps.append(s)

        # 9. Train SFT
        s = PipelineStep(
            id="train-sft",
            name="Supervised Fine-Tuning",
            description="Fine-tune CPT checkpoint on instruction data",
            depends_on=["train-cpt", "data-prep-sft"],
        )
        s._check_done = lambda: (
            self._sft_jsonl.exists()
            and self._find_latest_checkpoint(self._train_sft_dir, "sft") is not None
        )
        def _sft_cmd():
            cpt_ckpt = self._find_latest_checkpoint(self._train_cpt_dir, "cpt")
            if not cpt_ckpt:
                return []
            cmd = [
                self.python, "-m", "byol.train", "sft",
                "--model", cpt_ckpt,
                "--dataset", f"{lang}_english_sft",
                "--eval-dataset", f"{lang}_sft_test",
                "--tgt-lang", lang,
                "--override", f"dataset_dir={self._lang_data / 'sft' / 'bilingual_mix'}",
            ]
            if self.quick_test:
                cmd.extend([
                    "gradient_accumulation_steps=1",
                    "per_device_train_batch_size=1",
                    "per_device_eval_batch_size=1",
                    "cutoff_len=512",
                    "save_steps=100", "eval_steps=100",
                    "deepspeed=",
                    "--epochs", "1",
                ])
            cmd.extend(["--device", dev])
            return cmd
        s._build_cmd = _sft_cmd
        steps.append(s)

        # 10. Model Merging
        # Paper Eq. 2: M(α,β) = G_PT + α(G_IT − G_PT) + β(E_ℓ − G_PT)
        # With α = 1−λ, β = λ, λ = 0.6 (best bilingual average)
        merge_desc = (
            f"Merge generalist ({self.instruct_model}) with language expert (SFT checkpoint)"
            if self.instruct_model
            else "Model merging (requires --instruct-model)"
        )
        s = PipelineStep(
            id="merge",
            name="Model Merging",
            description=merge_desc,
            depends_on=["train-sft"],
        )
        s._check_done = lambda: (
            self._find_latest_checkpoint(self._train_merged_dir, "merged") is not None
        )
        def _merge_cmd():
            if not self.instruct_model:
                logger.error(
                    "❌ --instruct-model is required for model merging. "
                    "Example: --instruct-model google/gemma-3-4b-it"
                )
                return []
            sft_ckpt = self._find_latest_checkpoint(self._train_sft_dir, "sft")
            if not sft_ckpt:
                return []
            timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = Path(self.model).name.replace("-", "_")
            out_dir = str(self._train_merged_dir / f"{model_short}_merged_{timestamp}")
            return [
                self.python, "-m", "byol.train.merge", "general",
                "--model-pt", self.model,
                "--model-it", self.instruct_model,
                "--model-el", sft_ckpt,
                "--beta", "0.6",
                "--output", out_dir,
                "--device", dev,
                "--dtype", "bfloat16",
            ]
        s._build_cmd = _merge_cmd
        steps.append(s)

        # 11. Eval Base
        s = PipelineStep(
            id="eval-base",
            name="Evaluate Base Model",
            description="Run benchmarks on CPT checkpoint (few-shot)",
            depends_on=["train-cpt", "data-prep-eval", "add-eval-tasks"],
        )
        s._check_done = lambda: (
            (self._results_dir / "eval" / "base").exists()
            and any(
                (self._results_dir / "eval" / "base").rglob("*/results_*.json")
            )
        )
        def _eval_base_cmd():
            self._ensure_data_symlink()
            cpt_ckpt = self._find_latest_checkpoint(self._train_cpt_dir, "cpt")
            if not cpt_ckpt:
                return []
            return [
                self.python, "-m", "byol.eval",
                "--model", cpt_ckpt,
                "--type", "base", "--tgt-lang", lang,
                "--device", dev,
                *(["--limit", str(self.max_samples)] if self.max_samples else []),
            ]
        s._build_cmd = _eval_base_cmd
        steps.append(s)

        # 12. Eval Instruct
        s = PipelineStep(
            id="eval-instruct",
            name="Evaluate Instruct Model",
            description="Run benchmarks on SFT checkpoint (0-shot + chat)",
            depends_on=["train-sft", "data-prep-eval", "add-eval-tasks"],
        )
        s._check_done = lambda: (
            (self._results_dir / "eval" / "instruct").exists()
            and any(
                (self._results_dir / "eval" / "instruct").rglob("*/results_*.json")
            )
        )
        def _eval_instruct_cmd():
            self._ensure_data_symlink()
            sft_ckpt = self._find_latest_checkpoint(self._train_sft_dir, "sft")
            if not sft_ckpt:
                return []
            return [
                self.python, "-m", "byol.eval",
                "--model", sft_ckpt,
                "--type", "instruct", "--tgt-lang", lang,
                "--device", dev,
                *(["--limit", str(self.max_samples)] if self.max_samples else []),
            ]
        s._build_cmd = _eval_instruct_cmd
        steps.append(s)

        # 13. Eval Merged
        s = PipelineStep(
            id="eval-merged",
            name="Evaluate Merged Model",
            description="Run benchmarks on merged model (0-shot + chat)",
            depends_on=["merge", "data-prep-eval", "add-eval-tasks"],
        )
        s._check_done = lambda: (
            (self._results_dir / "eval" / "merged").exists()
            and any(
                (self._results_dir / "eval" / "merged").rglob("*/results_*.json")
            )
        )
        def _eval_merged_cmd():
            self._ensure_data_symlink()
            merged_ckpt = self._find_latest_checkpoint(self._train_merged_dir, "merged")
            if not merged_ckpt:
                return []
            return [
                self.python, "-m", "byol.eval",
                "--model", merged_ckpt,
                "--type", "merged", "--tgt-lang", lang,
                "--device", dev,
                *(["--limit", str(self.max_samples)] if self.max_samples else []),
            ]
        s._build_cmd = _eval_merged_cmd
        steps.append(s)

        return steps

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> list[tuple[PipelineStep, str]]:
        """Return (step, status) for all steps."""
        result = []
        done_ids = set()
        for step in self.steps:
            if step.is_done():
                done_ids.add(step.id)
                result.append((step, STATUS_DONE))
            elif step._is_optional:
                # Optional steps that aren't done are skippable
                deps_met = all(
                    dep_id in done_ids or self._step_by_id(dep_id)._is_optional
                    for dep_id in step.depends_on
                )
                result.append((step, STATUS_PENDING if deps_met else STATUS_SKIPPED))
            else:
                deps_met = all(dep_id in done_ids for dep_id in step.depends_on)
                result.append((step, STATUS_PENDING if deps_met else STATUS_SKIPPED))
        return result

    def _step_by_id(self, step_id: str) -> PipelineStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise ValueError(f"Unknown step: {step_id}")

    def print_status(self) -> None:
        """Print pipeline status to terminal."""
        statuses = self.get_status()
        done_count = sum(1 for _, s in statuses if s == STATUS_DONE)
        total = len(statuses)

        print()
        print("═" * 64)
        print(f"  BYOL Pipeline — {self._lang_name} ({self.tgt_lang})")
        print(f"  Device: {self.device}    Model: {self.model}")
        if self.max_samples:
            print(f"  Max samples: {self.max_samples} (test mode)")
        if self.quick_test:
            print(f"  Training: quick-test (reduced settings)")
        print("═" * 64)
        print()

        for i, (step, status) in enumerate(statuses, 1):
            icon = STATUS_ICONS.get(status, "?")
            opt = " (optional)" if step._is_optional else ""
            print(f"  {i:2d}. {icon}  {step.name}{opt}")
            print(f"      {step.description}")
            if status == STATUS_DONE:
                pass
            elif status == STATUS_PENDING:
                cmd = step.build_command()
                if cmd:
                    print(f"      ▸ {_format_cmd(cmd)}")
            print()

        print("─" * 64)
        print(f"  Progress: {done_count}/{total} steps complete")

        # Find next
        nxt = self._next_step(statuses)
        if nxt:
            print(f"  Next: {nxt.name}")
            print(f"  Run:  python -m byol.pipeline run --tgt-lang {self.tgt_lang} --step {nxt.id}")
        else:
            print("  🎉 Pipeline complete!")
        print("─" * 64)
        print()

    # ── Execution ────────────────────────────────────────────────────────

    def _next_step(
        self, statuses: list[tuple[PipelineStep, str]] | None = None,
    ) -> PipelineStep | None:
        """Find the next step to run."""
        if statuses is None:
            statuses = self.get_status()
        done_ids = {step.id for step, s in statuses if s == STATUS_DONE}
        for step, status in statuses:
            if status != STATUS_PENDING:
                continue
            if step._is_optional:
                continue
            deps_met = all(d in done_ids for d in step.depends_on)
            if deps_met:
                return step
        # If all required are done, check optional
        for step, status in statuses:
            if status == STATUS_PENDING and step._is_optional:
                deps_met = all(d in done_ids for d in step.depends_on)
                if deps_met:
                    return step
        return None

    def run_step(self, step_id: str) -> StepResult:
        """Run a single step by ID."""
        step = self._step_by_id(step_id)

        if step.is_done():
            msg = f"Step '{step.name}' is already complete — skipping."
            logger.info(msg)
            print(f"  ⏭️  {msg}")
            return StepResult(success=True, message=msg)

        cmd = step.build_command()
        if not cmd:
            msg = f"Step '{step.name}' cannot build command (missing dependencies?)"
            logger.error(msg)
            print(f"  ❌ {msg}")
            return StepResult(success=False, message=msg)

        # Warn about reduced training settings in quick-test mode
        if step.id in ("train-cpt", "train-sft") and self.quick_test:
            print()
            print("  ⚠️  Quick-test mode: using reduced training settings (epochs=1, batch=1).")
            print("     For full training, run without --quick-test.")
            print()

        print()
        print(f"  🔄 Running: {step.name}")
        print(f"     {_format_cmd(cmd)}")
        print()

        env = os.environ.copy()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
            )
            if proc.returncode == 0:
                msg = f"Step '{step.name}' completed successfully."
                print(f"\n  ✅ {msg}")
                return StepResult(success=True, message=msg)
            else:
                msg = f"Step '{step.name}' failed with exit code {proc.returncode}."
                print(f"\n  ❌ {msg}")
                return StepResult(success=False, message=msg)
        except Exception as e:
            msg = f"Step '{step.name}' raised exception: {e}"
            print(f"\n  ❌ {msg}")
            return StepResult(success=False, message=msg)

    def run_next(self) -> StepResult | None:
        """Find and run the next pending step."""
        nxt = self._next_step()
        if nxt is None:
            print("\n  🎉 All pipeline steps are complete!")
            return None
        return self.run_step(nxt.id)

    def run_all(self, skip_optional: bool = True) -> list[StepResult]:
        """Run all pending steps in order."""
        results = []
        while True:
            statuses = self.get_status()
            nxt = self._next_step(statuses)
            if nxt is None:
                break
            if skip_optional and nxt._is_optional:
                print(f"\n  ⏭️  Skipping optional: {nxt.name}")
                # Mark optional as "done" for dependency purposes by moving on
                break
            result = self.run_step(nxt.id)
            results.append(result)
            if not result.success:
                print(f"\n  ⛔ Pipeline stopped at '{nxt.name}'. Fix the issue and re-run.")
                break
        return results

    # ── Cleanup ──────────────────────────────────────────────────────────

    def collect_artifacts(self) -> list[tuple[str, str]]:
        """Collect all artifact paths for this language.

        Returns a list of (description, path) tuples.
        Paths may be files, directories, or symlinks.
        Only existing paths are returned.
        """
        lang = self.tgt_lang
        items: list[tuple[str, str]] = []

        # 1. ~/byol-data/<lang>  (CPT/SFT/eval data)
        p = self._lang_data
        if p.exists():
            items.append(("Language data (CPT/SFT/eval)", str(p)))

        # 2. results/<lang>  (LRA, train checkpoints, eval results)
        p = self._results_dir
        if p.exists():
            items.append(("Results (LRA, train, eval)", str(p)))

        # 3. data/<lang> symlink
        p = REPO_ROOT / "data" / lang
        if p.exists() or p.is_symlink():
            items.append(("Data symlink", str(p)))

        # 4. Data prep configs: configs/data_prep/{cpt,sft,eval}/<lang>.yaml
        for stage in ("cpt", "sft", "eval"):
            p = REPO_ROOT / "configs" / "data_prep" / stage / f"{lang}.yaml"
            if p.exists():
                items.append((f"Data prep config ({stage})", str(p)))

        # 5. Eval benchmark configs: configs/eval/benchmark_{base,instruct}_<lang>.yaml
        for kind in ("base", "instruct"):
            p = REPO_ROOT / "configs" / "eval" / f"benchmark_{kind}_{lang}.yaml"
            if p.exists():
                items.append((f"Eval config ({kind})", str(p)))

        # 6. Eval task YAMLs (scattered across byol/eval/tasks/)
        tasks_root = REPO_ROOT / "byol" / "eval" / "tasks"
        if tasks_root.exists():
            for yaml_path in sorted(tasks_root.rglob(f"*{lang}*")):
                if yaml_path.is_file() and yaml_path.suffix in (".yaml", ".yml", ".py"):
                    rel = yaml_path.relative_to(REPO_ROOT)
                    items.append(("Eval task file", str(yaml_path)))
            # Also check for language-named directories (e.g. Global-MMLU-Lite/<lang>/)
            for dir_path in sorted(tasks_root.rglob(lang)):
                if dir_path.is_dir():
                    items.append(("Eval task directory", str(dir_path)))

        return items

    def clean(self, force: bool = False) -> bool:
        """Remove all artifacts for this language.

        Lists what will be deleted, asks for confirmation (unless force=True),
        then deletes.

        Returns True if cleanup was performed, False if cancelled.
        """
        import shutil

        artifacts = self.collect_artifacts()
        if not artifacts:
            print(f"\n  ✨ No artifacts found for {self._lang_name} ({self.tgt_lang}). Already clean.")
            return True

        print()
        print("═" * 64)
        print(f"  Cleanup — {self._lang_name} ({self.tgt_lang})")
        print("═" * 64)
        print()
        print(f"  The following {len(artifacts)} item(s) will be DELETED:")
        print()

        for desc, path in artifacts:
            # Show size for directories
            p = Path(path)
            if p.is_dir() and not p.is_symlink():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                size_str = _format_size(size)
                print(f"    📁 {desc}")
                print(f"       {path}  ({size_str})")
            elif p.is_symlink():
                print(f"    🔗 {desc}")
                print(f"       {path} -> {os.readlink(path)}")
            else:
                print(f"    📄 {desc}")
                print(f"       {path}")
            print()

        print("─" * 64)

        if not force:
            try:
                answer = input("  Proceed with deletion? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return False
            if answer not in ("y", "yes"):
                print("  Cancelled.")
                return False

        # Delete
        deleted = 0
        for desc, path in artifacts:
            p = Path(path)
            try:
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
                deleted += 1
                print(f"  🗑️  Deleted: {path}")
            except Exception as e:
                print(f"  ⚠️  Failed to delete {path}: {e}")

        print()
        print(f"  ✅ Cleaned {deleted}/{len(artifacts)} items for {self._lang_name} ({self.tgt_lang})")
        print()
        return True


# =============================================================================
# Helpers
# =============================================================================


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _format_cmd(cmd: list[str]) -> str:
    """Format a command list as a readable string."""
    parts = []
    for arg in cmd:
        if " " in arg or "=" in arg:
            parts.append(f'"{arg}"')
        else:
            parts.append(arg)
    s = " ".join(parts)
    # Replace absolute python path with just 'python'
    if "/python" in s:
        idx = s.find("/python")
        # Find the start of this path segment
        start = s.rfind(" ", 0, idx)
        if start == -1:
            start = 0
        else:
            start += 1
        s = s[:start] + "python" + s[idx + len("/python"):]
        # Also clean up any version suffix like python3.12
        s = s.replace("python3.12", "python")
    return s
