# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Evaluation Runner for BYOL Framework.

Orchestrates model evaluation across multiple tasks using the
lm-evaluation-harness framework as sub-processes.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import EvalConfig, ModelConfig, TaskConfig
from .constants import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    LANG_NAMES,
    STATUS_FAILED,
    STATUS_ICONS,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    UNSAFE_TASKS,
)
from .secrets import setup_hf_environment

logger = logging.getLogger(__name__)


def _yaml_safe_load(path: str) -> dict | None:
    """Load a YAML file, ignoring lm-eval custom tags like ``!function``."""
    import yaml as _yaml

    class _PermissiveLoader(_yaml.SafeLoader):
        pass

    _PermissiveLoader.add_multi_constructor(
        "", lambda loader, suffix, node: None
    )

    try:
        with open(path) as f:
            return _yaml.load(f, Loader=_PermissiveLoader) or {}
    except Exception:
        return None


# =============================================================================
# Result Data Structure
# =============================================================================


@dataclass
class EvalResult:
    """Result of a single evaluation run.

    Attributes:
        model: Model name that was evaluated.
        task: Task name that was run.
        status: Evaluation status (``success``, ``failed``, ``skipped``).
        output_dir: Directory where results were saved.
        error: Error message if evaluation failed.
        duration_seconds: Time taken for evaluation.
        metrics: Extracted metrics mapping ``metric_name`` → value.
    """

    model: str
    task: str
    status: str
    output_dir: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Runner
# =============================================================================


class EvaluationRunner:
    """Main evaluation runner using lm-evaluation-harness.

    Orchestrates model evaluation across multiple tasks, printing a run plan
    before execution and a results summary afterwards.

    Args:
        config: Evaluation configuration.
        dry_run: If ``True``, print commands without executing.
        eval_type: Evaluation type (``'base'`` or ``'instruct'``).
        tgt_lang: Target language code.
        config_path: Path to the benchmark config file used.
    """

    def __init__(
        self,
        config: EvalConfig,
        dry_run: bool = False,
        skip_existing: bool = False,
        overwrite: bool = False,
        eval_type: str = "",
        tgt_lang: str = "",
        config_path: str = "",
        data_dir: str = "",
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        self.overwrite = overwrite
        self.eval_type = eval_type
        self.tgt_lang = tgt_lang
        self.config_path = config_path
        self.data_dir = data_dir
        setup_hf_environment(config.hf_token)
        self._task_config_cache: dict[str, str] = {}
        self._patched_tasks_dir: str | None = None
        if self.data_dir:
            self._patched_tasks_dir = self._create_patched_tasks()

    # ------------------------------------------------------------------ #
    # Task config lookup
    # ------------------------------------------------------------------ #

    def _find_task_config(self, task_name: str) -> str:
        """Find the YAML config file path for a task under *tasks_path*.

        Searches for ``<task_name>.yaml``, ``_<task_name>.yaml`` (group
        files), or YAML files containing ``group: <task_name>`` /
        ``task: <task_name>`` under the configured ``tasks_path``.

        Returns:
            A path relative to cwd, or ``"built-in"`` if lm-eval provides the
            task natively.
        """
        if task_name in self._task_config_cache:
            return self._task_config_cache[task_name]

        result = "built-in"
        if self.config.tasks_path:
            tasks_root = Path(self.config.tasks_path)
            if tasks_root.exists():
                # Pass 1: match by filename
                for yaml_path in tasks_root.rglob("*.yaml"):
                    stem = yaml_path.stem
                    if stem == task_name or stem == f"_{task_name}":
                        try:
                            result = str(yaml_path.relative_to(Path.cwd()))
                        except ValueError:
                            result = str(yaml_path)
                        break
                # Pass 2: search inside files
                if result == "built-in":
                    for yaml_path in tasks_root.rglob("*.yaml"):
                        try:
                            text = yaml_path.read_text(errors="ignore")
                        except OSError:
                            continue
                        for line in text.splitlines():
                            stripped = line.strip()
                            if stripped in (
                                f"group: {task_name}",
                                f"task: {task_name}",
                            ):
                                try:
                                    result = str(yaml_path.relative_to(Path.cwd()))
                                except ValueError:
                                    result = str(yaml_path)
                                break
                        if result != "built-in":
                            break
        self._task_config_cache[task_name] = result
        return result

    # ------------------------------------------------------------------ #
    # Data directory remapping
    # ------------------------------------------------------------------ #

    def _create_patched_tasks(self) -> str:
        """Create a temporary tasks directory with data_files remapped to --data-dir.

        Copies the original task YAMLs but rewrites any ``data_files`` paths
        to point at matching files inside the user-provided ``--data-dir``.
        Files are matched by ``{benchmark}_{split}`` prefix, so different
        translator suffixes are handled automatically.

        Returns:
            Path to the temporary tasks directory.
        """
        import shutil
        import tempfile

        src = Path(self.config.tasks_path) if self.config.tasks_path else None
        if not src or not src.exists():
            return ""

        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            logger.warning(f"--data-dir does not exist: {self.data_dir}")
            return ""

        # Index available files in data_dir by (benchmark, split) prefix
        available: dict[str, str] = {}
        for f in data_dir.iterdir():
            if f.is_file() and f.suffix == ".jsonl":
                available[f.name] = str(f.resolve())

        tmp = tempfile.mkdtemp(prefix="byol_eval_tasks_")
        shutil.copytree(src, Path(tmp) / "tasks", dirs_exist_ok=True)
        tasks_root = Path(tmp) / "tasks"

        for yaml_path in tasks_root.rglob("*.yaml"):
            try:
                text = yaml_path.read_text()
            except OSError:
                continue

            # Parse only to find data_files paths (using permissive loader)
            content = _yaml_safe_load(str(yaml_path))
            if not content:
                continue
            data_files = (content.get("dataset_kwargs") or {}).get("data_files")
            if not data_files or not isinstance(data_files, dict):
                continue

            # Do text-level replacement to preserve !function tags etc.
            new_text = text
            for split, orig_path in data_files.items():
                if not isinstance(orig_path, str) or not orig_path.startswith("data/"):
                    continue
                orig_name = Path(orig_path).name
                prefix = orig_name.split("_english2")[0] if "_english2" in orig_name else orig_name.rsplit("_", 1)[0]
                match = self._find_matching_file(prefix, split, available)
                if match:
                    new_text = new_text.replace(orig_path, match)

            if new_text != text:
                yaml_path.write_text(new_text)

        return str(tasks_root)

    @staticmethod
    def _find_matching_file(prefix: str, split: str, available: dict[str, str]) -> str:
        """Find a file in the available dict matching the benchmark/split prefix."""
        # Try exact prefix match: {prefix}_{split}_ in available filenames
        search = f"{prefix}_{split}_"
        for name, full_path in available.items():
            if name.startswith(search):
                return full_path
        # Fallback: prefix with split anywhere in name
        for name, full_path in available.items():
            if name.startswith(prefix) and split in name:
                return full_path
        # Fallback: match by benchmark name (first two parts, e.g. "xwinograd_aligned")
        # Handles files like xwinograd_aligned_<lang>_1000.jsonl where lang varies
        parts = prefix.split("_")
        if len(parts) >= 2:
            short_prefix = "_".join(parts[:2])
            for name, full_path in available.items():
                if name.startswith(short_prefix):
                    return full_path
        return ""

    def _build_command(self, model: ModelConfig, task: TaskConfig) -> list[str]:
        """Build the ``lm_eval`` command for a given model/task pair.

        Args:
            model: Model configuration.
            task: Task configuration.

        Returns:
            Command as a list of strings.
        """
        output_dir = self._get_output_dir(model, task)

        model_args = [
            f"pretrained={model.path}",
            f"dtype={model.dtype}",
            f"trust_remote_code={str(model.trust_remote_code).lower()}",
        ]

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", ",".join(model_args),
            "--tasks", task.name,
            "--output_path", output_dir,
        ]

        batch_size = task.batch_size or self.config.batch_size
        cmd.extend(["--batch_size", str(batch_size)])

        if task.num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(task.num_fewshot)])

        if self._patched_tasks_dir:
            cmd.extend(["--include_path", self._patched_tasks_dir])
        elif self.config.tasks_path:
            cmd.extend(["--include_path", self.config.tasks_path])

        if task.limit is not None:
            cmd.extend(["--limit", str(task.limit)])

        if self.config.log_samples:
            cmd.append("--log_samples")

        if task.apply_chat_template or self.config.apply_chat_template:
            cmd.append("--apply_chat_template")

        task_names = {t.strip().lower() for t in task.name.split(",")}
        if task_names & UNSAFE_TASKS:
            cmd.append("--confirm_run_unsafe_code")

        return cmd

    def _get_output_dir(self, model: ModelConfig, task: TaskConfig) -> str:
        """Generate output directory path.

        Structure: ``results/<lang>/eval/<type>/<model_name>/<task>``.
        """
        safe_model = model.name.replace("/", "_").replace("-", "_")
        safe_task = task.name.replace(",", "_").replace(" ", "_").replace("/", "_")
        eval_type = self.eval_type or "unknown"
        lang = self.tgt_lang or "unknown"
        return str(Path("results") / lang / "eval" / eval_type / safe_model / safe_task)

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run_single(self, model: ModelConfig, task: TaskConfig) -> EvalResult:
        """Run evaluation for a single model/task pair.

        Subprocess output goes to a log file.  Progress bars and the final
        results table are also shown on the terminal.

        Args:
            model: Model to evaluate.
            task: Task to run.

        Returns:
            ``EvalResult`` with status and metadata.
        """
        start_time = datetime.now()
        cmd = self._build_command(model, task)
        output_dir = cmd[cmd.index("--output_path") + 1]

        if not self.overwrite and self._has_completed_run(output_dir):
            print(f"  │  Results already exist at: {output_dir}")
            print(f"  │  To re-run, use: --overwrite")
            return EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)

        if self.dry_run:
            return EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)

        # Check if task data files exist before running lm-eval
        missing = self._check_task_data_files(task)
        if missing:
            for path in missing:
                print(f"  │  ⚠️  Missing data: {path}")
            print(f"  │  Run data prep first: python -m byol.data_prep --stage eval --tgt-lang {self.tgt_lang}")
            return EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(output_dir) / "lm_eval.log"

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.config.gpus

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT),
            )

            results_lines: list[str] = []

            def _stream_stderr(
                proc_stderr: io.BufferedReader,
                log_file: io.TextIOWrapper,
            ) -> None:
                in_results_table = False
                for raw_line in iter(proc_stderr.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace")
                    log_file.write(line)
                    log_file.flush()
                    stripped = line.strip()
                    if stripped.startswith("|"):
                        in_results_table = True
                        results_lines.append(line)
                        continue
                    if in_results_table and not stripped.startswith("|"):
                        in_results_table = False
                    if "%|" in line or "it/s" in line:
                        sys.stderr.write(line)
                        sys.stderr.flush()

            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Started: {start_time.isoformat()}\n")
                log_file.write("=" * 80 + "\n\n")

                stderr_thread = threading.Thread(
                    target=_stream_stderr,
                    args=(proc.stderr, log_file),
                    daemon=True,
                )
                stderr_thread.start()

                for raw_line in iter(proc.stdout.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace")
                    log_file.write(line)
                    if line.strip().startswith("|"):
                        results_lines.append(line)

                proc.wait()
                stderr_thread.join(timeout=5)

                log_file.write(f"\nFinished: {datetime.now().isoformat()}\n")
                log_file.write(f"Exit code: {proc.returncode}\n")

            duration = (datetime.now() - start_time).total_seconds()

            if results_lines:
                sys.stderr.write("\n")
                for rl in results_lines:
                    sys.stderr.write(rl)
                sys.stderr.flush()

            if proc.returncode == 0:
                metrics = self._extract_metrics(output_dir, task.name)
                return EvalResult(
                    model.name, task.name, STATUS_SUCCESS, output_dir,
                    duration_seconds=duration, metrics=metrics,
                )
            error_hint = self._extract_error_hint(log_path)
            return EvalResult(
                model.name, task.name, STATUS_FAILED, output_dir,
                error_hint or f"Exit code: {proc.returncode}", duration,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return EvalResult(
                model.name, task.name, STATUS_FAILED,
                error=str(e), duration_seconds=duration,
            )

    @staticmethod
    def _extract_metrics(output_dir: str, task_name: str) -> dict[str, float]:
        """Parse the lm-eval results JSON and extract key metrics.

        Looks for the top-level task entry and returns all numeric metrics,
        excluding ``stderr`` entries.
        """
        metrics: dict[str, float] = {}
        json_files = list(Path(output_dir).rglob("results_*.json"))
        if not json_files:
            return metrics
        try:
            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return metrics

        results = data.get("results", {})
        task_keys = [task_name] + [n.strip() for n in task_name.split(",")]
        for key in task_keys:
            if key in results:
                for metric_key, value in results[key].items():
                    if (
                        isinstance(value, (int, float))
                        and "stderr" not in metric_key
                        and metric_key != "alias"
                    ):
                        metrics[metric_key] = value
                if metrics:
                    break
        return metrics

    @staticmethod
    def _has_completed_run(output_dir: str) -> bool:
        """Check if a prior run completed successfully in output_dir."""
        out_path = Path(output_dir)
        if not out_path.exists():
            return False
        json_files = list(out_path.rglob("results_*.json"))
        if not json_files:
            return False
        log_path = out_path / "lm_eval.log"
        if not log_path.exists():
            return False
        try:
            with open(log_path, "rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(size - 4096, 0))
                tail = log_file.read().decode("utf-8", errors="ignore")
        except OSError:
            return False
        return "Exit code: 0" in tail

    def _check_task_data_files(self, task) -> list[str]:
        """Check if local data files referenced by a task YAML exist.

        Returns a list of missing file paths (empty if all present or task
        uses remote datasets).  When ``--data-dir`` is active, checks the
        patched YAMLs with remapped paths instead.
        """
        missing = []
        # Use patched task dir if available
        if self._patched_tasks_dir:
            task_yaml = self._find_task_in_dir(task.name, self._patched_tasks_dir)
        else:
            task_yaml = self._find_task_config(task.name)
        if not task_yaml or task_yaml == "built-in":
            return missing

        content = _yaml_safe_load(task_yaml)
        if not content:
            return missing

        data_files = (content.get("dataset_kwargs") or {}).get("data_files")
        if not data_files or not isinstance(data_files, dict):
            return missing

        for split, path in data_files.items():
            if not isinstance(path, str):
                continue
            # Absolute path (from --data-dir remapping) or relative path
            full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
            if path.startswith("data/") or Path(path).is_absolute():
                if not full.exists():
                    missing.append(path)
        return missing

    @staticmethod
    def _find_task_in_dir(task_name: str, tasks_dir: str) -> str:
        """Find a task YAML in a specific directory."""
        root = Path(tasks_dir)
        for yaml_path in root.rglob("*.yaml"):
            if yaml_path.stem == task_name or yaml_path.stem == f"_{task_name}":
                return str(yaml_path)
        return ""

    @staticmethod
    def _extract_error_hint(log_path: Path) -> str:
        """Extract a human-readable error hint from an lm-eval log file."""
        if not log_path.exists():
            return ""
        try:
            text = log_path.read_text(errors="ignore")
        except OSError:
            return ""
        if "Sample larger than population" in text:
            return "Not enough data rows for few-shot sampling (dataset too small for num_fewshot)"
        if "is out of bounds for size 0" in text:
            return "Dataset split has 0 rows (translated data may be missing or empty)"
        if "len(continuation_enc) <= self.max_length" in text:
            return "Tokenized input exceeds model max length"
        if "CUDA out of memory" in text:
            return "GPU out of memory"
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped.startswith(("ValueError:", "IndexError:", "RuntimeError:", "AssertionError", "FileNotFoundError:")):
                return stripped[:120]
        return ""

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run_all(self) -> list[EvalResult]:
        """Run all evaluations defined in the configuration.

        Groups compatible tasks into single lm-eval calls to avoid
        reloading the model for every task.  Tasks are compatible when they
        share the same ``num_fewshot``, ``batch_size``, ``limit``, and
        ``apply_chat_template`` settings.

        Returns:
            List of ``EvalResult`` objects for all model/task combinations.
        """
        self._print_run_plan()

        results: list[EvalResult] = []
        total = len(self.config.models) * len(self.config.tasks)
        idx = 0

        for model in self.config.models:
            # Phase 1: pre-filter tasks (skip/missing checks)
            runnable: list[TaskConfig] = []
            for task in self.config.tasks:
                idx += 1
                print(f"\n  ╭── [{idx}/{total}] {task.name}")
                print("  │")

                output_dir = self._get_output_dir(model, task)

                if not self.overwrite and self._has_completed_run(output_dir):
                    print(f"  │  Results already exist at: {output_dir}")
                    print(f"  │  To re-run, use: --overwrite")
                    result = EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)
                    self._print_result_footer(result)
                    results.append(result)
                    continue

                if self.dry_run:
                    result = EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)
                    self._print_result_footer(result)
                    results.append(result)
                    continue

                missing = self._check_task_data_files(task)
                if missing:
                    for path in missing:
                        print(f"  │  ⚠️  Missing data: {path}")
                    print(f"  │  Run data prep first: python -m byol.data_prep --stage eval --tgt-lang {self.tgt_lang}")
                    result = EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)
                    self._print_result_footer(result)
                    results.append(result)
                    continue

                print(f"  │  ⏳ queued for grouped execution")
                print(f"  │  📁 {output_dir}")
                print(f"  ╰{'─' * 56}")
                runnable.append(task)

            if not runnable:
                continue

            # Phase 2: group compatible tasks
            groups = self._group_tasks(runnable)
            print(f"\n  ── Executing {len(runnable)} tasks in {len(groups)} group(s) ──\n")

            for group_idx, (group_key, tasks) in enumerate(groups.items(), 1):
                task_names = [t.name for t in tasks]
                print(f"  ╭── Group {group_idx}/{len(groups)}: {len(tasks)} task(s)")
                print(f"  │  Tasks: {', '.join(task_names)}")
                print("  │")

                group_results = self._run_grouped(model, tasks)

                # If the grouped run failed and there are multiple tasks,
                # retry each task individually to isolate the failure
                all_failed = all(r.status == STATUS_FAILED for r in group_results)
                if all_failed and len(tasks) > 1:
                    print(f"  │  ⚠️  Group failed — retrying {len(tasks)} tasks individually...")
                    group_results = []
                    for task in tasks:
                        print(f"\n  ╭── Retry: {task.name}")
                        print("  │")
                        result = self.run_single(model, task)
                        self._print_result_footer(result)
                        group_results.append(result)
                else:
                    for result in group_results:
                        self._print_result_footer(result)

                results.extend(group_results)

        return results

    @staticmethod
    def _group_tasks(tasks: list[TaskConfig]) -> dict[tuple, list[TaskConfig]]:
        """Group tasks by compatible execution parameters."""
        groups: dict[tuple, list[TaskConfig]] = {}
        for task in tasks:
            key = (
                task.num_fewshot,
                task.batch_size,
                task.limit,
                task.apply_chat_template,
            )
            groups.setdefault(key, []).append(task)
        return groups

    def _run_grouped(self, model: ModelConfig, tasks: list[TaskConfig]) -> list[EvalResult]:
        """Run a group of compatible tasks in a single lm-eval call."""
        representative = tasks[0]
        combined_name = ",".join(t.name for t in tasks)

        # Build output path for the combined run
        safe_model = model.name.replace("/", "_").replace("-", "_")
        eval_type = self.eval_type or "unknown"
        lang = self.tgt_lang or "unknown"
        combined_safe = combined_name.replace(",", "_").replace(" ", "_").replace("/", "_")
        combined_output = str(Path("results") / lang / "eval" / eval_type / safe_model / combined_safe)

        model_args = [
            f"pretrained={model.path}",
            f"dtype={model.dtype}",
            f"trust_remote_code={str(model.trust_remote_code).lower()}",
        ]

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", ",".join(model_args),
            "--tasks", combined_name,
            "--output_path", combined_output,
        ]

        batch_size = representative.batch_size or self.config.batch_size
        cmd.extend(["--batch_size", str(batch_size)])

        if representative.num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(representative.num_fewshot)])

        if self._patched_tasks_dir:
            cmd.extend(["--include_path", self._patched_tasks_dir])
        elif self.config.tasks_path:
            cmd.extend(["--include_path", self.config.tasks_path])

        if representative.limit is not None:
            cmd.extend(["--limit", str(representative.limit)])

        if self.config.log_samples:
            cmd.append("--log_samples")

        if representative.apply_chat_template or self.config.apply_chat_template:
            cmd.append("--apply_chat_template")

        all_task_names = set()
        for t in tasks:
            all_task_names.update(n.strip().lower() for n in t.name.split(","))
        if all_task_names & UNSAFE_TASKS:
            cmd.append("--confirm_run_unsafe_code")

        # Execute
        start_time = datetime.now()
        Path(combined_output).mkdir(parents=True, exist_ok=True)
        log_path = Path(combined_output) / "lm_eval.log"

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.config.gpus

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT),
            )

            results_lines: list[str] = []

            def _stream_stderr(
                proc_stderr: io.BufferedReader,
                log_file: io.TextIOWrapper,
            ) -> None:
                in_results_table = False
                for raw_line in iter(proc_stderr.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace")
                    log_file.write(line)
                    log_file.flush()
                    stripped = line.strip()
                    if stripped.startswith("|") and "Tasks" in stripped:
                        in_results_table = True
                    if in_results_table:
                        if stripped.startswith("|"):
                            results_lines.append(line)
                        elif stripped == "":
                            in_results_table = False
                    sys.stderr.write(line)
                    sys.stderr.flush()

            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"Command: {' '.join(cmd)}\n\n")
                stderr_thread = threading.Thread(
                    target=_stream_stderr,
                    args=(proc.stderr, log_file),
                    daemon=True,
                )
                stderr_thread.start()
                proc.wait()
                stderr_thread.join(timeout=5)
                log_file.write(f"\nFinished: {datetime.now().isoformat()}\n")
                log_file.write(f"Exit code: {proc.returncode}\n")

            duration = (datetime.now() - start_time).total_seconds()

            if results_lines:
                sys.stderr.write("\n")
                for rl in results_lines:
                    sys.stderr.write(rl)
                sys.stderr.flush()

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return [
                EvalResult(model.name, t.name, STATUS_FAILED, error=str(e), duration_seconds=duration)
                for t in tasks
            ]

        # Build per-task results
        per_task_results: list[EvalResult] = []

        # Copy combined results to per-task output dirs
        for task in tasks:
            task_output = self._get_output_dir(model, task)
            Path(task_output).mkdir(parents=True, exist_ok=True)

            # Copy the combined results JSON to each task dir
            for json_file in Path(combined_output).rglob("results_*.json"):
                import shutil
                dest = Path(task_output) / json_file.parent.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_file, dest / json_file.name)

            # Copy log
            task_log = Path(task_output) / "lm_eval.log"
            if log_path.exists():
                import shutil
                shutil.copy2(log_path, task_log)

            if proc.returncode == 0:
                metrics = self._extract_metrics(combined_output, task.name)
                per_task_results.append(EvalResult(
                    model.name, task.name, STATUS_SUCCESS, task_output,
                    duration_seconds=duration / len(tasks), metrics=metrics,
                ))
            else:
                error_hint = self._extract_error_hint(log_path)
                per_task_results.append(EvalResult(
                    model.name, task.name, STATUS_FAILED, task_output,
                    error_hint or f"Exit code: {proc.returncode}",
                    duration / len(tasks),
                ))

        return per_task_results

    def _print_result_footer(self, result: EvalResult) -> None:
        """Print a single result line."""
        icon = STATUS_ICONS.get(result.status, "?")
        dur = f" ({result.duration_seconds:.1f}s)" if result.duration_seconds else ""
        print(f"  │  {icon} {result.status}{dur}")
        if result.error and result.status == STATUS_FAILED:
            print(f"  │  Reason: {result.error}")
        if result.metrics:
            metric_strs = [f"{k}: {v:.4f}" for k, v in result.metrics.items()]
            print(f"  │     {' · '.join(metric_strs)}")
        if result.output_dir:
            print(f"  │  📁 {result.output_dir}")
        print(f"  ╰{'─' * 56}")

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #

    def _print_run_plan(self) -> None:
        """Print a formatted summary table of what will be run."""
        W = 62
        chat = "Yes" if self.config.apply_chat_template else "No"
        model = self.config.models[0] if self.config.models else None
        lang_display = LANG_NAMES.get(self.tgt_lang, self.tgt_lang)
        eval_type_display = self.eval_type.capitalize() if self.eval_type else "—"

        print()
        print(f"  ┌{'─' * W}")
        print("  │  BYOL Evaluation")
        print(f"  ├{'─' * W}")
        if model:
            print(f"  │  Model       {model.path}")
        print(f"  │  Type        {eval_type_display}")
        print(f"  │  Language    {lang_display} ({self.tgt_lang})")
        if model:
            print(f"  │  Dtype       {model.dtype}")
        print(f"  │  GPUs        {self.config.gpus}")
        print(f"  │  Chat Template    {chat}")
        print(f"  │  Skip Existing    {'Yes' if self.skip_existing else 'No'}")
        print(f"  │  Batch       {self.config.batch_size}")
        lang = self.tgt_lang or "unknown"
        print(f"  │  Output      results/{lang}/eval/")
        if self.config_path:
            print(f"  │  Config      {self.config_path}")
        if self.dry_run:
            print("  │  Mode        DRY RUN")
        print(f"  ├{'─' * W}")
        print(f"  │  {'#':<4} {'Task':<36} {'Fewshot':<9} Batch")
        print(f"  │  {'─' * 4} {'─' * 36} {'─' * 9} {'─' * 10}")
        for i, task in enumerate(self.config.tasks, 1):
            fewshot = str(task.num_fewshot) if task.num_fewshot is not None else "default"
            batch = task.batch_size or self.config.batch_size
            first_name = task.name.split(",")[0].strip()
            cfg_path = self._find_task_config(first_name)
            print(f"  │  {i:<4} {task.name:<36} {fewshot:<9} {batch}")
            print(f"  │       └─ {cfg_path}")
        total = len(self.config.models) * len(self.config.tasks)
        print(f"  ├{'─' * W}")
        print(f"  │  Total: {total} evaluation(s)")
        print(f"  └{'─' * W}")
        print()

    @staticmethod
    def _format_metric_name(key: str) -> str:
        """Convert a metric key like ``'acc,none'`` to a readable name."""
        name = key.split(",")[0]
        return name.replace("_", " ")

    @staticmethod
    def print_summary(results: list[EvalResult]) -> None:
        """Print evaluation summary with metrics.

        Args:
            results: List of evaluation results.
        """
        W = 62
        successful = sum(1 for r in results if r.status == STATUS_SUCCESS)
        total_time = sum(r.duration_seconds for r in results)

        print()
        print(f"  ┌{'─' * W}")
        print("  │  Results Summary")
        print(f"  ├{'─' * W}")
        for r in results:
            icon = STATUS_ICONS.get(r.status, "?")
            dur = f"  ({r.duration_seconds:.1f}s)" if r.duration_seconds else ""
            print(f"  │  {icon} {r.task}{dur}")
            if r.metrics:
                parts = [
                    f"{EvaluationRunner._format_metric_name(k)}: {v:.4f}"
                    for k, v in r.metrics.items()
                ]
                print(f"  │     {' · '.join(parts)}")
        print(f"  ├{'─' * W}")
        print(f"  │  {successful}/{len(results)} passed  ·  {total_time:.1f}s total")
        print(f"  └{'─' * W}")
        print()
        # Guide user to extract benchmark tables
        if results:
            out_dir = results[0].output_dir or DEFAULT_OUTPUT_DIR
            print(f"  💡 To extract result tables: python -m byol.eval.extract_benchmarks {out_dir}")
            print()
