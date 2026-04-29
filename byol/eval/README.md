# byol.eval — Model Evaluation Framework

Evaluate language models on multilingual benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and LLM-as-Judge.

Supports **English (eng)**, **Māori (mri)**, and **Chichewa (nya)** with **base** (few-shot) and **instruct** (0-shot + chat template) evaluation modes.

## Quick Start

```bash
# Base model evaluation — Chichewa
python -m byol.eval --model google/gemma-3-4b-pt --type base --tgt-lang nya --device 0

# Instruct model evaluation — Māori
python -m byol.eval --model google/gemma-3-4b-it --type instruct --tgt-lang mri --device 0,1

# Use external translated data folder
python -m byol.eval --model google/gemma-3-4b-pt --type base --tgt-lang nya --device 3 \
  --data-dir /path/to/translated/eval/data/

# Single task evaluation
python -m byol.eval --model google/gemma-3-4b-it --type instruct --tgt-lang eng --device 3 --tasks copa

# Dry run (preview commands without executing)
python -m byol.eval --model google/gemma-3-4b-pt --type base --tgt-lang eng --dry-run

# LLM-as-Judge evaluation
python -m byol.eval judge --model-config configs/eval/judge_models.yaml \
                         --dataset-config configs/eval/judge_datasets.yaml

# Extract benchmark scores from results
python -m byol.eval.extract_benchmarks results/<lang>/eval/ --type base --tgt-lang eng
```

## Architecture

### Evaluation Modes

| Mode | `--type` | Description |
|---|---|---|
| **Base** | `base` | Few-shot prompting for pretrained models (no chat template) |
| **Instruct** | `instruct` | Zero-shot with `--apply_chat_template` for instruction-tuned models |

### Supported Languages

| Code | Language | Custom tasks |
|---|---|---|
| `eng` | English | Standard lm-eval tasks + Global-MMLU |
| `mri` | Māori | Translated benchmarks (ARC, HellaSwag, PIQA, XCOPA, XNLI, etc.) |
| `nya` | Chichewa | Translated benchmarks (same suite as Māori) |

## Repository Layout

```
byol/eval/                     # Python package (cli.py, config.py, runner.py, judge.py, etc.)
configs/eval/                  # Run configurations (benchmark_{type}_{lang}.yaml)
data/eval/                     # Translated .jsonl datasets (chichewa/, maori/)
```

## CLI Reference

### Benchmark Mode (default)

```
python -m byol.eval --model MODEL --type {base,instruct} --tgt-lang {eng,mri,nya} [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--model`, `-m` | *(required)* | HuggingFace model ID or local path |
| `--type` | *(required)* | `base` (few-shot) or `instruct` (0-shot + chat template) |
| `--tgt-lang` | *(required)* | Target language: `eng`, `mri`, or `nya` |
| `--config`, `-c` | *auto* | Path to YAML config (overrides `--type`/`--tgt-lang`) |
| `--dtype` | `bfloat16` | Model dtype: `bfloat16`, `float16`, `float32`, `auto` |
| `--tasks`, `-t` | `all` | Comma-separated task names or `all` |
| `--num-fewshot`, `-n` | *from config* | Override few-shot count for all tasks |
| `--limit` | *none* | Max samples per task (useful for debugging) |
| `--device`, `--gpus`, `-g` | `0` | Comma-separated GPU device IDs |
| `--batch-size`, `-b` | `auto:4` | Batch size or `auto:N` for auto-scaling |
| `--output-dir`, `-o` | `results/<lang>/eval` | Output directory for results |
| `--data-dir` | *none* | Path to folder of translated eval JSONL files (overrides default data paths) |
| `--overwrite` | `false` | Re-run evaluations even if results already exist |
| `--skip` | `false` | Skip tasks with existing completed outputs |
| `--log-samples` | `false` | Log individual sample predictions |
| `--dry-run` | `false` | Preview lm-eval commands without executing |
| `--verbose`, `-v` | `0` | Increase verbosity (`-v`, `-vv`) |

### Judge Mode

```
python -m byol.eval judge --model-config PATH --dataset-config PATH [--output-dir DIR]
```

| Argument | Default | Description |
|---|---|---|
| `--model-config`, `-m` | *(required)* | Path to judge model configuration YAML |
| `--dataset-config`, `-d` | *(required)* | Path to judge dataset configuration YAML |
| `--output-dir`, `-o` | `results/<lang>/eval/judge` | Output directory for judge results |

## Python API

```python
from byol.eval import EvalConfig, EvaluationRunner

config = EvalConfig.from_yaml("configs/eval/benchmark_base_mri.yaml")
config.model.path = "my-org/my-model"

runner = EvaluationRunner(config)
results = runner.run_all()
for result in results:
    print(f"{result.task_name}: {result.metrics}")
```

## Configuration

### Run Configuration

YAML configs in `configs/eval/` define complete evaluation runs. See [`configs/eval/benchmark_base_mri.yaml`](../../configs/eval/benchmark_base_mri.yaml) for a full example.

Config naming convention: `benchmark_{type}_{lang}.yaml`. The CLI auto-resolves the config from `--type` and `--tgt-lang`:

```bash
# Loads configs/eval/benchmark_base_mri.yaml
python -m byol.eval --model my/model --type base --tgt-lang mri
```

## Task Definitions (tasks/)

Task definitions follow the [lm-evaluation-harness task format](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md). Each task directory contains YAML definitions and optional Python utilities.

### Available Task Suites

| Directory | Benchmarks | Languages |
|---|---|---|
| `arc/` | ARC-Challenge, ARC-Easy | mri, nya |
| `flores/` | FLORES translation | eng↔mri, eng↔nya |
| `Global-MMLU-Lite/` | Global MMLU (lite) | mri, nya, + many others |
| `hellaswag/` | HellaSwag | mri, nya |
| `MGSM/` | Multilingual GSM (math) | mri, nya |
| `piqa/` | PIQA physical reasoning  | mri, nya |
| `realtoxicitypromptsllama/` | RealToxicityPrompts | eng |
| `truthfulqa-multi-chichewa/` | TruthfulQA-Multi | mri, nya |
| `xcopa/` | X-COPA causal reasoning | mri, nya |
| `xnli/` | XNLI natural language inference | mri, nya |
| `xstorycloze/` | XStoryCloze | mri, nya |
| `xwinograd/` | XWinograd | mri, nya |

### Task Data References

Task YAMLs reference translated datasets using paths relative to the repo root:

```yaml
# byol/eval/tasks/arc/arc_challenge_mri.yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: data/eval/maori/arc_challenge_test_english2Maori_microsoft_translated.jsonl
    train: data/eval/maori/arc_challenge_train_english2Maori_microsoft_translated.jsonl
```

## Prerequisites

```bash
# 1. Install lm-evaluation-harness (pinned commit)
cd third_party_libs/
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 69145a03d3e6ad0e552f729b7fe57b2762e7e4ca
pip install -e ".[multilingual]"
cd ../..

# 2. Apply patches
python byol/eval/patches/apply_lmeval_patches.py

# 3. Set HuggingFace token
export HF_TOKEN="your-huggingface-token"
```

## Adding a New Language

The quickest way is the scaffolding command:

```bash
python -m byol.eval add-language --lang yor --name Yoruba
```

This auto-detects data file suffixes and creates configs, task definitions, and data directories.

For manual setup:

1. **Register** the language in `byol/eval/constants.py` (`VALID_LANGS` and `LANG_NAMES`)
2. **Create configs**: `configs/eval/benchmark_base_yor.yaml` and `benchmark_instruct_yor.yaml` (copy from an existing language)
3. **Add task definitions** under `byol/eval/tasks/<benchmark>/` (YAML files referencing translated data)
4. **Place translated datasets** in `data/eval/yoruba/`
5. **Update extraction** in `byol/eval/extract_benchmarks.py` (add to `SUPPORTED_LANGUAGES`)
6. **Validate**: `python -m byol.eval --model google/gemma-3-4b-pt --type base --tgt-lang yor --dry-run`

## Testing

```bash
conda run -n byol pytest tests/test_eval/ -v
```

Tests cover config file existence, path resolution, language purity, base vs. instruct settings, dataclass validation, task registry lookup, CLI parsing, dry-run integration, and more.
