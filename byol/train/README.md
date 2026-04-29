# BYOL Training Framework

Train LLMs using [LlamaFactory](https://github.com/hiyouga/LlamaFactory). Supports Continual Pre-Training (CPT), Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and LoRA adapter merging.

## Prerequisites

```bash
# 1. Install LlamaFactory (pinned)
cd third_party_libs/
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory && pip install -e ".[torch,metrics]" && cd ../..

# 2. Install BYOL
pip install -e .
```

### DeepSpeed (Optional)

```bash
conda install -y -c nvidia cuda-toolkit=12.8
export CUDA_HOME=$CONDA_PREFIX
pip install deepspeed
```

### Secrets

```bash
export HF_TOKEN="your-huggingface-token"
# Or create byol/train/secrets_local.py:
#   HF_TOKEN = "your-token"
```

## Data Setup

### Where data lives

LlamaFactory uses a **two-layer data system**. Training data is stored per-language under `~/byol-data/<lang>/`:

```
~/byol-data/<lang>/cpt/bilingual_mix/    ← dataset_dir for CPT
├── dataset_info.json                     ← registry: maps dataset names → files + column mappings
└── cpt/
    └── {name}_english_cpt.jsonl

~/byol-data/<lang>/sft/bilingual_mix/    ← dataset_dir for SFT
├── dataset_info.json
└── sft/
    ├── {name}_english_sft.jsonl
    └── {name}_sft_test.jsonl
```

By default, `dataset_dir` is resolved automatically when using `--tgt-lang`. File paths in `dataset_info.json` are relative to this directory.

### Using an external data directory

To override the default data location, set `dataset_dir` in your YAML config:

```yaml
dataset_dir: ~/byol-data/nya/cpt/bilingual_mix
```

The directory must contain:
- A `dataset_info.json` (the dataset registry)
- The JSONL files referenced by the registry

### Data formats

| Stage | Format | Example |
|---|---|---|
| CPT | `{"text": "..."}` | One document per line |
| SFT | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` | OpenAI chat format |
| DPO | `{"instruction": "...", "chosen": "...", "rejected": "..."}` | Preference pairs |

Each format requires a corresponding entry in `dataset_info.json` within the `dataset_dir`. See existing entries for examples.

## Quick Start

The **recommended** way is to use `--tgt-lang`, which auto-resolves `--dataset`, `--eval-dataset`, and `dataset_dir`:

```bash
# CPT — just specify language and model
python -m byol.train cpt \
  --model google/gemma-3-4b-pt \
  --tgt-lang gug \
  --device 3

# SFT — same simplicity
python -m byol.train sft \
  --model google/gemma-3-4b-pt \
  --tgt-lang nya \
  --device 0,1,2,3 \
  --epochs 3 --batch-size 2 --grad-accum 32 --lr 1e-5

# SFT with LoRA
python -m byol.train sft \
  --model google/gemma-3-4b-it \
  --tgt-lang nya \
  --lora --lora-rank 64 --device 0,1,2,3
```

Other useful commands:

```bash
# Dry run (preview LlamaFactory config without training)
python -m byol.train sft --model google/gemma-3-4b-pt --dry-run

# Merge LoRA adapter
python -m byol.train merge \
  --base-model google/gemma-3-4b-pt \
  --adapter results/nya/train/sft/checkpoint \
  --output results/nya/train/merged
```

## CLI Reference

```
python -m byol.train {cpt,sft,dpo,merge} [options]
```

| Flag | Description | Default |
|---|---|---|
| `--model`, `-m` | HuggingFace model ID or local path | *required* |
| `--config`, `-c` | YAML config file path | auto-detect |
| `--dataset`, `-d` | Training dataset name (must match `dataset_info.json`) | — |
| `--eval-dataset` | Evaluation dataset name | same as `--dataset` |
| `--device`, `--gpus`, `-g` | Comma-separated GPU IDs | `0` |
| `--epochs`, `-e` | Training epochs | `3` |
| `--batch-size`, `-b` | Per-device batch size | `4` |
| `--grad-accum` | Gradient accumulation steps | `4` |
| `--lr` | Learning rate | `5e-5` |
| `--cutoff-len` | Maximum sequence length | `8192` |
| `--template` | Chat template | `gemma` |
| `--lora` | Enable LoRA fine-tuning | off |
| `--lora-rank` | LoRA rank | `16` |
| `--lora-alpha` | LoRA alpha | `32` |
| `--output-dir`, `-o` | Output directory | `results/train` |
| `--tgt-lang` | Target language ISO-3 code — auto-resolves `--dataset`, `--eval-dataset`, and `dataset_dir` (e.g., `--tgt-lang gug` sets dataset to `gug_english_cpt`, dataset_dir to `~/byol-data/gug/cpt/bilingual_mix`) | — |
| `--wandb-project` | W&B project for logging | — |
| `--dry-run` | Preview config without running | off |
| `--override` | Override config values (`key=value`) | — |

## Python API

```python
from byol.train import TrainConfig, TrainingRunner

config = TrainConfig.from_yaml("configs/train/sft.yaml")
runner = TrainingRunner(config)
result = runner.run()
```

For LoRA merging and delta merging, see `byol.train.merge`:

```python
from byol.train.merge import merge_lora_simple

result = merge_lora_simple(
    base_model="google/gemma-3-4b-pt",
    lora_path="results/nya/train/sft/checkpoint",
    output_dir="results/nya/train/merged",
)
```

## Configuration

YAML files in `configs/train/` are passed through to LlamaFactory. See [`configs/train/sft.yaml`](../../configs/train/sft.yaml) for a full example. Any field not handled by `TrainConfig` is forwarded as-is to LlamaFactory (`passthrough_args`).

**Config resolution order:** CLI args override YAML values. If no `--config` is given, the default config for the stage (`configs/train/{cpt,sft,dpo}.yaml`) is loaded automatically.

**`dataset_dir`:** Resolved automatically to `~/byol-data/<lang>/{cpt,sft}/bilingual_mix` when `--tgt-lang` is provided. Override with `--override dataset_dir=<path>` if needed.

## How to Add a New Language

1. **Prepare data** — create JSONL files matching the formats above (CPT: `{"text": "..."}`, SFT: `{"messages": [...]}`)
2. **Place files** — put them under `cpt/` or `sft/` inside your `dataset_dir` (default: `~/byol-data/<lang>/{cpt,sft}/bilingual_mix/`)
3. **Register datasets** — add entries to `dataset_info.json` inside the `dataset_dir` following existing patterns
4. **Create configs** (optional) — copy an existing YAML from `configs/train/` and update `dataset`, `dataset_dir`, and model fields
5. **Train** — `python -m byol.train sft --model <model> --tgt-lang <lang>`

## Troubleshooting

| Issue | Solution |
|---|---|
| `token_type_ids required` | Ensure `transformers>=4.56.0` (included in `environment.yml`) |
| `CUDA_HOME does not exist` | `export CUDA_HOME=$CONDA_PREFIX` |
| `OOM error` | Reduce `--batch-size`, increase `--grad-accum` |
| `LlamaFactory CLI not found` | Install LlamaFactory: `cd third_party_libs/LlamaFactory && pip install -e ".[torch,metrics]"` |
| Dataset not found | Ensure the name in `--dataset` matches an entry in the `dataset_info.json` within your `dataset_dir` (e.g., `~/byol-data/<lang>/sft/bilingual_mix/dataset_info.json`) and the JSONL file exists at the referenced path |
