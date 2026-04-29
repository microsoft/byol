# BYOL Data Preparation

Automated data preparation pipelines for continual pretraining (CPT), supervised finetuning (SFT), and evaluation datasets.

## Quick Start

### CPT

Create a YAML config for your target language (see [`configs/data_prep/cpt/nya.yaml`](../../configs/data_prep/cpt/nya.yaml) for a full example), then run:

```bash
python -m byol.data_prep --stage cpt --config configs/data_prep/cpt/nya.yaml
```

Or simply use `--tgt-lang` — a config file is auto-generated on first run:

```bash
python -m byol.data_prep --stage cpt --tgt-lang nya
```

For quick testing, limit each step to a small number of samples:

```bash
python -m byol.data_prep --stage cpt \
    --config configs/data_prep/cpt/nya.yaml \
    --max-samples 10
```

### SFT

Create a YAML config (see [`configs/data_prep/sft/nya.yaml`](../../configs/data_prep/sft/nya.yaml) for a full example), then run:

```bash
python -m byol.data_prep --stage sft --config configs/data_prep/sft/nya.yaml
```

Or use only CLI flags (no config file):

```bash
python -m byol.data_prep --stage sft --tgt-lang nya
```

For quick testing, limit each step to a small number of samples:

```bash
python -m byol.data_prep --stage sft \
    --config configs/data_prep/sft/nya.yaml \
    --max-samples 10
```

### Eval Data Prep

Translate English evaluation benchmarks into the target language. This uses
the `byol.translation_backends` package (Microsoft Translator, Google, GPT-5, etc.)
to translate each benchmark's fields.

```bash
# Translate all benchmarks for Chichewa
python -m byol.data_prep --stage eval --tgt-lang nya

# With config file
python -m byol.data_prep --stage eval --config configs/data_prep/eval/nya.yaml

# Quick test: only 10 samples per split
python -m byol.data_prep --stage eval --tgt-lang nya --max-samples 10

# Translate specific benchmarks only
python -m byol.data_prep --stage eval --tgt-lang nya --benchmarks copa mmlu_lite

# Override translator for all benchmarks
python -m byol.data_prep --stage eval --tgt-lang nya --translator gpt-5

```

Common flags: `--dry-run` (preview plan), `--output-dir PATH` (override output), `--device 3` or `--device 2,3` (GPU for local models).

**Supported benchmarks** (10 total):

| Benchmark | Source | Default Translator |
|---|---|---|
| `copa` | Local files (shipped with repo) | microsoft-translator |
| `ai2_arc_hard` | HF `allenai/ai2_arc` ARC-Challenge | microsoft-translator |
| `ai2_arc_easy` | HF `allenai/ai2_arc` ARC-Easy | microsoft-translator |
| `hellaswag` | HF `Rowan/hellaswag` | microsoft-translator |
| `piqa` | HF `baber/piqa` | microsoft-translator |
| `xnli` | HF `facebook/xnli` (en) | microsoft-translator |
| `xstory_cloze` | HF `juletxara/xstory_cloze` (en) | microsoft-translator |
| `mgsm` | HF `juletxara/mgsm` (en) | microsoft-translator |
| `HiTZ-truthfulqa-multi` | HF `HiTZ/truthfulqa-multi` (en) | microsoft-translator |
| `mmlu_lite` | HF `CohereForAI/Global-MMLU-Lite` (en) | gpt-5 |

Output files are written to `~/byol-data/<lang>/eval/` by default, with names like:
`copa_test_english2chichewa_microsoft_translated.jsonl`

## Auto-Generated Config Files

When you run with `--tgt-lang` (without `--config`), a YAML config file is
automatically generated at `configs/data_prep/{stage}/{lang}.yaml` capturing
the effective settings used. CLI overrides (e.g., `--translator`, `--output-dir`)
are persisted to this file.

This means:
- **First run:** config created with defaults → `configs/data_prep/cpt/gug.yaml`
- **Subsequent runs with overrides:** config updated → translator changed from `microsoft-translator` to `gpt-5-chat`
- **Runs with `--config`:** no auto-generation (user manages the file)

## Translator Language Support

Before translating, the eval pipeline validates that the chosen translator
supports the target language. If not, it fails fast with a clear error and
suggests alternatives:

```
ERROR: copa → microsoft-translator  (try: NLLB, MADLAD-400, GPT, DeepSeek)
```

Language support data is maintained in `data/translator_language_support.csv`
(1503 languages × 8 translator families). Unknown languages are allowed through
(permissive behavior).

## Configuration

YAML configs control every aspect of the pipeline. See the example configs for full details:

- **CPT:** [`configs/data_prep/cpt/nya.yaml`](../../configs/data_prep/cpt/nya.yaml)
- **SFT:** [`configs/data_prep/sft/nya.yaml`](../../configs/data_prep/sft/nya.yaml)
- **Eval:** [`configs/data_prep/eval/nya.yaml`](../../configs/data_prep/eval/nya.yaml)

Set any step to `false` to skip it. Steps that have already produced output are skipped by default — pass `--overwrite` to re-run them.

### Adding Extra Data Sources

Both CPT and SFT pipelines support injecting additional JSONL files into the
bilingual mix **without modifying any Python code**. Add an `extra_sources`
list to your YAML config:

```yaml
# CPT example — each row must have a "text" field
extra_sources:
  - path: /data/my_monolingual_corpus.jsonl
    dataset_tag: my_corpus
    format: text          # default; one "text" field per row

  - path: /data/additional_translations.jsonl
    dataset_tag: extra_trans
    format: text
```

```yaml
# SFT example — rows with "messages" (sharegpt) or "inputs"/"targets" (aya)
extra_sources:
  - path: /data/my_chat_data.jsonl
    dataset_tag: custom_chat
    format: sharegpt      # rows have a "messages" list

  - path: /data/my_qa_pairs.jsonl
    dataset_tag: custom_qa
    format: aya           # rows have "inputs" and "targets" fields
```

Each extra source entry requires:

| Field         | Type   | Description                                              |
|---------------|--------|----------------------------------------------------------|
| `path`        | string | Absolute or relative path to a JSONL file.               |
| `dataset_tag` | string | Short identifier used to tag rows in the mix.            |
| `format`      | string | `"text"` (CPT), `"sharegpt"` or `"aya"` (SFT).          |

Extra sources are read, tagged with `dataset_tag`, and shuffled together with
the core pipeline data during the bilingual mix step. The `max_samples` cap
applies per source (including extras).

## CPT Pipeline

The pipeline produces three data components for continual pretraining:

1. **Real target-language text** — downloaded from FineWeb-2, optionally refined via LLM
2. **Real English text** — a token-matched subset of FineWeb-Edu, optionally refined via LLM
3. **Synthetic target-language text** — translated from the (refined) English subset

| # | Step | Config flag | Description |
|---|------|-------------|-------------|
| 1 | Download FineWeb-2 | `download_tgt_lang_fineweb2` | Download target-language web text from HuggingFace |
| 2 | Refine target lang | `refine_tgt_lang` | Refine/clean target-language text via LLM |
| 3 | Download FineWeb-Edu + extract subset | `download_eng_finewebedu` | Download English parquets and extract a subset matching the target-language token count (tiktoken `o200k_base`) |
| 4 | Refine English | `refine_eng` | Clean/enhance English text via LLM |
| 5 | Translate | `translate_eng_to_tgt_lang` | Translate refined English → target language via LLM |
| 6 | Bilingual mix | *(always runs)* | Concatenate all 3 sources, shuffle, and write training-ready JSONL + `dataset_info.json` |

Steps that have already produced output are skipped by default. Pass `--overwrite` (or set `overwrite: true` in YAML) to re-run them.

### Output Structure

All output is written under `<output_dir>/` (default `~/byol-data/<lang>/cpt/`, configurable in YAML or via `--output-dir`):

```
~/byol-data/<lang>/cpt/
├── {lang}_fineweb2_raw/                     # Step 1
│   └── {lang}_train.jsonl
├── {lang}_fineweb2_refined/                 # Step 2
│   └── {lang}_train_refined.jsonl
├── eng_fineweb_edu_raw/                     # Step 3
│   ├── sample/10BT/*.parquet                #   downloaded shards
│   └── eng_train.jsonl                      #   extracted subset
├── eng_fineweb_edu_refined/                 # Step 4
│   └── eng_train_refined.jsonl
├── fineweb_edu_translated_to_{lang}/        # Step 5
│   └── eng2{lang}_translated.jsonl
└── bilingual_mix/                           # Step 6 — training-ready
    ├── dataset_info.json                    #   LlamaFactory dataset registry
    └── cpt/
        └── {lang}_english_cpt.jsonl         #   shuffled mix of all 3 sources
```

Where `{lang}` is the ISO 639-3 code (e.g. `nya`). Filenames use the ISO code (e.g. `nya_english_cpt.jsonl`).

The `bilingual_mix/` directory is directly usable as `dataset_dir` in `byol.train` (use `--tgt-lang` to auto-resolve paths).

## SFT Pipeline

The pipeline produces instruction-tuning data from two sources:

1. **SmolTalk2** — 10 conversation subsets from HuggingFace (English + translated to target language)
2. **AYA dataset** — multilingual instruction data from CohereForAI (native + translated entries)

| # | Step | Config flag | Description |
|---|------|-------------|-------------|
| 1 | Download SmolTalk2 | `download_smoltalk2` | Download 10 SmolTalk2 subsets from HuggingFace, sample, and combine into a single JSONL |
| 2 | Translate SmolTalk2 | `translate_smoltalk2` | Translate English conversations → target language via LLM (preserves multi-turn structure) |
| 3 | Download AYA | `download_aya` | Download CohereForAI/aya_dataset, filter for target + source languages, split train/test |
| 4 | Translate AYA | `translate_aya` | Translate AYA inputs/targets → target language via LLM (native target-lang entries are kept as-is) |
| 5 | SFT bilingual mix | *(always runs)* | Combine all sources into shuffled sharegpt JSONL + test set + `dataset_info.json` |

Steps that have already produced output are skipped by default. Pass `--overwrite` (or set `overwrite: true` in YAML) to re-run them.

### Data Sources

**SmolTalk2 subsets** (10 configs, ~243k conversations total):
- `everyday-conversations`, `smol-contraints`, `smol-rewrite`, `smol-summarize` (from `HuggingFaceTB/smoltalk`)
- `smol-magpie-ultra`, `smol-reasoning`, `self-oss-instruct` (from `HuggingFaceTB/smoltalk`)
- `smoltalk2-gpt-5-mini`, `smoltalk2-llama-3.1-8b-instruct`, `smoltalk2-Qwen2.5-7B-Instruct` (from `HuggingFaceTB/smoltalk2`)

**AYA dataset**: Filtered for the target language plus source languages (default: `eng`, `fra`, `deu`, `spa`, `ita`).

### Output Structure

All output is written under `<output_dir>/` (default `~/byol-data/<lang>/sft/`, configurable in YAML or via `--output-dir`):

```
~/byol-data/<lang>/sft/
├── {lang}/
│   ├── smoltalk2_english/                                       # Step 1
│   │   └── smoltalk_combined.jsonl
│   ├── smoltalk2_translated_to_{lang}/                          # Step 2
│   │   └── smoltalk2_subset_translated_to_{name}_using_gpt5.jsonl
│   ├── aya_dataset/                                             # Step 3
│   │   ├── aya_filtered_train.jsonl
│   │   └── aya_filtered_test.jsonl
│   └── aya_translated_to_{lang}/                                # Step 4
│       ├── aya_dataset_translated_to_{name}_train.jsonl
│       └── aya_dataset_translated_to_{name}_test.jsonl
└── bilingual_mix/                                               # Step 5
    ├── dataset_info.json
    └── sft/
        ├── {lang}_english_sft.jsonl                             #   train
        └── {lang}_sft_test.jsonl                                #   test
```

Where `{lang}` is the ISO 639-3 code (e.g. `nya`). Filenames use the ISO code (e.g. `nya_english_sft.jsonl`).

The `bilingual_mix/` directory is directly usable as `dataset_dir` in `byol.train` (use `--tgt-lang` to auto-resolve paths).

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | (required) | `cpt`, `sft`, or `eval` |
| `--tgt-lang` | — | ISO 639-3 code (required without `--config`) |
| `--config` | — | Path to YAML config file |
| `--output-dir` | `~/byol-data/<lang>/{stage}` | Output base directory (overrides YAML `output_dir`) |
| `--max-samples` | — | Cap each step to N samples (for quick testing) |
| `--overwrite` | `false` | Re-run steps even if output exists |
| `--dry-run` | `false` | Preview plan without executing |
| `--verbose` | `false` | Enable DEBUG logging |
| `--no-download-tgt-lang-fineweb2` | enabled | Skip FineWeb-2 download |
| `--no-refine-tgt-lang` | enabled | Skip target-language refinement |
| `--no-download-eng-finewebedu` | enabled | Skip FineWeb-Edu download + extract |
| `--no-refine-eng` | enabled | Skip English refinement |
| `--no-translate` | enabled | Skip English → target translation (CPT) |
| `--translator` / `--eval-translator` | `microsoft-translator` | Override translator for all eval benchmarks (`--translator` is the preferred form) |
| `--eval-max-workers` | `8` | Number of parallel translation workers (eval) |
| `--device` | — | GPU device(s) for local models (e.g., `3` or `2,3`) |
| `--no-download-smoltalk2` | enabled | Skip SmolTalk2 download (SFT) |
| `--no-translate-smoltalk2` | enabled | Skip SmolTalk2 translation (SFT) |
| `--no-download-aya` | enabled | Skip AYA dataset download (SFT) |
| `--no-translate-aya` | enabled | Skip AYA dataset translation (SFT) |

## Python API

```python
from byol.data_prep import CPTDataPrepConfig, CPTDataPrepRunner

config = CPTDataPrepConfig.from_yaml("configs/data_prep/cpt/nya.yaml")
runner = CPTDataPrepRunner(config)
result = runner.run()
```

```python
from byol.data_prep import SFTDataPrepConfig, SFTDataPrepRunner

config = SFTDataPrepConfig.from_yaml("configs/data_prep/sft/nya.yaml")
runner = SFTDataPrepRunner(config)
result = runner.run()
```

Extra sources can be added programmatically via `ExtraSource`:

```python
from byol.data_prep import CPTDataPrepConfig, CPTDataPrepRunner, ExtraSource

config = CPTDataPrepConfig(
    tgt_lang_code="nya",
    extra_sources=[
        ExtraSource(path="/data/my_corpus.jsonl", dataset_tag="my_corpus", format="text"),
    ],
)
CPTDataPrepRunner(config).run()
```

## Prerequisites

- Azure OpenAI endpoint configured (`AZURE_OPENAI_ENDPOINT` env var)
- Azure Entra (AAD) authentication for GPT-5 models
- HuggingFace access for dataset downloads
- `tiktoken` (for automatic token counting)
- `pip install -e ".[dev]"` for development

## Repository Layout

```
byol/data_prep/
├── __init__.py                    # Package exports
├── cli.py                         # CLI entry point
├── config.py                      # CPTDataPrepConfig, SFTDataPrepConfig, ExtraSource
├── constants.py                   # Default values, dataset tags
├── cpt_data_prep_runner.py        # CPTDataPrepRunner
├── sft_data_prep_runner.py        # SFTDataPrepRunner
├── eval_data_prep_runner.py       # EvalDataPrepRunner
├── prompts.py                     # LLM prompt templates
└── steps/
    ├── cpt_bilingual_mix.py       # CPT bilingual mix
    ├── sft_bilingual_mix.py       # SFT bilingual mix
    ├── download_tgt_lang_fineweb2.py
    ├── refine.py
    ├── extract_subset.py
    ├── translate.py
    ├── download_smoltalk2.py
    ├── translate_smoltalk2.py
    ├── download_aya.py
    ├── translate_aya.py
    └── eval_translate_benchmark.py
configs/data_prep/                 # YAML configurations
tests/test_data_prep/              # Tests
```
