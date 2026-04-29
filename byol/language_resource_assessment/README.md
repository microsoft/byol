# Language Resource Assessment

Assess a target language's digital presence and find the best models for working with it.

## Overview

This module answers three key questions about a target language:

| Task | Question | Output |
|------|----------|--------|
| `language-classification` | What is the language's resource level? | Classification + recommendations |
| `find-best-translator` | Which translator is best for generating synthetic data? | Ranked model comparison |
| `find-best-llm` | Which open-weight LLM is best to adapt? | Ranked model comparison |

## Tasks

### 1. Language Classification

Classify a language by its digital resource level and get practical recommendations.

```bash
python -m byol.language_resource_assessment \
    --task language-classification \
    --tgt-lang nya
```

**Output:**

```
============================================================
  Language Name:      Nyanja
  ISO-3 Code:         nya
  Classification:     Low-Resource
  Number of Speakers: 14,507,700
  Corpus Size:        63,283,501 words
  Language Family:    Niger-Congo
  Script Type:        Latin
  Category:           Major Regional
============================================================

  ℹ️   Limited but usable data, candidate for continual pretraining
```

**Classifications:**

| Level | Description |
|-------|-------------|
| `High-Resource` | Abundant web-scale corpora, comprehensive LLM support |
| `Medium-Resource` | Substantial resources, light adaptation can close performance gaps |
| `Low-Resource` | Limited but usable data, candidate for continual pretraining |
| `Extreme-Low-Resource` | Minimal digital presence, MT-based access is the most practical route |

---

### 2. Find Best Translator

Compare translation-focused models to find the best one for your target language. Uses round-trip translation evaluation.

```bash
# Basic usage - uses default translators from config
python -m byol.language_resource_assessment \
    --task find-best-translator \
    --tgt-lang nya

# Language names and aliases also work (Chichewa → nya, Guarani → gug)
python -m byol.language_resource_assessment \
    --task find-best-translator \
    --tgt-lang Chichewa

# Specify which translators to compare
python -m byol.language_resource_assessment \
    --task find-best-translator \
    --tgt-lang nya \
    --translators gpt-5-chat,nllb-200-3.3b,microsoft-translator

# Run only the translation step (skip metrics)
python -m byol.language_resource_assessment \
    --task find-best-translator \
    --tgt-lang mri \
    --step translate

# Multi-GPU: local models run in parallel across GPUs
python -m byol.language_resource_assessment \
    --task find-best-translator \
    --tgt-lang nya \
    --output-dir ./results/my_experiment \
    --device 2,3
```

**Available Translators:**

| Provider | Models |
|----------|--------|
| Microsoft | `microsoft-translator` |
| Google | `google-translator` |
| DeepSeek | `deepseek-r1`, `deepseek-r1-0528` |
| OpenAI/GPT | `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat` |
| Meta NLLB | `nllb-200-600m`, `nllb-200-1.3b`, `nllb-200-3.3b` |
| Meta Seamless | `seamless-m4t-medium`, `seamless-m4t-large` |
| Google MADLAD | `madlad-400-3b`, `madlad-400-7b` |
| Google TranslateGemma | `translategemma` |
| Helsinki | `marian` |
| Cohere | `aya-101` |

---

### 3. Find Best LLM

Compare open-weight LLMs to find the best one to adapt for your target language.

```bash
# Basic usage
python -m byol.language_resource_assessment \
    --task find-best-llm \
    --tgt-lang nya \
    --llms qwen3-8b,gemma-3-12b-it,apertus-8b

# Multi-GPU parallel evaluation
python -m byol.language_resource_assessment \
    --task find-best-llm \
    --tgt-lang swh \
    --llms qwen3-4b,qwen3-8b,gemma-3-4b-it \
    --device 2,3 \
    --max-samples 100
```

---

## CLI Reference

```bash
python -m byol.language_resource_assessment [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--task` | Task to run: `language-classification`, `find-best-translator`, `find-best-llm` |
| `--tgt-lang` | Target language — ISO-3 code (`nya`), full name (`Chichewa`), or alias (`Guarani` → `gug`) |
| `--translators` | Comma-separated list of translators (for `find-best-translator`) |
| `--llms` | Comma-separated list of LLMs (for `find-best-llm`) |
| `--step` | Run specific step: `translate`, `metrics`, or both (default) |
| `--dataset` | Path to custom dataset (JSONL format) |
| `--output-dir` | Custom output directory |
| `--device` | GPU device(s): single (`0`), comma-separated for parallel local models (`2,3`), or `cpu` |
| `--max-samples` | Limit number of samples to evaluate |

**Execution model:**

- **API models** (GPT, Microsoft, Google, DeepSeek) and **local models** (NLLB, Seamless, etc.) run **concurrently** — API requests are dispatched in parallel while local models are evaluated on GPU.
- With comma-separated `--device` (e.g., `--device 2,3`), local models are distributed across GPUs for parallel evaluation.
- **Ctrl+C** cleanly terminates all child processes via a signal handler.

**Language aliases:** `--tgt-lang` accepts ISO-3 codes (`nya`), full names (`Chichewa`), or common aliases (`Guarani` → `gug`). Aliases are resolved via `language_codes.yaml`.

---

## Output Structure

Results are organized by task type:

```
results/<tgt_lang>/lra/
├── translator_comparison/           # --task find-best-translator
│   ├── translations_*.jsonl
│   ├── translations_*_with_metrics_openai.jsonl
│   └── ranking_*.png
└── llm_comparison/                  # --task find-best-llm
    ├── translations_*.jsonl
    ├── translations_*_with_metrics_openai.jsonl
    └── ranking_*.png
```

---

## Configuration

### Config Files

Located in `configs/language_resource_assessment/`:

**Add a new translator** (`translators.yaml`):

```yaml
translators:
  my-new-translator:
    type: api  # or "local"
    factory_name: azure-openai
    src_lang_format: full_name
    tgt_lang_format: full_name
    params:
      model_name: my-model
      temperature: 0.0
      max_tokens: 1024
```

**Add a new LLM** (`llms.yaml`):

```yaml
translators:
  my-new-llm:
    type: local
    factory_name: qwen3  # or gemma3, causal_lm, etc.
    src_lang_format: full_name
    tgt_lang_format: full_name
    params:
      model_name: organization/model-name
      max_new_tokens: 256
```

**Add a new language**:

```yaml
language_codes:
  MyLanguage:
    full_name: MyLanguage
    iso2: ml
    iso3: myl
    flores200: myl_Latn
    aliases:
      - Alternative Name
```

---

## Architecture

```
language_resource_assessment/
├── __main__.py              # Entry point
├── cli.py                   # CLI argument parsing
├── config.py                # Configuration management
├── find_best_model.py       # Model evaluation (translators & LLMs)
├── language_digital_presence.py  # Language classification
├── metrics.py               # Evaluation metrics
├── io.py                    # Data loading/saving
└── visualize.py             # Result visualization
```
