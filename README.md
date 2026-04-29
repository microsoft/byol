# BYOL: Bring Your Own Language Into LLMs

[Syed Waqas Zamir](https://scholar.google.ae/citations?hl=en&user=POoai-QAAAAJ), [Wassim Hamidouche](https://scholar.google.com/citations?user=ywBnUIAAAAAJ&hl=fr), [Boulbaba Ben Amor](https://scholar.google.com/citations?user=8kGJcCIAAAAJ&hl=en), [Luana Marotti](https://uy.linkedin.com/in/luana-marotti), [Inbal Becker-Reshef](https://scholar.google.com/citations?user=gPfbTcMAAAAJ&hl=en), and [Juan Lavista Ferres](https://www.microsoft.com/en-us/research/people/jlavista/)

[![paper](https://img.shields.io/badge/arXiv-Paper-2196F3.svg)](https://arxiv.org/abs/2601.10804)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Collection-yellow)](https://huggingface.co/collections/ai-for-good-lab/byol)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Global_MMLU--Lite-green)](https://huggingface.co/datasets/ai-for-good-lab/Global-MMLU-Lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<hr />

BYOL is a scalable framework for extending LLMs to low- and extreme-low-resource languages based on each language's digital footprint. Given a target language, BYOL assesses its digital resources, prepares training and evaluation data, performs continual pre-training and instruction tuning, and evaluates the adapted model on multilingual benchmarks.

<!-- TODO: Add an overview figure of the BYOL pipeline here -->
<!-- <p align="center"><img src="assets/byol_overview.png" width="800"/></p> -->

#### News
- **Apr 2026:** BYOL toolkit released
- **Apr 2026:** Trained models for Chichewa and Māori released on 🤗 HuggingFace ([collection](https://huggingface.co/collections/ai-for-good-lab/byol))
- **Apr 2026:** Human-translated [Global MMLU-Lite](https://huggingface.co/datasets/ai-for-good-lab/Global-MMLU-Lite) for Inuktitut, Chichewa, and Māori released on 🤗 HuggingFace

---

## Installation

See **[SETUP.md](SETUP.md)** for the full setup guide (environment, credentials, third-party libs, verification).

---

## Quick Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "ai-for-good-lab/byol-nya-4b-merged"  # Chichewa 4B
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)

messages = [{"role": "user", "content": "Tandiuzeni za dziko la Malawi."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Released Models

We release BYOL-adapted LLMs for Chichewa and Māori. **Merged** models are recommended for most users (chat/instruction-following). **CPT** models are base models for text completion. Intermediate IT checkpoints are also available in the [collection](https://huggingface.co/collections/ai-for-good-lab/byol).

| Language | 1B | 4B | 12B |
|---|---|---|---|
| **Chichewa (nya)** | [CPT](https://huggingface.co/ai-for-good-lab/byol-nya-1b-cpt) | [CPT](https://huggingface.co/ai-for-good-lab/byol-nya-4b-cpt) · [Merged](https://huggingface.co/ai-for-good-lab/byol-nya-4b-merged) | [CPT](https://huggingface.co/ai-for-good-lab/byol-nya-12b-cpt) · [Merged](https://huggingface.co/ai-for-good-lab/byol-nya-12b-merged) |
| **Māori (mri)** | [CPT](https://huggingface.co/ai-for-good-lab/byol-mri-1b-cpt) | [CPT](https://huggingface.co/ai-for-good-lab/byol-mri-4b-cpt) · [Merged](https://huggingface.co/ai-for-good-lab/byol-mri-4b-merged) | [CPT](https://huggingface.co/ai-for-good-lab/byol-mri-12b-cpt) · [Merged](https://huggingface.co/ai-for-good-lab/byol-mri-12b-merged) |


## Released Evaluation Data

We release **human-translated** [Global MMLU-Lite](https://huggingface.co/datasets/ai-for-good-lab/Global-MMLU-Lite) for Chichewa, Māori, and Inuktitut — extending the [original 18-language benchmark](https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite) with 3 new low-resource languages.

```python
from datasets import load_dataset
ds = load_dataset("ai-for-good-lab/Global-MMLU-Lite", "mri", split="test")  # nya, mri, or iku
```

---

## Step-by-Step BYOL Pipeline

The BYOL pipeline takes any language from assessment to evaluation: classify the language's digital footprint, find the best translators, prepare training and evaluation data, train, merge, and evaluate on multilingual benchmarks.

Each step can be run independently. Replace `<LANG>` with an ISO-639-3 code (e.g., `nya`, `mri`). Detailed documentation is linked within each section — refer to those for the full set of options and configurations.

<details>
<summary><b>Step 1-3: Language Resource Assessment</b></summary>

Classify the language and find the best translation models. READ [Full docs](byol/language_resource_assessment/README.md)

```bash
# Classify language resource level
python -m byol.language_resource_assessment --task language-classification --tgt-lang <LANG>

# Benchmark translators (API + local models, multi-GPU)
python -m byol.language_resource_assessment --task find-best-translator --tgt-lang <LANG> --device 0,1

# Benchmark open-weight LLMs for language adaptation
python -m byol.language_resource_assessment --task find-best-llm --tgt-lang <LANG> --device 0,1
```
</details>

<details>
<summary><b>Step 4-6: Data Preparation</b></summary>

Prepare training (CPT, SFT) and evaluation data. READ [Full docs](byol/data_prep/README.md)

```bash
# CPT: download, refine, translate bilingual training corpus
python -m byol.data_prep --stage cpt --tgt-lang <LANG>

# SFT: instruction-tuning data from SmolTalk2 + AYA
python -m byol.data_prep --stage sft --tgt-lang <LANG>

# Eval: translate 10 English benchmarks to target language
python -m byol.data_prep --stage eval --tgt-lang <LANG>
```

> Config files are auto-generated at `configs/data_prep/{stage}/<LANG>.yaml` on first run.
> Add `--max-samples 10` for quick testing. Use `--translator <translator name>` to override the default (see list of **supported Machine Translators** [here](byol/translation_backends/README.md)).
</details>

<details>
<summary><b>Step 7: Generate Eval Task Configs</b></summary>

Scaffold lm-evaluation-harness task YAMLs for the new language:

```bash
python -m byol.eval add-language --lang <LANG> --name <LanguageName>
```
</details>

<details>
<summary><b>Step 8-9: Training</b></summary>

Train with LlamaFactory. [Full docs](byol/train/README.md)

```bash
# Continual Pre-Training
python -m byol.train cpt --tgt-lang <LANG> --model google/gemma-3-4b-pt --device 0

# Supervised Fine-Tuning (on top of CPT checkpoint)
python -m byol.train sft --tgt-lang <LANG> --model <cpt_checkpoint> --device 0
```
</details>

<details>
<summary><b>Step 10: Model Merging</b></summary>

```bash
python -m byol.train.merge general \
  --model-pt google/gemma-3-4b-pt \
  --model-it google/gemma-3-4b-it \
  --model-el <sft_checkpoint> \
  --beta 0.6 --device 3 --dtype bfloat16 \
  --output results/<LANG>/train/merged/<name>
```
</details>

<details>
<summary><b>Step 11-13: Evaluation</b></summary>

Evaluate on multilingual benchmarks using lm-evaluation-harness. [Full docs](byol/eval/README.md)

```bash
# Base model (few-shot)
python -m byol.eval --model <cpt_checkpoint> --type base --tgt-lang <LANG> --device 0

# Merged model (0-shot + chat template)
python -m byol.eval --model <merged_checkpoint> --type merged --tgt-lang <LANG> --device 0

# Extract result tables
python -m byol.eval.extract_benchmarks results/<LANG>/eval/base/<model>/
```
</details>

---

## Full Pipeline -- One Command

Run the complete pipeline (classify - assess - data prep - train - eval) for any language:

```bash
# Full pipeline for Chichewa (uses configs/train/*.yaml settings)
python -m byol.pipeline run-all --tgt-lang nya --device 0 \
  --model google/gemma-3-4b-pt --instruct-model google/gemma-3-4b-it

# Quick test (10 samples, reduced training, ~30 min)
python -m byol.pipeline run-all --tgt-lang nya --device 0 \
  --model google/gemma-3-4b-pt --instruct-model google/gemma-3-4b-it \
  --max-samples 10 --quick-test

# Check progress
python -m byol.pipeline status --tgt-lang nya

# Clean all artifacts for a language (data, results, configs, eval tasks)
python -m byol.pipeline clean --tgt-lang nya
```

This runs all 13 steps automatically, skipping what is already done. 

---


## Translation API

Use any of 30+ translation models with a unified interface. [Full docs](byol/translation_backends/README.md)

```python
from byol import translate, list_models

list_models()  # show all available models
translate("Hello, how are you?", tgt_lang="Spanish", model="gpt-5-chat")
translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", model="nllb-200-3.3b")
```

---

## Project Structure

```
byol/
  byol/
    common/                          # Shared utilities
    translation_backends/            # Unified translation API (30+ models)
    language_resource_assessment/    # Language classification and model benchmarking
    data_prep/                       # Data preparation (CPT, SFT, Eval)
    train/                           # Training via LlamaFactory
    eval/                            # Evaluation via lm-evaluation-harness
    pipeline/                        # End-to-end orchestrator
  configs/                           # YAML configs (auto-generated per language)
  data/                              # Benchmarks, eval data, reference CSVs
  tests/                             # Unit and integration tests
```

## Citation

If you use BYOL, please consider citing:

```bibtex
@article{zamir2026byolbringlanguagellms,
    title={BYOL: Bring Your Own Language Into LLMs}, 
    author={Syed Waqas Zamir and Wassim Hamidouche and Boulbaba Ben Amor and Luana Marotti and Inbal Becker-Reshef and Juan Lavista Ferres},
    year={2026},
    journal={arXiv:2601.10804},
    url={https://arxiv.org/abs/2601.10804}, 
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
