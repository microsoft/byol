# Data Attribution and Licensing

This document lists all data files bundled with or downloaded by the BYOL toolkit,
their original sources, licenses, and proper citations.

For the full details on how these datasets are used in the BYOL framework, see:

> Zamir, S.W., Hamidouche, W., Ben Amor, B., Marotti, L., Becker-Reshef, I., & Lavista Ferres, J. (2026).
> *BYOL: Bring Your Own Language Into LLMs*. arXiv:2601.10804.
> https://arxiv.org/abs/2601.10804

---

## Bundled Datasets (shipped with the repository)

### 1. COPA — Choice of Plausible Alternatives

| | |
|---|---|
| **Files** | `data/eval/sources/copa_test_fixed.jsonl` (500 samples), `data/eval/sources/copa_validation_fixed.jsonl` (100 samples) |
| **Description** | English COPA benchmark for commonsense causal reasoning. Used as the source for translating XCOPA evaluations into target languages. |
| **Original source** | https://asgordon.github.io/copa.html |
| **HuggingFace** | Part of SuperGLUE (`super_glue`, `copa` config) |
| **License** | Released for research purposes by the original authors. BSD-2-Clause (as distributed via SuperGLUE). |
| **Citation** | Roemmele, M., Bejan, C.A., & Gordon, A.S. (2011). *Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning.* AAAI Spring Symposium. |

### 2. XWinograd (English subset, aligned)

| | |
|---|---|
| **Files** | `data/eval/sources/xwinograd_aligned_english_1000.jsonl` (935 samples) |
| **Description** | English Winograd-schema sentences used as the source for translating XWinograd evaluations into target languages. |
| **Original source** | https://huggingface.co/datasets/Muennighoff/xwinograd |
| **License** | CC BY 4.0 (Creative Commons Attribution 4.0 International) |
| **Citation** | Tikhonov, A. & Ryabinin, M. (2021). *It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning.* arXiv:2106.12066. |

### 3. RTTBench-Mono

| | |
|---|---|
| **Files** | `data/rttbench_mono_dataset/RTTBench-Mono.jsonl` (1,250 samples) |
| **Description** | Monolingual English benchmark for evaluating round-trip translation quality across 25 domains. Used by the Language Resource Assessment module to benchmark translation models. |
| **Original source** | Created as part of the BYOL project (Microsoft). |
| **License** | MIT (same as this repository) |
| **Citation** | Zamir et al. (2026). *BYOL: Bring Your Own Language Into LLMs.* arXiv:2601.10804. |

### 4. Language Digital Presence

| | |
|---|---|
| **Files** | `data/language_digital_presence.csv` (1,503 languages) |
| **Description** | Language metadata table with corpus sizes, speaker counts, language families, script types, and resource-level cluster assignments. Used by the Language Resource Assessment module to classify a language's digital footprint. |
| **Original source** | Compiled by the BYOL project (Microsoft) from publicly available linguistic resources. |
| **License** | MIT (same as this repository) |
| **Citation** | Zamir et al. (2026). *BYOL: Bring Your Own Language Into LLMs.* arXiv:2601.10804. |

### 5. Translator Language Support

| | |
|---|---|
| **Files** | `data/translator_language_support.csv` (1,503 languages) |
| **Description** | Support matrix indicating which translation backends (Microsoft Translator, Google Translator, NLLB, SeamlessM4T, MADLAD-400, TranslateGemma, GPT, DeepSeek) support each language. |
| **Original source** | Compiled by the BYOL project (Microsoft) from each translator's official documentation. |
| **License** | MIT (same as this repository) |
| **Citation** | Zamir et al. (2026). *BYOL: Bring Your Own Language Into LLMs.* arXiv:2601.10804. |

---

## Downloaded Datasets (fetched at runtime by `byol.data_prep`)

These datasets are **not** bundled with the repository. They are downloaded from
HuggingFace Hub when the user runs the data preparation pipeline. Each dataset
retains its original license.

| Benchmark | HuggingFace Dataset ID | Config | License |
|---|---|---|---|
| ARC (Challenge) | `allenai/ai2_arc` | `ARC-Challenge` | CC BY-SA 4.0 |
| ARC (Easy) | `allenai/ai2_arc` | `ARC-Easy` | CC BY-SA 4.0 |
| HellaSwag | `Rowan/hellaswag` | — | MIT |
| PIQA | `baber/piqa` | — | AFL-3.0 |
| XNLI | `facebook/xnli` | `en` | CC BY-NC 4.0 |
| XStoryCloze | `juletxara/xstory_cloze` | `en` | CC BY-SA 4.0 |
| MGSM | `juletxara/mgsm` | `en` | CC BY-SA 4.0 |
| TruthfulQA-Multi | `HiTZ/truthfulqa-multi` | `en` | Apache 2.0 |
| Global MMLU-Lite | `CohereForAI/Global-MMLU-Lite` | `en` | Apache 2.0 |
| FineWeb-2 | `HuggingFaceFW/fineweb-2` | varies | ODC-By 1.0 |
| SmolTalk2 | `HuggingFaceTB/smoltalk` | varies | Apache 2.0 |
| AYA | `CohereForAI/aya_dataset` | varies | Apache 2.0 |

> **Note:** License information above reflects what was documented at the time of
> writing. Always verify the current license on each dataset's HuggingFace page
> before redistribution.
