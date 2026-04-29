# Translation Backends

Unified interface for multiple translation services - API-based and local models.

## Quick Start

```python
from byol.translation_backends import translate, translate_batch, list_models, get_supported_models

# See available models
list_models()

# Translate with any model by name
translate("Hello world", tgt_lang="French", model="gpt-5-chat")
translate("Hello", tgt_lang="nya", model="microsoft-translator")
translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", model="nllb-200-3.3b")
```

## Supported Models

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
| Google Gemma 3 | `gemma-3-1b-it`, `gemma-3-4b-it`, `gemma-3-12b-it`, `gemma-3-27b-it` |
| Alibaba Qwen 3 | `qwen3-4b`, `qwen3-8b`, `qwen3-14b` |
| Apertus | `apertus-8b` |
| Helsinki | `marian` |
| Cohere | `aya-101` |

## API Reference

### `translate()`

Translate a single text.

```python
translate(
    text: str,
    tgt_lang: str,
    src_lang: str = "auto",  # Auto-detect for API models only
    model: str = "microsoft-translator",
    device: str = None,      # For local models: "cuda:0", "cuda:1", etc.
    **kwargs
) -> str
```

**Examples:**

```python
# API models support auto-detection (src_lang="auto")
translate("Hello", tgt_lang="Spanish", model="gpt-5-chat")
translate("Hello", tgt_lang="ko", model="microsoft-translator")

# Local models require explicit source language
translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", model="nllb-200-3.3b")
translate("Hello", src_lang="en", tgt_lang="ny", model="translategemma")

# Specify GPU for local models
translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", 
          model="nllb-200-3.3b", device="cuda:1")
```

### `list_models()`

Pretty-print all available models.

```python
list_models()
```

### `get_supported_models()`

Get models programmatically.

```python
models = get_supported_models()
# Returns: {'Microsoft': [...], 'OpenAI': [...], 'Meta': [...], ...}
```

## Language Code Formats

Different backends use different language code formats:

| Backend | Code Format | Example | Reference |
|---------|-------------|---------|-----------|
| Microsoft Translator | Language names or ISO codes | `Chichewa`, `ny` | [Docs](https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support) |
| Google Translator | ISO 639-1 codes | `ny`, `sw` | [Docs](https://cloud.google.com/translate/docs/languages) |
| NLLB | Flores-200 codes | `eng_Latn`, `nya_Latn` | [Flores-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) |
| SeamlessM4T | Short codes | `eng`, `spa` | [Docs](https://github.com/facebookresearch/seamless_communication) |
| MADLAD-400 | BCP-47 codes | `en`, `ny` | [Paper](https://arxiv.org/pdf/2309.04662) |
| TranslateGemma | ISO 639-1 codes | `en`, `ny`, `es` | [HuggingFace](https://huggingface.co/google/translategemma-12b-it) |
| GPT Models | Full language names | `Spanish`, `Chichewa` | Any language GPT knows |

## Auto-Detection Support

Only API-based models support automatic source language detection (`src_lang="auto"`):

| Model | Auto-Detect |
|-------|-------------|
| `microsoft-translator` | ✅ |
| `google-translator` | ✅ |
| `gpt-*` models | ✅ |
| `deepseek-*` models | ✅ |
| Local LLMs (`gemma-3-*`, `qwen3-*`, `apertus-*`, `aya-101`) | ❌ Requires explicit `src_lang` |
| Local models (NLLB, Seamless, etc.) | ❌ Requires explicit `src_lang` |

## Environment Variables

```bash
# Azure OpenAI (for GPT models)
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

# Azure Translator
export AZURE_TRANSLATOR_ENDPOINT="https://api.cognitive.microsofttranslator.com/"

# Azure AI Foundry (for DeepSeek)
export AZURE_AI_FOUNDRY_DEEPSEEK_R1_ENDPOINT="..."
export AZURE_AI_FOUNDRY_DEEPSEEK_R1_MODEL="DeepSeek-R1"

# Google Cloud
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# HuggingFace (for local models)
export HF_TOKEN="your-hf-token"
```

## Architecture

```
translation_backends/
├── unified.py          # Main API: translate(), translate_batch()
├── factory.py          # get_translator() - creates backend instances
├── registry.py         # Model registry
├── base.py             # BaseTranslator ABC
├── api/                # API-based backends
│   ├── azure_openai.py
│   ├── azure_translator.py
│   ├── google.py
│   └── deepseek.py
└── local/              # Local model backends
    ├── nllb.py
    ├── marian.py
    ├── seamless.py
    ├── madlad.py
    ├── gemma3.py
    ├── qwen3.py
    ├── apertus.py
    ├── aya101.py
    ├── translate_gemma.py
    └── causal_lm.py
```
