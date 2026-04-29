# Repository Setup Guide

Step-by-step instructions to set up the BYOL toolkit from scratch.

## 1. Clone & Create Environment

```bash
git clone https://github.com/microsoft/byol
cd byol

# Create conda environment (includes BYOL in editable mode)
conda env create -f environment.yml
conda activate byol
```

## 2. Set Up Credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

| Variable | Required For | How to Get |
|----------|-------------|------------|
| `HF_TOKEN` | Downloading models & datasets | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `AZURE_OPENAI_ENDPOINT` | GPT-based translation & refinement | Azure Portal → OpenAI resource → Keys & Endpoint |
| `AZURE_TRANSLATOR_ENDPOINT` | Microsoft Translator (eval benchmarks) | Azure Portal → Translator resource → Keys & Endpoint |

**Optional credentials** (only if you use these specific translators):

| Variable | Required For |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` + `GOOGLE_CLOUD_PROJECT` | Google Cloud Translation |
| `AZURE_AI_FOUNDRY_DEEPSEEK_R1_ENDPOINT` | DeepSeek R1 models |

> **Minimum viable setup:** `HF_TOKEN` + `AZURE_OPENAI_ENDPOINT` is enough to run the full pipeline with GPT-based translators.

**Azure authentication:** BYOL uses Azure Entra ID (`DefaultAzureCredential`) for all Azure services. Ensure you are logged in before running:

```bash
az login
```

If running on an Azure VM with managed identity, no login is needed.

## 3. Data Directory

By default, BYOL stores training data (CPT, SFT) under `~/byol-data/<lang>/`. To use a different location, set `BYOL_DATA_DIR`:

```bash
# In .env or shell
export BYOL_DATA_DIR=/path/to/your/data
```

Evaluation data for Chichewa and Māori is **shipped with the repo** at `data/<lang>/eval/` and does not require this setting.

## 4. Install Third-Party Libraries

These are needed for **training** (LlamaFactory) and **evaluation** (lm-evaluation-harness):

```bash
# LlamaFactory (for training)
cd third_party_libs/
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory && pip install -e ".[torch,metrics]" && cd ../..

# lm-evaluation-harness (for evaluation)
cd third_party_libs/
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 69145a03d3e6ad0e552f729b7fe57b2762e7e4ca
pip install -e ".[multilingual]"
cd ../..

# Apply BYOL patches to lm-eval
python byol/eval/patches/apply_lmeval_patches.py
```

> **Skip if** you only need data prep / translation — third-party libs are only required for Steps 8-11.

## 5. Verify Installation

```bash
# Check BYOL is importable
python -c "from byol import translate, list_models; print('OK')"

# Check LlamaFactory (optional)
python -c "import llamafactory; print('LlamaFactory OK')"

# Check lm-eval (optional)
python -c "import lm_eval; print('lm_eval OK')"

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## 6. Quick Smoke Test

Run the full pipeline for a test language with 10 samples to verify everything works:

```bash
python -m byol.pipeline run-all --tgt-lang nya --device 0 \
  --model google/gemma-3-4b-pt --instruct-model google/gemma-3-4b-it \
  --max-samples 10 --quick-test
```
