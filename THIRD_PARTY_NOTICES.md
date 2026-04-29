# Third-Party Notices for `byol`

This file records the main third-party software components used by the `byol` project.

It is intended as an engineering attribution inventory for repository sharing and release preparation. It is **not** a substitute for Component Governance, SBOM generation, or legal review. 

## Scope

- This inventory covers:
  - third-party repositories referenced by the setup flow
  - direct Python dependencies declared in `pyproject.toml`
  - optional and development dependencies declared in `pyproject.toml`
- The full pinned runtime environment is listed in `environment.yml`.
- `third_party_libs/` is ignored by Git in this repository, so those cloned repositories may exist locally without being tracked in the repo itself.

## Third-party repositories used by setup

The setup instructions in `SETUP.md` clone the following third-party repositories into `third_party_libs/`:

| Component | How it is used | Upstream source | Local ref observed during audit | License | Notes |
|---|---|---|---|---|---|
| LlamaFactory | Training integration | `https://github.com/hiyouga/LlamaFactory.git` | `e67ab9e2f2c924a459b8586e3370b01dd5734b36` | Apache License 2.0 | `SETUP.md` clones this repo and installs it with `pip install -e ".[torch,metrics]"`. The local ref above was observed in the current workspace; `SETUP.md` does not currently pin a commit. |
| lm-evaluation-harness | Evaluation integration | `https://github.com/EleutherAI/lm-evaluation-harness.git` | `69145a03d3e6ad0e552f729b7fe57b2762e7e4ca` | MIT License | `SETUP.md` pins this commit and installs it with `pip install -e ".[multilingual]"`. BYOL also applies local patches via `byol/eval/patches/apply_lmeval_patches.py`. |

## Direct runtime dependencies declared in `pyproject.toml`

The table below summarizes the direct runtime dependencies declared by the project. Resolved versions and license metadata were read from the local `byol` conda environment used for validation.

| Package | Declared requirement | Resolved version | License metadata | Upstream / homepage |
|---|---|---|---|---|
| `datasets` | `datasets>=2.14.0` | `4.0.0` | Apache Software License | `https://github.com/huggingface/datasets` |
| `sacremoses` | `sacremoses>=0.0.53` | `0.1.1` | MIT License | `https://github.com/hplt-project/sacremoses` |
| `tqdm` | `tqdm>=4.65.0` | `4.67.1` | MIT License; Mozilla Public License 2.0 (MPL 2.0) | — |
| `torch` | `torch>=2.0.0` | `2.9.1` | BSD-3-Clause | — |
| `evaluate` | `evaluate>=0.4.0` | `0.4.6` | Apache Software License | `https://github.com/huggingface/evaluate` |
| `numpy` | `numpy>=1.24.0` | `2.2.6` | BSD License | — |
| `pandas` | `pandas>=2.0.0` | `2.3.3` | BSD License | — |
| `matplotlib` | `matplotlib>=3.7.0` | `3.10.8` | Python Software Foundation License | — |
| `seaborn` | `seaborn>=0.12.0` | `0.13.2` | BSD License | — |
| `pyyaml` | `pyyaml>=6.0` | `6.0.3` | MIT License | `https://pyyaml.org/` |
| `python-dotenv` | `python-dotenv>=1.0.0` | `1.2.1` | BSD-3-Clause | — |
| `transformers` | `transformers>=4.35.0` | `4.57.6` | Apache Software License | `https://github.com/huggingface/transformers` |
| `accelerate` | `accelerate>=0.25.0` | `1.11.0` | Apache Software License | `https://github.com/huggingface/accelerate` |
| `openai` | `openai>=1.0.0` | `2.15.0` | Apache Software License | — |
| `azure-identity` | `azure-identity>=1.14.0` | `1.25.1` | MIT | — |
| `azure-ai-translation-text` | `azure-ai-translation-text>=1.0.0` | `1.0.1` | MIT License | `https://github.com/Azure/azure-sdk-for-python/tree/main/sdk` |
| `azure-ai-inference` | `azure-ai-inference>=1.0.0b9` | `1.0.0b9` | MIT License | `https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-inference` |
| `azure-keyvault-secrets` | `azure-keyvault-secrets>=4.7.0` | `4.10.0` | MIT License | `https://github.com/Azure/azure-sdk-for-python/tree/main/sdk` |
| `huggingface-hub` | `huggingface-hub>=0.20.0` | `0.36.0` | Apache Software License | `https://github.com/huggingface/huggingface_hub` |
| `langdetect` | `langdetect>=1.0.9` | `1.0.9` | Apache Software License | `https://github.com/Mimino666/langdetect` |
| `immutabledict` | `immutabledict>=4.2.0` | `4.3.0` | MIT | — |
| `requests` | `requests>=2.28.0` | `2.32.5` | Apache Software License | `https://requests.readthedocs.io` |

## Optional and development dependencies declared in `pyproject.toml`

| Group | Package | Declared requirement | Resolved version | License metadata | Upstream / homepage |
|---|---|---|---|---|---|
| `dev` | `pytest` | `pytest>=7.0.0` | `9.0.2` | MIT | — |
| `dev` | `pytest-cov` | `pytest-cov>=4.0.0` | `7.0.0` | MIT | — |
| `dev` | `mypy` | `mypy>=1.0.0` | `1.19.1` | MIT License | — |
| `dev` | `ruff` | `ruff>=0.1.0` | `0.15.1` | MIT License | `https://docs.astral.sh/ruff` |
| `google` / `all` | `google-cloud-translate` | `google-cloud-translate>=3.11.0` | `3.24.0` | Apache Software License | `https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-translate` |

## Additional notes

- `environment.yml` contains the broader pinned environment used in practice, including transitive and toolchain dependencies that are not repeated in full here.
- Some upstream packages publish additional bundled-license details beyond the short metadata shown above. For final release review, consult upstream license files and run the required Component Governance / SBOM process against the final environment.
- This file should be updated whenever:
  - `pyproject.toml` direct dependencies change
  - optional dependency groups change
  - setup begins cloning different third-party repositories
  - pinned refs for cloned repositories change

## Source files used to prepare this notice

- `pyproject.toml`
- `environment.yml`
- `SETUP.md`
- `third_party_libs/LlamaFactory/LICENSE`
- `third_party_libs/lm-evaluation-harness/LICENSE.md`
- `byol/eval/patches/apply_lmeval_patches.py`

**Note:** This file is provided as a convenient summary but does not replace the original licenses. Please refer to the original license files of each project for the exact terms. 