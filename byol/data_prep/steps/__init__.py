# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pipeline steps for data preparation.

CPT steps:
    - download_fineweb2, download_finewebedu
    - refine_tgt_lang, refine_eng
    - translate_eng_to_tgt_lang
    - bilingual_mix

SFT steps:
    - download_smoltalk2
    - translate_smoltalk2
    - download_aya
    - translate_aya
    - sft_bilingual_mix
"""
