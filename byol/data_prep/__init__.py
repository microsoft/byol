# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BYOL Data Preparation — automated CPT/SFT/Eval data pipelines.

Orchestrates downloading, refining, subsetting, and translating datasets
for continual pretraining (CPT), supervised finetuning (SFT), and evaluation.

Example:
    # CLI
    python -m byol.data_prep --stage cpt --tgt-lang nya
    python -m byol.data_prep --stage sft --tgt-lang mri
    python -m byol.data_prep --stage eval --tgt-lang nya
    python -m byol.data_prep --stage cpt --tgt-lang nya --dry-run

    # Python API
    from byol.data_prep import CPTDataPrepConfig, CPTDataPrepRunner
    config = CPTDataPrepConfig.from_yaml("configs/data_prep/cpt/nya.yaml")
    runner = CPTDataPrepRunner(config)
    runner.run()

    from byol.data_prep import EvalDataPrepConfig, EvalDataPrepRunner
    config = EvalDataPrepConfig.from_yaml("configs/data_prep/eval/nya.yaml")
    runner = EvalDataPrepRunner(config)
    runner.run()
"""

__version__ = "0.1.0"
__author__ = "BYOL Team"

from .config import (
    CPTDataPrepConfig,
    EvalBenchmarkConfig,
    EvalDataPrepConfig,
    ExtraSource,
    SFTDataPrepConfig,
)
from .constants import (
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_STAGES,
)
from .cpt_data_prep_runner import CPTDataPrepResult, CPTDataPrepRunner
from .eval_data_prep_runner import EvalDataPrepResult, EvalDataPrepRunner
from .sft_data_prep_runner import SFTDataPrepRunner, SFTDataPrepResult

__all__ = [
    "__version__",
    "CPTDataPrepConfig",
    "SFTDataPrepConfig",
    "EvalDataPrepConfig",
    "EvalBenchmarkConfig",
    "ExtraSource",
    "CPTDataPrepRunner",
    "CPTDataPrepResult",
    "SFTDataPrepRunner",
    "SFTDataPrepResult",
    "EvalDataPrepRunner",
    "EvalDataPrepResult",
    "DEFAULT_OUTPUT_DIR",
    "SUPPORTED_STAGES",
]
