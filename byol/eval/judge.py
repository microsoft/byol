# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LLM-as-Judge Evaluation Module for BYOL Framework."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths relative to this package
_EVAL_DIR = Path(__file__).resolve().parent          # byol/eval/
_REPO_ROOT = _EVAL_DIR.parent.parent                 # repo root
_API_PATH = _REPO_ROOT / "api"
_API_EXAMPLE_PATH = _REPO_ROOT / "api.example"
_LOCAL_CONFIGS = _REPO_ROOT / "configs" / "eval"


class LLMJudgeRunner:
    """Wrapper for LLM-as-Judge evaluation.

    Requires:
        - ``api/`` folder with ``get_azure_api.py`` (copy from ``api.example/``).

    Setup::

        cp -r api.example api
        # Configure api/get_azure_api.py with your LLM provider credentials
    """

    def __init__(
        self,
        model_config: str | None = None,
        dataset_config: str | None = None,
        output_dir: str = "results/eval/judge",
    ) -> None:
        """Initialise LLM-as-Judge runner.

        Args:
            model_config: Path to model configuration YAML.
            dataset_config: Path to dataset configuration YAML.
            output_dir: Directory for evaluation results.

        Raises:
            FileNotFoundError: If required files/folders are missing.
        """
        self.model_config = str(model_config or _LOCAL_CONFIGS / "judge_models.yaml")
        self.dataset_config = str(dataset_config or _LOCAL_CONFIGS / "judge_datasets.yaml")
        self.output_dir = output_dir

        # Validate config files
        if not Path(self.model_config).exists():
            raise FileNotFoundError(f"Model config not found: {self.model_config}")
        if not Path(self.dataset_config).exists():
            raise FileNotFoundError(f"Dataset config not found: {self.dataset_config}")

        # Validate API module
        if not _API_PATH.exists():
            setup_instructions = (
                f"API module not found: {_API_PATH}\n\n"
                "To set up the API module:\n"
                "  1. Copy the example: cp -r api.example api\n"
                "  2. Configure credentials in api/get_azure_api.py\n"
                "  3. Set environment variables:\n"
                "     - AZURE_OPENAI_ENDPOINT\n"
                "     - AZURE_OPENAI_API_KEY\n"
                "     - Or OPENAI_API_KEY for direct OpenAI\n\n"
                f"See {_API_EXAMPLE_PATH / 'README.md'} for details."
            )
            raise FileNotFoundError(setup_instructions)

    def run(self) -> None:
        """Run the LLM-as-Judge evaluation."""
        from .llm_judge import run_evaluation

        logger.info("=" * 60)
        logger.info("LLM-AS-JUDGE EVALUATION")
        logger.info(f"  Model config:   {self.model_config}")
        logger.info(f"  Dataset config: {self.dataset_config}")
        logger.info(f"  Output dir:     {self.output_dir}")
        logger.info("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)
        run_evaluation(self.model_config, self.dataset_config, self.output_dir)
