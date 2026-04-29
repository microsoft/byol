# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Training runner - orchestrates LlamaFactory CLI execution."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .config import TrainConfig
from .constants import (
    ENV_CUDA_VISIBLE_DEVICES,
    TEMP_CONFIG_PREFIX,
    TEMP_CONFIG_SUFFIX,
)
from .secrets import get_hf_token, mask_token, setup_environment

logger = logging.getLogger("byol-train")


@dataclass
class TrainResult:
    """Result of a training run."""

    success: bool
    output_dir: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    config_path: Optional[str] = None


class TrainingRunner:
    """Training runner that invokes LlamaFactory CLI."""

    def __init__(self, config: TrainConfig, dry_run: bool = False) -> None:
        """Initialize runner with config."""
        self.config = config
        self.dry_run = dry_run
        self._temp_config_path: Optional[str] = None

    def _setup_environment(self) -> None:
        """Configure environment variables for training."""
        setup_environment()

        # Reduce CUDA memory fragmentation for large model allocations
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        if self.config.gpus:
            os.environ[ENV_CUDA_VISIBLE_DEVICES] = self.config.gpus
            logger.info(f"🖥️  CUDA devices: {self.config.gpus}")

    def _generate_output_dir(self) -> str:
        """Generate unique output directory path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.config.model_name_or_path).name.replace("-", "_")
        dir_name = f"{model_name}_{self.config.stage}_{timestamp}"
        output_dir = Path(self.config.output_dir) / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir.resolve())

    def _create_temp_config(self, output_dir: str) -> str:
        """Create temporary YAML config for LlamaFactory."""
        llama_config = self.config.to_llamafactory(output_dir)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=TEMP_CONFIG_SUFFIX,
            prefix=TEMP_CONFIG_PREFIX,
            delete=False,
        ) as f:
            yaml.dump(llama_config, f, default_flow_style=False)
            self._temp_config_path = f.name

        logger.debug(f"Created temp config: {self._temp_config_path}")
        return self._temp_config_path

    def _cleanup_temp_config(self) -> None:
        """Remove temporary config file if it exists."""
        if self._temp_config_path and Path(self._temp_config_path).exists():
            Path(self._temp_config_path).unlink()
            logger.debug(f"Cleaned up temp config: {self._temp_config_path}")

    def run(self) -> TrainResult:
        """Execute training and return result."""
        start_time = datetime.now()
        output_dir = self._generate_output_dir()

        # Log training info
        logger.info("=" * 60)
        logger.info("🚀 BYOL TRAINING")
        logger.info(f"   Stage: {self.config.stage.upper()}")
        logger.info(f"   Model: {self.config.model_name_or_path}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   HF Token: {mask_token(get_hf_token())}")
        if self.config.lora:
            logger.info(f"   LoRA: rank={self.config.lora.rank}, alpha={self.config.lora.alpha}")
        logger.info("=" * 60)

        self._setup_environment()
        config_path = self._create_temp_config(output_dir)

        if self.dry_run:
            logger.info("[DRY RUN] Config file:")
            logger.info(Path(config_path).read_text())
            self._cleanup_temp_config()
            return TrainResult(success=True, output_dir=output_dir, config_path=config_path)

        try:
            cmd = ["llamafactory-cli", "train", config_path]
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=False, check=False)
            duration = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                logger.info(f"✅ Training complete: {output_dir}")
                return TrainResult(
                    success=True,
                    output_dir=output_dir,
                    duration_seconds=duration,
                    config_path=config_path,
                )
            else:
                error_msg = f"LlamaFactory exited with code {result.returncode}"
                logger.error(f"❌ Training failed: {error_msg}")
                return TrainResult(
                    success=False,
                    output_dir=output_dir,
                    error=error_msg,
                    duration_seconds=duration,
                    config_path=config_path,
                )

        except FileNotFoundError as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"LlamaFactory CLI not found: {e}"
            logger.error(f"❌ {error_msg}")
            return TrainResult(
                success=False,
                output_dir=output_dir,
                error=error_msg,
                duration_seconds=duration,
            )

        except subprocess.SubprocessError as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Subprocess error: {e}"
            logger.error(f"❌ {error_msg}")
            return TrainResult(
                success=False,
                output_dir=output_dir,
                error=error_msg,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unexpected error: {e}"
            logger.exception(error_msg)
            return TrainResult(
                success=False,
                output_dir=output_dir,
                error=error_msg,
                duration_seconds=duration,
            )

        finally:
            self._cleanup_temp_config()
