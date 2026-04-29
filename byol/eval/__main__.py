# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Entry point for ``python -m byol.eval``.

Usage:
    python -m byol.eval --model <path> --type base --tgt-lang mri
    python -m byol.eval judge --model-config <config> --dataset-config <config>
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
