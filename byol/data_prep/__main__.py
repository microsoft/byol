# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Entry point for ``python -m byol.data_prep``.

Usage:
    python -m byol.data_prep --stage cpt --tgt-lang nya
    python -m byol.data_prep --stage sft --tgt-lang mri
    python -m byol.data_prep --stage cpt --config configs/data_prep/cpt/nya.yaml
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
