#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
BYOL Language Resource Assessment

Evaluate translation quality for low-resource languages via round-trip translation.
Compare different translators (API and local models) to find the best one.

Usage:
    python -m byol.language_resource_assessment --task find-best-translator --tgt-lang Chichewa
    python -m byol.language_resource_assessment --list-translators
"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
