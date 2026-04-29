# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Backward-compatible re-export from ``cpt_bilingual_mix``.

.. deprecated::
    Import from ``byol.data_prep.steps.cpt_bilingual_mix`` instead.
"""

from .cpt_bilingual_mix import create_bilingual_mix  # noqa: F401

__all__ = ["create_bilingual_mix"]
