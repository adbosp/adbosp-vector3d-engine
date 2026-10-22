"""Confidence tiers shared by every stage of the pipeline.

Core rule: geometry that can be determined mathematically must be
reconstructed mathematically. AI is only allowed to fill what is missing,
uncertain or creative, and its output must always be labelled as such.
"""
from __future__ import annotations

from enum import Enum


class Confidence(str, Enum):
    VERIFIED = "verified"          # derived exactly from input data (units, dimensions, cross-view agreement)
    ESTIMATED = "estimated"        # derived by a deterministic heuristic (bbox alignment, scale propagation)
    AI_GENERATED = "ai_generated"  # produced by a model; never overrides VERIFIED data

    @staticmethod
    def weakest(*levels: "Confidence") -> "Confidence":
        order = [Confidence.VERIFIED, Confidence.ESTIMATED, Confidence.AI_GENERATED]
        return max(levels, key=order.index)
