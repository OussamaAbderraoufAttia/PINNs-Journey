"""
Sampling Package

Collocation point sampling strategies for PINNs.
"""

from .collocation import (
    SamplingConfig,
    Sampler,
    UniformSampler,
    LHSSampler,
    SobolSampler,
    AdaptiveSampler,
    ResidualBasedSampler,
    create_sampler,
)

__all__ = [
    "SamplingConfig",
    "Sampler",
    "UniformSampler",
    "LHSSampler",
    "SobolSampler",
    "AdaptiveSampler",
    "ResidualBasedSampler",
    "create_sampler",
]