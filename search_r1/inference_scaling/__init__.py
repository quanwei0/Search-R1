"""
Inference-time scaling algorithms for Search-R1.
"""

from .base_scaler import BaseInferenceGenerator
from .best_of_n import BestOfNGenerator
# from .stepwise_bon import StepwiseBestOfNScaler
from .beam_search import BeamSearchGenerator

__all__ = ['BaseInferenceGenerator', 'BestOfNGenerator', 'BeamSearchGenerator']