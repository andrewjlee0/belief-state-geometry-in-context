"""The HMM package: the family constructors of definitions.py and the exact belief computations of core.py.

Paper section: Section 2 (Background). See the two modules for the equations each implements.
"""
from .definitions import *
from .core import (stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs,
                   emission_matrix, next_token_probs, precompute_belief_tables, compute_k_beliefs)
