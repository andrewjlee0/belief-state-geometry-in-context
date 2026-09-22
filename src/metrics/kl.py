"""KL divergence between rows of two arrays of next-token probabilities.

Paper section: Section 4.1 (the definition of in-context prediction accuracy).

Claim: In the paper's words, prediction accuracy at position t is "the KL divergence between the HMM's ground-truth
NTP, p(x_1:t) (extended to the LLM's full vocabulary size) and the LLM's NTP at position t."

Experiment: This module runs no experiment. run_kl.py uses it for the order-1 and order-0 baselines, whose
distributions live on the HMM tokens. The model's full-vocabulary KL is computed in src/model_utils.py.

Result: Not applicable.

How the code works: kl_divergence returns, for each row, the sum over tokens of p log((p + eps) / (q + eps)) with
eps = 1e-12.
"""
import numpy as np

# KL(p || q) per row, with a small epsilon inside the logarithm for zero probabilities.
def kl_divergence(p, q, eps=1e-12):
    """KL(p || q) per row."""
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)
