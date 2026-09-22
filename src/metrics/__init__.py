"""The metrics package: the linear probes of probes.py and the KL divergence of kl.py.

Paper section: Section 4.1 (the KL divergence) and Section 5.1 (the probes).
"""
from .probes import fit_and_evaluate_multi, fit_probe, predict_probe, compute_r2, fit_and_evaluate
from .kl import kl_divergence
