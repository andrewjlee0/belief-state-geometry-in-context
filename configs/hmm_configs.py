"""The 40 HMMs of the paper: four families with 10 parametrizations each, and the representative parametrization of
each family.

Paper section: Section 3 (Experimental setup, HMMs) and Appendix C.2 with Tables 1 and 2 and Figure 10.

Claim: In the paper's words, "We experiment on 40 HMMs with non-trivial belief geometries, spanning four families, Mess3,
Arch, Wing, and Strata, with 10 parametrizations per family", and "Results are presented for one illustrative
parametrization per family: Mess3(alpha = 0.01, x = 0.02), Arch(alpha = 0.99), Wing(alpha = 0.98, x = 0.4),
Strata(alpha = 0.97, t0 = 0.38, t1 = 0.54)."

Experiment: This module runs no experiment. Every experiment script reads its families and parametrizations here,
and every results file labels its rows with the labels defined here.

Saved Outputs: None. This module writes no files.

How the code works: HMMS maps each family name to its constructor, its order-1 and order-0 constructors, the list of
parameter tuples, a label function that produces the parametrization label used in every results file (for example
"a=0.98, x=0.4"), the token letters, and the numbers of tokens and hidden states. REPRESENTATIVES holds the
main-text parametrization of each family.
"""
import numpy as np
from src.hmm.definitions import *

# The four families. Parameter tuples are passed positionally to the constructors of src/hmm/definitions.py.
HMMS = {
    # Mess3: alpha and x. Ten pairs with x fixed at 0.02 except for the first, and no order-0 constructor because
    # its stationary token distribution is uniform.
    "Mess3": {
        "fn": mess3_matrices, "order_one_fn": mess3_order_one, "order_zero_fn": None,
        "params": [(0.005,0.01),(0.005,0.02),(0.01,0.02),(0.05,0.02),(0.10,0.02),
                   (0.60,0.02),(0.70,0.02),(0.80,0.02),(0.85,0.02),(0.90,0.02)],
        "label_fn": lambda p: f"a={p[0]}, x={p[1]}",
        "token_names": np.array(["F","Q","V"]), "n_tokens": 3, "n_states": 3,
    },
    # Arch: alpha from 0.90 to 0.99.
    "Arch": {
        "fn": arch_matrices, "order_one_fn": arch_order_one, "order_zero_fn": arch_order_zero,
        "params": [(a,) for a in np.arange(0.90, 1.00, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}",
        "token_names": np.array(["F","Q","V"]), "n_tokens": 3, "n_states": 4,
    },
    # Wing: alpha from 0.90 to 0.99 with x = 0.4.
    "Wing": {
        "fn": wing_matrices, "order_one_fn": wing_order_one, "order_zero_fn": wing_order_zero,
        "params": [(a, 0.4) for a in np.arange(0.90, 1.00, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}, x={p[1]}",
        "token_names": np.array(["F","Q"]), "n_tokens": 2, "n_states": 3,
    },
    # Strata: alpha from 0.90 to 0.99 with t0 = 0.38 and t1 = 0.54.
    "Strata": {
        "fn": strata_matrices, "order_one_fn": strata_order_one, "order_zero_fn": strata_order_zero,
        "params": [(a, 0.38, 0.54) for a in np.arange(0.90, 1.00, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}, t0={p[1]}, t1={p[2]}",
        "token_names": np.array(["F","Q"]), "n_tokens": 2, "n_states": 3,
    },
}

# The parametrization of each family shown in the main text.
REPRESENTATIVES = {"Mess3": (0.01,0.02), "Arch": (0.99,), "Wing": (0.98,0.4), "Strata": (0.97,0.38,0.54)}
