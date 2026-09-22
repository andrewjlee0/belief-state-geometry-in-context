"""k-suffix belief probes from activations on full-HMM, 1-HMM, and 0-HMM sequences.

Paper section: Section 5.2, the paragraph "Residual stream activations capture more than order-one belief state
structure", with Figure 5 (top), Appendix K with Figures 41 to 46, and Table 3 of Appendix F.

Claim: "Residual stream activations capture more than order-one belief state structure." In the paper's words, "If the
activations encode belief states of short-suffix approximations of the full HMM, then the full HMM's belief state
should be equally decodable whether the activations are induced by full HMM sequences or by sequences that preserve
only short-range correlations."

Experiment: For every parametrization and seed, three token sequences are generated: one from the full HMM, one from
its 1-HMM (each token depends on the previous token only), and one from its 0-HMM (i.i.d. tokens from the stationary
token distribution, Section 2.3). Each sequence is run through the model, and at every layer a probe is fit from the
late-window activations to the k-suffix belief state of the full HMM, eta(x_{t-k+1:t}), computed from that sequence's
own tokens, for every k from 1 to 20. The result is an R² curve over k for every sequence, layer, and token source.

Saved Outputs: ksuffix_probes_<model>.csv in --output_dir (with an optional --tag before the model key), with the
columns hmm, param, dist (real, order-1, or order-0), layer, k, seed, and R2, one row per token source, layer, suffix
length, and seed for every parametrization.

How the code works: k-suffix beliefs for small k come from lookup tables over all possible k-token suffixes (up to
k = 20 with two tokens and k = 12 with three) and from direct filtering for larger k. For each parametrization,
token source, and seed, the script generates the sequence (seeded HMM sampling for the full HMM and the 1-HMM,
seeded i.i.d. draws for the 0-HMM), tokenizes it, checks that every HMM token is matched to a model token, extracts
the late-window activations at every layer in one chunked forward pass, and computes the full HMM's k-suffix
beliefs from the matched tokens. The targets for all k are stacked side by side, so that one probe fit per layer
(pseudo-inverse with a ones column, seeded 20 percent training split) yields every k's held-out R² by slicing. Each
(source, seed) sequence is processed and freed before the next one. Rows of ksuffix_probes_<model>.csv carry hmm,
param, dist (real, order-1, or order-0), layer, k, seed, and R2.
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import (stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs,
                     precompute_belief_tables, compute_k_beliefs, emission_matrix)
from src.metrics.probes import fit_and_evaluate_multi_stacked
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

# The three token sources of Figure 5 (top): the full HMM, its order-1 approximation, and its order-0 approximation.
DISTS = ["real", "order-1", "order-0"]

def order0_tokens(source, T, pi, n_tok, seq_len, seed):
    """Tokens of the 0-HMM: i.i.d. draws from a generator seeded with seed + 9999. Source 'hmm' draws from the
    stationary token distribution pi M of Section 2.3, which is the paper's setting. Source 'uniform' draws every
    token with equal probability, which coincides with 'hmm' only for Mess3."""
    rng = np.random.default_rng(seed + 9999)
    if source == "uniform":
        return rng.integers(0, n_tok, size=seq_len)
    p0 = np.asarray(pi @ emission_matrix(T), dtype=np.float64)
    p0 = p0 / p0.sum()
    return rng.choice(n_tok, size=seq_len, p=p0)

def main():
    # Command-line options. The defaults are the paper's settings: 20,000-token sequences, probes on positions 15,000
    # onward, 10 seeds, a 20 percent training split, suffix lengths up to 20, and all three token sources.
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--k_max", type=int, default=20)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--params", nargs="+", default=None, help="explicit parametrization labels (default: all)")
    P.add_argument("--dists", nargs="+", default=DISTS, choices=DISTS, help="which source distributions to run")
    P.add_argument("--order0_source", choices=["hmm", "uniform"], default="hmm",
                   help="hmm = i.i.d. tokens from the stationary token distribution pi M (Section 2.3, the paper's setting); "
                        "uniform = equiprobable tokens")
    P.add_argument("--tag", default="", help="output filename tag: ksuffix_probes_{tag}{model}.csv")
    P.add_argument("--smoke", action="store_true", help="tiny plumbing test (short sequences, 2 seeds, k<=3, one parametrization)")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()

    # Smoke mode is a plumbing test: 2,400-token sequences probed from position 2,000, two seeds, k up to 3, and one
    # parametrization per family.
    if args.smoke:
        args.seq_len, args.probe_start, args.n_seeds, args.k_max = 2400, 2000, 2, 3
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load the model once. Every layer is probed.
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    # Suffix lengths k = 1, ..., k_max.
    K_VALUES = list(range(1, args.k_max + 1))
    out_csv = os.path.join(args.output_dir, f"ksuffix_probes_{args.tag}{ms}.csv")
    print(f"dists {args.dists}; order-0 source: {args.order0_source}; output {out_csv}", flush=True)

    all_rows = []
    # Loop over the requested families and their parametrizations (optionally restricted to explicit labels).
    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        params = [p for p in cfg["params"] if cfg["label_fn"](p) in args.params] if args.params else cfg["params"]
        if args.smoke: params = params[:1]
        if not params: continue
        print(f"\n===== {hmm_name} ({len(params)} parametrizations) =====", flush=True)
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        n_tok = cfg["n_tokens"]
        n_states = cfg["n_states"]

        # k-suffix beliefs are read from lookup tables over all n_tok^k suffixes while the table is small enough, and are
        # filtered directly for larger k.
        max_k_lookup = 20 if n_tok == 2 else 12
        pbar = tqdm(total=len(params) * len(args.dists) * args.n_seeds, desc=hmm_name)
        for param in params:
            label = cfg["label_fn"](param)
            # The full HMM: per-token transition matrices, their stacked array, and the stationary distribution.
            T = cfg["fn"](*param)
            T_stack = np.stack(T)
            pi = stationary_distribution(T)

            # The 1-HMM is built only when order-1 sequences are requested.
            T_o1 = pi_o1 = None
            if "order-1" in args.dists:
                assert cfg["order_one_fn"] is not None, f"{hmm_name} has no order-one construction"
                T_o1 = cfg["order_one_fn"](*param)
                pi_o1 = stationary_distribution(T_o1)
            small_ks = [k for k in K_VALUES if k <= max_k_lookup]

            # Lookup tables of the full HMM's k-suffix beliefs: the belief reached from the stationary prior after each
            # possible k-token suffix.
            tables = precompute_belief_tables(small_ks, T, pi) if small_ks else {}

            # Token source, then seed. The full-HMM and 1-HMM sequences are sampled from the respective process with the
            # same seed; the 0-HMM tokens are i.i.d. draws.
            for dist_name in args.dists:
                for seed in range(args.n_seeds):
                    if dist_name == "real":
                        tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                    elif dist_name == "order-1":
                        tokens = sample_hmm_sequence(T_o1, pi_o1, args.seq_len, seed=seed)
                    else:
                        tokens = order0_tokens(args.order0_source, T, pi, n_tok, args.seq_len, seed)

                    # Write the sequence as space-separated letters and locate the model token carrying each HMM token.
                    prompt = tokens_to_prompt(tokens, cfg["token_names"])
                    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                    pos_indices, tok_at_pos = match_positions(input_ids, tok_ids)
                    n_matched = min(len(tokens), len(pos_indices))

                    # Every HMM token must map to exactly one model token, so that the k-suffix beliefs computed below from the
                    # matched tokens line up with the activations.
                    assert n_matched == len(tokens), f"alignment: {n_matched} matched of {len(tokens)} tokens"
                    late_pos = pos_indices[args.probe_start:n_matched]
                    n = len(late_pos)
                    if n == 0: pbar.update(1); continue

                    # Late-window activations at every layer from one chunked forward pass with KV caching.
                    acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)

                    tok = tok_at_pos[:n_matched].astype(np.int64)
                    # The probe target is always the FULL HMM's k-suffix belief eta(x_{t-k+1:t}), computed from this sequence's
                    # own tokens whatever process generated them (Section 5.2). Element j of each array belongs to late
                    # position probe_start + j.
                    beliefs_by_k = compute_k_beliefs(tok, args.probe_start, n, K_VALUES, tables, T_stack, pi, n_tok, max_k_lookup)

                    # The paper's split: a random 20 percent of the late positions train the probe and the rest score it, seeded
                    # by the sequence seed.
                    idx_tr, idx_te = train_test_split(np.arange(n), train_size=args.train_frac, random_state=seed)
                    # The targets of all k are stacked side by side, so that one probe fit per layer serves every k and the R² of
                    # each k is read off its slice of the prediction.
                    Y_tr = torch.tensor(np.hstack([beliefs_by_k[k][idx_tr] for k in K_VALUES]), device=device, dtype=torch.float32)
                    Y_te = torch.tensor(np.hstack([beliefs_by_k[k][idx_te] for k in K_VALUES]), device=device, dtype=torch.float32)

                    # One pseudo-inverse of the augmented training activations per layer, shared by all k.
                    for l in layers:
                        X = acts[l]
                        if X.numel() == 0: continue
                        X_tr = X[idx_tr].to(device).float()
                        X_te = X[idx_te].to(device).float()
                        r2s = fit_and_evaluate_multi_stacked(X_tr, X_te, Y_tr, Y_te, n_states, K_VALUES, use_bias=True)
                        for k, r2 in r2s.items():
                            all_rows.append({"hmm": hmm_name, "param": label, "dist": dist_name, "layer": l, "k": k, "seed": seed, "R2": r2})
                        del X_tr, X_te

                    del acts, Y_tr, Y_te, beliefs_by_k
                    gc.collect()
                    torch.cuda.empty_cache()
                    pbar.update(1)

            # Checkpoint the accumulated rows after every parametrization, so that a partial run is usable.
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        pbar.close()
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Done. {len(all_rows)} rows -> {out_csv}", flush=True)

if __name__ == "__main__": main()
