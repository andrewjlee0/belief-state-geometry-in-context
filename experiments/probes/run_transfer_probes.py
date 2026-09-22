"""Transfer probes between the parametrizations of a family, and their ground-truth counterpart.

Paper section: Section 5.2, the paragraph "Residual stream activations carry HMM-specific belief state information",
with Figure 4, and Appendix J with Figures 35 to 40.

Claim: "Residual stream activations carry HMM-specific belief state information." In the paper's words, "If activations
carry HMM-specific information, R² should be higher between parametrizations whose belief state geometries are more
similar, and lower between dissimilar ones. That is, the pattern of transfer R² across parametrization pairs should
covary with the degree to which the parametrizations' belief states predict one another."

Experiment: For each family and layer, an empirical matrix is built whose entry (i, j) is the R² of a transfer probe
trained on the activations of parametrization i's sequence to predict parametrization j's belief states computed on
that same sequence, with the paper's 20/80 train-test split. It is compared with a ground-truth matrix that involves
no LLM activations, whose entry (i, j) is the R² of a regression from parametrization i's belief states to
parametrization j's belief states on the same sequence.

Result: Across most layers the empirical and ground-truth matrices are correlated for all HMM families and all six
LLMs, which supports the HMM-specificity account. The empirical transfer R² values are substantially higher in
magnitude than the ground-truth ones, which suggests that the activations also carry some HMM-generic information.

How the code works: Two passes. In the empirical pass, for each source parametrization i and seed, the script samples
i's sequence, runs the model once, extracts the late-window activations at every layer, computes the belief states of
every parametrization j of the family on that same token sequence with j's own transition matrices and prior, and at
every layer fits probes from i's activations to each j's beliefs with one shared pseudo-inverse (ones column for the
bias) on the seeded 20 percent training split, scoring held-out R². The rows (hmm, source, target, layer, seed, R2,
self) go to transfer_probes_<model>.csv. In the ground-truth pass, with the same sequences and splits, j's beliefs are
regressed on i's beliefs with bias and the rows go to transfer_probes_gt_<model>.csv.
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_and_evaluate_multi, fit_probe, predict_probe, compute_r2
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

def main():
    # Command-line options. The defaults are the paper's settings: 20,000-token sequences, probes on positions 15,000
    # onward, 10 seeds, a 20 percent training split, and forward passes in chunks of 4,096 tokens.
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load the model once. Every layer is probed.
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    # Empirical pass: probes from parametrization i's activations to every parametrization j's belief states on i's
    # sequence, for every family, source parametrization, seed, and layer.
    cross_rows = []

    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        param_labels = [cfg["label_fn"](p) for p in cfg["params"]]
        n_states = cfg["n_states"]

        # Transition matrices and stationary distributions of every parametrization of the family, so that each
        # parametrization's belief states can be computed on any sequence.
        all_T = {}
        all_pi = {}
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param)
            all_T[label] = np.stack(T)
            all_pi[label] = stationary_distribution(T)

        pbar = tqdm(total=len(cfg["params"]) * args.n_seeds, desc=hmm_name)

        for source_param in cfg["params"]:
            source_label = cfg["label_fn"](source_param)

            for seed in range(args.n_seeds):
                # Source sequence: parametrization i's own sequence for this seed, written as space-separated letters,
                # matched to model tokens, and run through the model once. The residual stream at every layer is kept
                # for the late window only.
                tokens = sample_hmm_sequence(
                    cfg["fn"](*source_param), all_pi[source_label],
                    args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, tok_at_pos = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                late_pos = pos_indices[args.probe_start:n_matched]
                n_late = len(late_pos)
                if n_late == 0:
                    pbar.update(1)
                    continue

                acts, _ = extract_activations_chunked(
                    wrapper, input_ids, layers, late_pos, args.chunk_size, device
                )

                # Targets: the belief states of every parametrization j of the family, computed on this same token
                # sequence with j's own transition matrices and prior. The entry j = i is the ordinary belief probe.
                beliefs = {}
                for target_label in param_labels:
                    b = full_bayesian_beliefs(tok_i64, all_T[target_label], all_pi[target_label])
                    beliefs[target_label] = b[args.probe_start:n_matched]

                # The paper's split: a random 20 percent of the late positions for training and the rest for scoring,
                # seeded by the sequence seed.
                idx_tr, idx_te = train_test_split(
                    np.arange(n_late), train_size=args.train_frac, random_state=seed
                )

                # At every layer, one pseudo-inverse of the augmented training activations serves every target
                # parametrization, and each target's held-out R² is recorded.
                for l in layers:
                    X = acts[l]
                    if X.numel() == 0: continue
                    X_tr, X_te = X[idx_tr], X[idx_te]

                    # Train and test targets for every parametrization j, as tensors on the device.
                    targets = {}
                    for target_label in param_labels:
                        y = beliefs[target_label]
                        Y_tr = torch.tensor(y[idx_tr], device=device, dtype=torch.float32)
                        Y_te = torch.tensor(y[idx_te], device=device, dtype=torch.float32)
                        targets[target_label] = (Y_tr, Y_te)

                    r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                    for target_label, r2 in r2s.items():
                        cross_rows.append({
                            "hmm": hmm_name,
                            "source": source_label,
                            "target": target_label,
                            "layer": l,
                            "seed": seed,
                            "R2": r2,
                            "self": source_label == target_label,
                        })

                del acts, beliefs
                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)

        pbar.close()

        # Write transfer_probes_<model>.csv after each family (rewritten with all rows accumulated so far).
        pd.DataFrame(cross_rows).to_csv(
            os.path.join(args.output_dir, f"transfer_probes_{ms}.csv"), index=False
        )

    # Ground-truth transfer matrix (no model involved): entry (i, j) is the R² of a linear regression with bias from
    # parametrization i's belief states to parametrization j's belief states on i's sequence, with the same sequences
    # and the same seeded 20/80 split as the empirical pass.
    print("\n===== Ground-truth cross-R² =====")
    gt_rows = []
    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        param_labels = [cfg["label_fn"](p) for p in cfg["params"]]

        all_T = {}
        all_pi = {}
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param)
            all_T[label] = np.stack(T)
            all_pi[label] = stationary_distribution(T)

        pbar = tqdm(total=len(cfg["params"]) * args.n_seeds, desc=f"GT {hmm_name}")

        for source_param in cfg["params"]:
            source_label = cfg["label_fn"](source_param)

            for seed in range(args.n_seeds):
                # The same sequence as in the empirical pass (same parametrization, same seed).
                tokens = sample_hmm_sequence(
                    cfg["fn"](*source_param), all_pi[source_label],
                    args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)

                # Parametrization i's belief states on the late window are the regressors.
                b_source = full_bayesian_beliefs(
                    tok_i64, all_T[source_label], all_pi[source_label]
                )[args.probe_start:]

                n = len(b_source)
                idx_tr, idx_te = train_test_split(
                    np.arange(n), train_size=args.train_frac, random_state=seed
                )

                X_tr = torch.tensor(b_source[idx_tr], device=device, dtype=torch.float32)
                X_te = torch.tensor(b_source[idx_te], device=device, dtype=torch.float32)

                # Every parametrization j's belief states on the same sequence are the targets.
                targets = {}
                for target_label in param_labels:
                    b_target = full_bayesian_beliefs(
                        tok_i64, all_T[target_label], all_pi[target_label]
                    )[args.probe_start:]
                    Y_tr = torch.tensor(b_target[idx_tr], device=device, dtype=torch.float32)
                    Y_te = torch.tensor(b_target[idx_te], device=device, dtype=torch.float32)
                    targets[target_label] = (Y_tr, Y_te)

                r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                for target_label, r2 in r2s.items():
                    gt_rows.append({
                        "hmm": hmm_name,
                        "source": source_label,
                        "target": target_label,
                        "seed": seed,
                        "R2": r2,
                        "self": source_label == target_label,
                    })

                pbar.update(1)
        pbar.close()

    # Write transfer_probes_gt_<model>.csv.
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(os.path.join(args.output_dir, f"transfer_probes_gt_{ms}.csv"), index=False)
    print(f"Done. Cross: {len(cross_rows)} rows, GT: {len(gt_rows)} rows.")

if __name__ == "__main__":
    main()
