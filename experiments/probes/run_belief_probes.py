"""Belief-state probes at every layer with shuffled and random controls, and the decoded belief geometries.

Paper section: Section 5.1 (Method) and Section 5.2, the paragraph "Belief states are linearly decodable from residual
stream activations", with Figure 3 and the middle panel of Figure 1 (b), Appendix D with Figures 11 to 14, and
Appendix I with Figures 29 to 34.

Claim: "Belief states are linearly decodable from residual stream activations." In the paper's words, "linear probes
achieve high R² values in early-to-middle layers and remain high thereafter. Across all HMM families and all LLMs, the
peak per-HMM R² ranges from 0.83 to 0.99", and "Two control probes cannot explain these high R² values."

Experiment: In the paper's words, "For each input sequence x_1:N, we extract residual stream activations at every
token position t well after convergence (t >= 15,000) and every layer. At each layer, we fit linear regression via
ordinary least squares from the activations to the corresponding belief state, training on a random 20% of positions
(1,000 tokens) and evaluating on the held-out 80% (4,000 tokens). Probes are fit per-sequence." The shuffled-belief
control permutes the belief states across positions and the random-belief control replaces them with draws from a
symmetric Dirichlet distribution, both with the same activations and the same split.

Result: The peak per-HMM R² ranges from 0.83 to 0.99 across all families and LLMs, while both controls stay near
zero at every layer. The geometries decoded at each parametrization's best layer reproduce the ground-truth belief
geometries (Figures 3, 11, and 12), and the control probes recover no structure (Figures 13 and 14).

How the code works: Pass 1 runs for every parametrization and seed. It samples the sequence, computes the exact
belief states, tokenizes the sequence and matches HMM tokens to model positions, builds the shuffled labels (rng
seed + 77777) and the Dirichlet labels (rng seed + 88888) on the late window, extracts the activations at every layer
in one chunked forward pass, splits the late positions with train_test_split(random_state=seed), and at every layer
fits the three probes with one shared pseudo-inverse (ones column for the bias) and scores them on the held-out
positions. The rows (hmm, param, layer, seed, target in real/shuffle/random, R2) go to r2_<model>.csv. Pass 2 takes
each parametrization's best layer, the layer whose real-target R² averaged over the seeds is highest, and runs one
more forward pass per sequence at that layer only. It stores an in-sample geometry (probe fit in float64 on all late
positions, predictions on those same positions) in geompool_insample_<model>.npz, with sequence 0 also written to
geom_<model>.npz, and the held-out geometry of the pass-1 protocol (float32, seeded 20 percent split, predictions on
the held-out positions) pooled over the sequences in geompool_all_<model>.npz, together with the same held-out
protocol for the shuffled and random controls in geompool_ctl_<model>.npz.
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

# Coefficient of determination in float64, pooled over all output dimensions (the definition of
# src.metrics.probes.compute_r2).
def _r2_np(Y, P):
    Y = np.asarray(Y, np.float64)
    P = np.asarray(P, np.float64)
    return float(1.0 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

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
    P.add_argument("--params", nargs="+", default=None, help="explicit parametrization labels (default: all)")
    P.add_argument("--skip_geometry", action="store_true", help="pass 1 only")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load the model once. Every layer is probed in pass 1.
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")

    # Pass 1: at every layer, probes to the true belief states and to the two control targets, for every
    # parametrization and seed.
    all_rows = []

    # Loop over the requested families (all of them by default) and their parametrizations (optionally restricted to
    # explicit labels).
    families = args.families or list(HMMS.keys())
    for hmm_name in families:
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        params = [p for p in cfg["params"] if cfg["label_fn"](p) in args.params] if args.params else cfg["params"]
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        pbar = tqdm(total=len(params)*args.n_seeds, desc=hmm_name)
        for param in params:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param)
            T_stack = np.stack(T)
            pi = stationary_distribution(T)
            # One sequence per seed. Sampling is seeded, so every script in the repository sees the same sequence for a
            # given parametrization and seed.
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)

                # Write the sequence as space-separated letters, locate the model token carrying each HMM token, and keep the
                # late window of belief states as the probe target.
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                y_real = beliefs[args.probe_start:n_matched]
                n_late = len(y_real)
                late_pos = pos_indices[args.probe_start:n_matched]
                if n_late == 0: pbar.update(1); continue

                # Control labels on the same activations: the shuffled control permutes the belief states across positions
                # (rng seed + 77777) and the random control draws them from a symmetric Dirichlet distribution (rng seed + 88888).
                rng_sh = np.random.default_rng(seed+77777)
                y_shuffle = y_real[rng_sh.permutation(n_late)]
                rng_rd = np.random.default_rng(seed+88888)
                y_random = rng_rd.dirichlet(np.ones(cfg["n_states"]), size=n_late)

                # One chunked forward pass with KV caching; the residual stream at every layer is kept for the late positions only.
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)

                # The paper's split (Section 5.1): a random 20 percent of the late positions train the probes and the remaining
                # 80 percent score them, seeded by the sequence seed.
                idx_tr, idx_te = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=seed)
                tgts_np = {"real": y_real, "shuffle": y_shuffle, "random": y_random}

                # At every layer one pseudo-inverse of the augmented training activations is shared by the three targets, and
                # each target's held-out R² is recorded.
                for l in layers:
                    X = acts[l]
                    if X.numel() == 0: continue
                    tgts = {t: (torch.tensor(y[idx_tr],device=device,dtype=torch.float32),
                               torch.tensor(y[idx_te],device=device,dtype=torch.float32)) for t,y in tgts_np.items()}
                    r2s = fit_and_evaluate_multi(X[idx_tr], X[idx_te], tgts, use_bias=True)
                    for t, r2 in r2s.items():
                        all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":t,"R2":r2})

                del acts
                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)
        pbar.close()

        # Write r2_<model>.csv after each family (rewritten with all rows accumulated so far).
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, f"r2_{ms}.csv"), index=False)
    if args.skip_geometry:
        print("Done (pass 1 only).")
        return

    # Pass 2: the decoded geometries at each parametrization's best layer, from one more forward pass per sequence.
    r2_df = pd.DataFrame(all_rows)
    geom, ins, pooled, ctl = {}, {}, {}, {}
    for hmm_name in families:
        cfg = HMMS.get(hmm_name)
        if not cfg or hmm_name not in r2_df["hmm"].values: continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        params = [p for p in cfg["params"] if cfg["label_fn"](p) in args.params] if args.params else cfg["params"]
        for param in params:
            label = cfg["label_fn"](param)
            key = f"{hmm_name}__{label}"

            # The best layer of this parametrization: the layer whose real-target R² averaged over the seeds is highest,
            # which is the layer shown in Figures 3 and 11 to 15.
            sub = r2_df[(r2_df["hmm"]==hmm_name)&(r2_df["param"]==label)&(r2_df["target"]=="real")]
            if len(sub)==0: continue
            layer_means = sub.groupby("layer")["R2"].mean()
            best_layer = int(layer_means.idxmax())
            best_r2 = float(layer_means.max())
            T = cfg["fn"](*param)
            T_stack = np.stack(T)
            pi = stationary_distribution(T)

            # Accumulators over the sequences: in-sample predictions (ins), held-out predictions of the pass-1 protocol (pl),
            # and the two controls (ct).
            ins_p, ins_t, ins_s, ins_r = [], [], [], []
            pl_p, pl_t, pl_s, pl_r = [], [], [], []
            ct = {c: ([], [], []) for c in ("shuffle", "random")}

            # One forward pass per sequence, at the best layer only.
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                late_pos = pos_indices[args.probe_start:n_matched]
                y_true = beliefs[args.probe_start:n_matched]
                n_late = len(y_true)
                acts, _ = extract_activations_chunked(wrapper, input_ids, [best_layer], late_pos, args.chunk_size, device)
                X = acts[best_layer]

                # In-sample geometry: the probe is fit in float64 on all late positions and its predictions on those same
                # positions are kept (Figure 12). Sequence 0 also provides geom_<model>.npz, whose true beliefs define the
                # PCA frame of the geometry grids in the notebook.
                W = fit_probe(X.double(), torch.tensor(y_true, device=device, dtype=torch.float64), use_bias=True)
                y_pred = predict_probe(X.double(), W, use_bias=True).cpu().numpy()
                if seed == 0:
                    geom[key] = {"true": y_true, "pred": y_pred, "best_layer": best_layer, "r2": best_r2, "param": label}
                ins_p.append(y_pred.astype(np.float32))
                ins_t.append(y_true.astype(np.float32))
                ins_s.append(np.full(n_late, seed))
                ins_r.append(_r2_np(y_true, y_pred))

                # Held-out geometry with exactly the pass-1 protocol: float32, the seeded 20 percent training split, and
                # predictions on the held-out 80 percent (Figures 3 and 11). The shuffled and random controls use the same
                # rngs as pass 1 (Figures 13 and 14).
                idx_tr, idx_te = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=seed)
                Xtr, Xte = X[idx_tr].float(), X[idx_te].float()
                Wf = fit_probe(Xtr, torch.tensor(y_true[idx_tr], device=device, dtype=torch.float32), use_bias=True)
                pf = predict_probe(Xte, Wf, use_bias=True).cpu().numpy()
                pl_p.append(pf)
                pl_t.append(y_true[idx_te].astype(np.float32))
                pl_s.append(np.full(len(idx_te), seed))
                pl_r.append(_r2_np(y_true[idx_te], pf))
                rng_sh = np.random.default_rng(seed+77777)
                y_shuffle = y_true[rng_sh.permutation(n_late)]
                rng_rd = np.random.default_rng(seed+88888)
                y_random = rng_rd.dirichlet(np.ones(cfg["n_states"]), size=n_late)
                for cname, y in (("shuffle", y_shuffle), ("random", y_random)):
                    Wc = fit_probe(Xtr, torch.tensor(y[idx_tr], device=device, dtype=torch.float32), use_bias=True)
                    pc = predict_probe(Xte, Wc, use_bias=True).cpu().numpy()
                    ct[cname][0].append(pc)
                    ct[cname][1].append(y[idx_te].astype(np.float32))
                    ct[cname][2].append(_r2_np(y[idx_te], pc))
                del acts, X
                gc.collect()
                torch.cuda.empty_cache()

            ins[key] = {"pred": np.concatenate(ins_p), "true": np.concatenate(ins_t), "seed": np.concatenate(ins_s), "r2_seeds": np.array(ins_r)}
            pooled[key] = {"pred": np.concatenate(pl_p), "true": np.concatenate(pl_t), "seed": np.concatenate(pl_s), "r2_seeds": np.array(pl_r), "layers": np.full(args.n_seeds, best_layer)}
            for cname in ct:
                ctl[f"{key}_{cname}"] = {"pred": np.concatenate(ct[cname][0]), "true": np.concatenate(ct[cname][1]), "r2_seeds": np.array(ct[cname][2])}
            print(f"  {hmm_name} [{label}] geom: layer {best_layer}, R²={best_r2:.4f}; held-out per-seq mean {np.mean(pl_r):.4f}, in-sample {np.mean(ins_r):.4f}", flush=True)

    # Write the four geometry files. Every key has the form <family>__<label>_<quantity>.
    np.savez(os.path.join(args.output_dir, f"geom_{ms}.npz"),
             **{f"{k}_{v}": (np.array(geom[k][v]) if not isinstance(geom[k][v], np.ndarray) else geom[k][v])
                for k in geom for v in ["true", "pred", "best_layer", "r2", "param"]})
    np.savez(os.path.join(args.output_dir, f"geompool_insample_{ms}.npz"), **{f"{k}_{v}": d[v] for k, d in ins.items() for v in d})
    np.savez(os.path.join(args.output_dir, f"geompool_all_{ms}.npz"), **{f"{k}_{v}": d[v] for k, d in pooled.items() for v in d})
    np.savez(os.path.join(args.output_dir, f"geompool_ctl_{ms}.npz"), **{f"{k}_{v}": d[v] for k, d in ctl.items() for v in d})
    print("Done.")

if __name__ == "__main__": main()
