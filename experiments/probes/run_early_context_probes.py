"""Belief-state probes in the early context: probes fit within the first 5,000 tokens and evaluated on the first 250,
plus the late-window probe transferred to the early positions.

Paper section: Section 5.2, the end of the paragraph "Belief states are linearly decodable from residual stream
activations", and Appendix D with Figure 15.

Claim: In the paper's words, "we find that probes fit to the first 5,000 token positions recover visibly coarser
geometries with lower R² when evaluated at very early windows of the sequence (Figure 15), consistent with the
model's in-context prediction accuracy being worse early in-context (see Figure 2)."

Experiment: In the paper's words, "In Figure 15, we visualize the geometry from a linear probe trained on 1,000 of
the first 5,000 tokens, evaluated on its held-out positions among the first 250 tokens, to the belief state. We
evaluate on the first 250 tokens only, because the model's in-context prediction accuracy almost fully converges
after the first few hundred tokens (Figure 2)." The probes have the same size as those of Figure 11, and only the
positions on which they are evaluated differ.

Result: The early-context geometries are visibly coarser and their R² lower than those of the late-window probes
of Figure 11.

How the code works: For each target parametrization and seed, the sequence, its belief states, and its tokenization
are exactly those of the other probe scripts. The residual stream is extracted for the early positions 0 to
early_len - 1 and, unless --no_late is given, for the late window as well, either at every layer or only at the
parametrization's best late layer taken from --r2_csv. For each window W:n_train in --windows, a probe is fit on
n_train random positions among the first W (train_test_split(random_state=seed), pseudo-inverse with a ones column)
and its predictions on the remaining W - n_train positions are stored together with their indices, for the true
belief states and for shuffled and random labels. The late-window probe of Section 5 (fit on the seeded 20 percent
split of the late window) is also applied to every early position and its predictions stored, with its predictions
on the late test split as a reference. Everything is stored as predictions and labels in
early_context_<model>__<family>__<label>.npz, and the notebook computes R² on the positions it draws (the first
250 for Figure 15).
"""
import argparse, gc, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

# The four main-text parametrizations. --all_params replaces them by every parametrization of the four families.
DEFAULT_TARGETS = ["Mess3=a=0.01, x=0.02", "Arch=a=0.99", "Wing=a=0.98, x=0.4", "Strata=a=0.97, t0=0.38, t1=0.54"]

# Append a ones column so that the least-squares probe has a bias term.
def _aug(X):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], dim=1)

def main():
    # Command-line options. The paper's Figure 15 uses --early_len 5000 --windows 5000:1000 --layers best with the
    # r2_<model>.csv of the belief probes; the defaults for sequences, seeds, and split match the other probe scripts.
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--early_len", type=int, default=4096)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--windows", default="5000:1000", help="comma list of W:n_train (paper: 5000:1000)")
    P.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="Family=label")
    P.add_argument("--all_params", action="store_true", help="all parametrizations of the four families (overrides --targets)")
    P.add_argument("--r2_csv", default=None, help="r2_{model}.csv; best layer per target for the acts dump")
    P.add_argument("--output_dir", required=True)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true")
    P.add_argument("--layers", default="all", choices=["all", "best"], help="all layers, or only the late best layer (needs --r2_csv)")
    P.add_argument("--no_late", action="store_true", help="skip the late window entirely (window probes only)")
    P.add_argument("--no_acts", action="store_true", help="do not dump early activations")
    args = P.parse_args()

    # Smoke mode is a plumbing test on short sequences with one seed and one target.
    if args.smoke:
        args.seq_len, args.probe_start, args.early_len, args.n_seeds = 3000, 2000, 512, 1
        args.windows, args.targets = "200:40", args.targets[:1]
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    windows = [(int(w.split(":")[0]), int(w.split(":")[1])) for w in args.windows.split(",")]
    r2 = pd.read_csv(args.r2_csv) if args.r2_csv else None

    # Load the model once.
    wrapper, tokenizer = load_model(args.model, device)
    all_layers = list(range(wrapper.n_layers))

    targets = args.targets
    if args.all_params:
        targets = [f"{fam}={HMMS[fam]['label_fn'](p)}" for fam in ["Mess3", "Arch", "Wing", "Strata"] for p in HMMS[fam]["params"]]

    # One output file per target parametrization. An existing file is skipped, so an interrupted run can be resumed.
    for spec in targets:
        fam, label = spec.split("=", 1)
        cfg = HMMS[fam]
        k = cfg["n_states"]
        param = next(p for p in cfg["params"] if cfg["label_fn"](p) == label)
        out_fn = os.path.join(args.output_dir, f"early_context_{ms}__{fam}__{label}.npz")
        if os.path.exists(out_fn): print(f"skip existing {out_fn}", flush=True); continue

        # The parametrization's best late layer is the layer whose seed-mean real-target R² in r2_<model>.csv is highest.
        best_layer = None
        if r2 is not None:
            sub = r2[(r2.hmm == fam) & (r2.param == label) & (r2.target == "real")]
            if len(sub): best_layer = int(sub.groupby("layer")["R2"].mean().idxmax())
        layers = all_layers if args.layers == "all" else [best_layer]
        L = len(layers)
        assert not (args.layers == "best" and best_layer is None), "--layers best needs --r2_csv with this target"
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])

        # The HMM, the early window length E, and a store of per-seed arrays keyed by name.
        T = cfg["fn"](*param)
        T_stack = np.stack(T)
        pi = stationary_distribution(T)
        E = min(args.early_len, args.probe_start)
        store = {}
        def put(key, seed, arr):
            store.setdefault(key, {})[seed] = arr
        t0 = time.time()

        # One sequence per seed, sampled and tokenized exactly as in the other probe scripts.
        for seed in range(args.n_seeds):
            tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
            beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi).astype(np.float32)
            prompt = tokens_to_prompt(tokens, cfg["token_names"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
            pos_indices, _ = match_positions(input_ids, tok_ids)
            n_matched = min(len(tokens), len(pos_indices))
            if n_matched < args.probe_start + 100 or n_matched < E:
                print(f"WARNING {fam} {label} s{seed}: n_matched={n_matched}; skipped", flush=True)
                continue

            # Positions and labels: the early window, the late window, and the shuffled and Dirichlet control labels for
            # each (independent rngs for the early labels).
            early_idx = np.arange(0, E)
            late_idx = np.arange(args.probe_start, n_matched)
            y_early = beliefs[early_idx]
            y_late = beliefs[late_idx]
            n_late = len(y_late)
            rng_sh = np.random.default_rng(seed + 77777)
            y_late_sh = y_late[rng_sh.permutation(n_late)]
            rng_rd = np.random.default_rng(seed + 88888)
            y_late_rd = rng_rd.dirichlet(np.ones(k), size=n_late).astype(np.float32)
            rng_sh_e = np.random.default_rng(seed + 177777)
            y_early_sh = y_early[rng_sh_e.permutation(E)]
            rng_rd_e = np.random.default_rng(seed + 188888)
            y_early_rd = rng_rd_e.dirichlet(np.ones(k), size=E).astype(np.float32)

            # One chunked forward pass; the residual stream at the selected layers is kept for the early positions and,
            # unless --no_late, the late positions (early rows first).
            positions = pos_indices[early_idx] if args.no_late else np.concatenate([pos_indices[early_idx], pos_indices[late_idx]])
            acts, _ = extract_activations_chunked(wrapper, input_ids, layers, positions, args.chunk_size, device)

            # The late-window split of Section 5, seeded by the sequence seed, and the labels stored for scoring.
            idx_tr, idx_te = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=seed)
            idx_te = np.sort(idx_te)
            put("idx_te_late", seed, idx_te.astype(np.int32))
            put("n_matched", seed, np.int64(n_matched))
            put("true_early_real", seed, y_early)
            put("true_early_shuffle", seed, y_early_sh)
            put("true_early_random", seed, y_early_rd)
            if not args.no_late:
                put("true_late_te_real", seed, y_late[idx_te])
                put("true_late_te_shuffle", seed, y_late_sh[idx_te])
                put("true_late_te_random", seed, y_late_rd[idx_te])
            late_tr = {"real": y_late[idx_tr], "shuffle": y_late_sh[idx_tr], "random": y_late_rd[idx_tr]}
            early_all = {"real": y_early, "shuffle": y_early_sh, "random": y_early_rd}

            # Window splits: n_train random positions among the first W train the window probe and the rest are held out.
            win_idx = {}
            for (Wn, ntr) in windows:
                jtr, jte = train_test_split(np.arange(min(Wn, E)), train_size=ntr, random_state=seed)
                win_idx[Wn] = (jtr, np.sort(jte))
                put(f"w{Wn}_idx_te", seed, np.sort(jte).astype(np.int32))

            # Prediction arrays in float16: the late probe on the late test split and on every early position, and each window
            # probe on its held-out positions.
            pred_late = {t: np.zeros((L, len(idx_te), k), np.float16) for t in late_tr}
            pred_early = {t: np.zeros((L, E, k), np.float16) for t in late_tr}
            pred_win = {(Wn, t): np.zeros((L, len(win_idx[Wn][1]), k), np.float16) for (Wn, _) in windows for t in late_tr}

            # Per layer: the late probe (one pseudo-inverse shared by the three label kinds) transferred verbatim to the early
            # positions, then one probe per window fit inside the early context.
            for li, l in enumerate(layers):
                X = acts[l]
                Xe, Xl = X[:E], X[E:]
                if not args.no_late:
                    Pinv = torch.linalg.pinv(_aug(Xl[idx_tr]))
                    Ae_all = _aug(Xe)
                    Al_te = _aug(Xl[idx_te])
                    for t, ytr in late_tr.items():
                        Wt = Pinv @ torch.tensor(ytr, device=device, dtype=torch.float32)
                        pred_late[t][li] = (Al_te @ Wt).half().cpu().numpy()
                        pred_early[t][li] = (Ae_all @ Wt).half().cpu().numpy()
                    del Pinv
                for (Wn, ntr) in windows:
                    jtr, jte = win_idx[Wn]
                    Pw = torch.linalg.pinv(_aug(Xe[jtr]))
                    Aw_te = _aug(Xe[jte])
                    for t, y in early_all.items():
                        Ww = Pw @ torch.tensor(y[jtr], device=device, dtype=torch.float32)
                        pred_win[(Wn, t)][li] = (Aw_te @ Ww).half().cpu().numpy()
                    del Pw

            # Move the predictions of this seed into the store.
            if not args.no_late:
                for t in late_tr:
                    put(f"pred_late_te_{t}", seed, pred_late[t])
                    put(f"pred_early_{t}", seed, pred_early[t])
            for (Wn, t), arr in pred_win.items(): put(f"w{Wn}_pred_{t}", seed, arr)

            # Optionally dump the early activations at the best layer for offline analysis (not used by the paper's figures).
            if best_layer is not None and not args.no_acts:
                np.savez(os.path.join(args.output_dir, f"acts_early_{ms}__{fam}__{label}__s{seed}.npz"),
                         acts=acts[best_layer][:E].half().cpu().numpy(), beliefs=y_early, layer=best_layer, positions=early_idx)
            del acts
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{fam} [{label}] s{seed}: n_matched={n_matched} done ({time.time()-t0:.0f}s)", flush=True)

        # Stack every stored array over the seeds (truncating to the shortest seed if the lengths differ) and write the
        # file.
        out = {"layers": np.array(layers), "seeds": np.array(sorted(store["idx_te_late"])), "E": np.int64(E),
               "probe_start": np.int64(args.probe_start), "windows": np.array(windows), "best_layer": np.int64(-1 if best_layer is None else best_layer)}
        for key, d in store.items():
            seeds = sorted(d)
            arrs = [d[s] for s in seeds]
            if np.ndim(arrs[0]) == 0: out[key] = np.array(arrs); continue
            m = min(a.shape[-2] if a.ndim >= 2 else a.shape[0] for a in arrs)
            if any((a.shape[-2] if a.ndim >= 2 else a.shape[0]) != m for a in arrs): print(f"WARNING {key}: unequal sizes, truncating to {m}", flush=True)
            out[key] = np.stack([a[..., :m, :] if a.ndim >= 2 else a[:m] for a in arrs])
        np.savez(out_fn, **out)
        print(f"saved {out_fn} ({len(out)} arrays)", flush=True)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
