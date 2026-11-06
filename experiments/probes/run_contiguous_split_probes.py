"""Belief-state probes under a contiguous train/test split of the late window.

Paper section: Section 5.1 (the sentence on contiguous-split and held-out sequence evaluations) and Appendix F with
Table 3.

Claim: In the paper's words, "For Mess3, Wing, and Strata, the three protocols agree to within 0.03 for all LLMs, and
the best layer remains unchanged", whereas for Arch "The peak R² for this process decreases monotonically with its
parameter α under both protocols".

Experiment: In the paper's words, "In the contiguous split protocol, the probe is fit on the first 20% of the window
(positions 15,000–16,000) and evaluated on the remaining 80% (16,000–20,000)." Everything else follows
run_belief_probes.py: the same sequences and seeds, the same late window (positions 15,000 onward), the same belief
targets, and the same float32 pseudo-inverse fit with a bias term. The training count matches sklearn's floor(0.2 n).

Saved Outputs: contiguous_split_probes_<model>.csv in --output_dir, with the columns hmm, param, layer, seed, and R2,
one row per layer and seed of every parametrization. The script resumes from an existing file on (hmm, param, seed).

How the code works: The model is loaded once. For each parametrization and seed, the sequence is sampled, its belief
states are computed, and the late-window activations of every layer are extracted and staged on the CPU. The probes
of all layers are fit with one batched pseudo-inverse on the first 20 percent of the window and scored by R² on the
remaining positions in chunks, so that a 32 GB card holds the 9B model. --selftest checks the batched fit against
fit_and_evaluate_multi on synthetic data and exits.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_and_evaluate_multi


def batched_fit_eval(X_cpu, idx_tr, idx_te, y, device, te_chunk=2048):
    """Batched-across-layers OLS with bias (float32 pinv), R² on test rows.
    X_cpu: (L, n, d) float32 on CPU; y: (n, m) numpy. Returns [R² per layer]."""
    L = X_cpu.shape[0]
    Xa_tr = torch.cat([X_cpu[:, idx_tr].to(device),
                       torch.ones(L, len(idx_tr), 1, device=device)], dim=2)
    P = torch.linalg.pinv(Xa_tr)                                   # (L, d+1, n_tr)
    Y_tr = torch.tensor(y[idx_tr], device=device, dtype=torch.float32)
    W = P @ Y_tr                                                   # (L, d+1, m)
    del Xa_tr, P
    Y_te_full = torch.tensor(y[idx_te], device=device, dtype=torch.float32)
    ss_tot = ((Y_te_full - Y_te_full.mean(dim=0)) ** 2).sum()
    ss_res = torch.zeros(L, device=device)
    for s in range(0, len(idx_te), te_chunk):
        sl = idx_te[s:s + te_chunk]
        Xa = torch.cat([X_cpu[:, sl].to(device),
                        torch.ones(L, len(sl), 1, device=device)], dim=2)
        pred = Xa @ W                                              # (L, c, m)
        Y_c = torch.tensor(y[sl], device=device, dtype=torch.float32)
        ss_res += ((Y_c - pred) ** 2).sum(dim=(1, 2))
        del Xa, pred, Y_c
    if float(ss_tot) <= 0:
        return np.zeros(L)
    return (1.0 - ss_res / ss_tot).cpu().numpy()


def selftest(device):
    """Batched fit must reproduce probes.fit_and_evaluate_multi per layer."""
    g = torch.Generator().manual_seed(0)
    L, n, d, m = 4, 400, 64, 3
    X = torch.randn(L, n, d, generator=g)
    W = torch.randn(L, d, m, generator=g)
    Y = (X @ W + 0.1 * torch.randn(L, n, m, generator=g))[0].numpy().astype(np.float32)
    idx_tr, idx_te = np.arange(80), np.arange(80, n)
    got = batched_fit_eval(X.float(), idx_tr, idx_te, Y, device, te_chunk=100)
    for l in range(L):
        tg = {"real": (torch.tensor(Y[idx_tr], device=device, dtype=torch.float32),
                       torch.tensor(Y[idx_te], device=device, dtype=torch.float32))}
        ref = fit_and_evaluate_multi(X[l, idx_tr].float().to(device),
                                     X[l, idx_te].float().to(device), tg, use_bias=True)["real"]
        assert abs(got[l] - ref) < 1e-3, f"layer {l}: batched {got[l]:.6f} vs ref {ref:.6f}"
    print("selftest OK: batched fit matches fit_and_evaluate_multi", flush=True)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--device", default="cuda")
    P.add_argument("--selftest", action="store_true")
    args = P.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.selftest:
        selftest(device); return
    from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                                 get_tok_ids, extract_activations_chunked)
    os.makedirs(args.output_dir, exist_ok=True)
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_csv = os.path.join(args.output_dir, f"contiguous_split_probes_{ms}.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(map(tuple, prev[["hmm", "param", "seed"]].drop_duplicates().values))
        print(f"resuming: {len(done)} (hmm,param,seed) already done", flush=True)

    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))

    header_needed = not os.path.exists(out_csv)
    for hmm_name in args.families:
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            for seed in range(args.n_seeds):
                if (hmm_name, label, seed) in done: continue
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                y_real = beliefs[args.probe_start:n_matched].astype(np.float32)
                n_late = len(y_real)
                late_pos = pos_indices[args.probe_start:n_matched]
                if n_late == 0: continue
                n_tr = int(args.train_frac * n_late)               # sklearn's floor count
                idx_tr, idx_te = np.arange(n_tr), np.arange(n_tr, n_late)

                acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos,
                                                      args.chunk_size, device)
                d_model = acts[layers[0]].shape[1]
                X_cpu = torch.empty(len(layers), n_late, d_model, dtype=torch.float32)
                for i, l in enumerate(layers):
                    X_cpu[i] = acts[l].float().cpu(); acts[l] = None
                del acts; gc.collect(); torch.cuda.empty_cache()

                per_layer = batched_fit_eval(X_cpu, idx_tr, idx_te, y_real, device)
                rows = [{"hmm": hmm_name, "param": label, "layer": l, "seed": seed,
                         "R2": float(per_layer[i])} for i, l in enumerate(layers)]
                pd.DataFrame(rows).to_csv(out_csv, mode="a", header=header_needed, index=False)
                header_needed = False
                del X_cpu; gc.collect(); torch.cuda.empty_cache()
                print(f"{hmm_name} [{label}] s{seed}: best contig R2 {max(per_layer):.3f} "
                      f"(L{int(np.argmax(per_layer))})", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
