"""Belief-state probes evaluated on held-out sequences: fit on one sequence, scored on the next.

Paper section: Section 5.1 (the sentence on contiguous-split and held-out sequence evaluations) and Appendix F with
Table 3.

Claim: In the paper's words, "For Mess3, Wing, and Strata, the three protocols agree to within 0.03 for all LLMs, and
the best layer remains unchanged", whereas for Arch "The peak R² for this process decreases monotonically with its
parameter α under both protocols".

Experiment: In the paper's words, "In the held-out sequence protocol, the probe is fit on the training positions of one
sequence and evaluated on all 5,000 window positions of a different sequence from the same HMM, cycling over the 10
sequences so that each serves once as the test sequence." The training positions are those of run_belief_probes.py
(train_test_split with train_size 0.2 and random_state equal to the training seed), and the fit is the same float32
pseudo-inverse with a bias term.

Saved Outputs: heldout_sequence_probes_<model>.csv in --output_dir, with the columns hmm, param, layer, train_seed,
test_seed, and R2, one row per layer and per (train, test) pair of every parametrization. The script resumes from an
existing file on (hmm, param).

How the code works: The model is loaded once. For each parametrization, every seed's sequence is run once and its
late-window activations of all layers are kept on the CPU in float16. Each consecutive pair of seeds (seed s trains
and seed s + 1 tests, with the last seed training for seed 0) fits the probes of all layers with one batched
pseudo-inverse on the training positions of the training sequence and scores them on all window positions of the
test sequence.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs


def batched_fit_eval(Xtr_cpu, ytr, Xte_cpu, yte, device, te_chunk=2048):
    """(L, n_tr, d) x (n_tr, m) -> R2 per layer on (L, n_te, d) x (n_te, m)."""
    L = Xtr_cpu.shape[0]
    Xa_tr = torch.cat([Xtr_cpu.to(device).float(),
                       torch.ones(L, Xtr_cpu.shape[1], 1, device=device)], dim=2)
    W = torch.linalg.pinv(Xa_tr) @ torch.tensor(ytr, device=device, dtype=torch.float32)
    del Xa_tr
    Y_te = torch.tensor(yte, device=device, dtype=torch.float32)
    ss_tot = ((Y_te - Y_te.mean(dim=0)) ** 2).sum()
    ss_res = torch.zeros(L, device=device)
    n_te = Xte_cpu.shape[1]
    for s in range(0, n_te, te_chunk):
        e = min(s + te_chunk, n_te)
        Xa = torch.cat([Xte_cpu[:, s:e].to(device).float(),
                        torch.ones(L, e - s, 1, device=device)], dim=2)
        ss_res += ((Y_te[s:e] - Xa @ W) ** 2).sum(dim=(1, 2))
        del Xa
    if float(ss_tot) <= 0: return np.zeros(L)
    return (1.0 - ss_res / ss_tot).cpu().numpy()


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
    args = P.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                                 get_tok_ids, extract_activations_chunked)
    os.makedirs(args.output_dir, exist_ok=True)
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_csv = os.path.join(args.output_dir, f"heldout_sequence_probes_{ms}.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(map(tuple, prev[["hmm", "param"]].drop_duplicates().values))
        print(f"resuming: {len(done)} (hmm,param) done", flush=True)

    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    header_needed = not os.path.exists(out_csv)

    def get_acts(cfg, tok_ids, T, T_stack, pi, seed):
        tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
        beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
        prompt = tokens_to_prompt(tokens, cfg["token_names"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
        pos_indices, _ = match_positions(input_ids, tok_ids)
        n_matched = min(len(tokens), len(pos_indices))
        late_pos = pos_indices[args.probe_start:n_matched]
        y = beliefs[args.probe_start:n_matched].astype(np.float32)
        acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos,
                                              args.chunk_size, device)
        X = torch.empty(len(layers), len(y), acts[layers[0]].shape[1], dtype=torch.float16)
        for i, l in enumerate(layers):
            X[i] = acts[l].half().cpu(); acts[l] = None
        del acts; gc.collect(); torch.cuda.empty_cache()
        return X, y

    for hmm_name in args.families:
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            if (hmm_name, label) in done: continue
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            rows = []
            X0 = y0 = Xp = yp = None
            for seed in range(args.n_seeds):
                X, y = get_acts(cfg, tok_ids, T, T_stack, pi, seed)
                if seed == 0:
                    X0, y0 = X, y
                else:
                    itr, _ = train_test_split(np.arange(len(yp)), train_size=args.train_frac,
                                              random_state=seed - 1)
                    r2s = batched_fit_eval(Xp[:, itr], yp[itr], X, y, device)
                    rows += [{"hmm": hmm_name, "param": label, "layer": l,
                              "train_seed": seed - 1, "test_seed": seed, "R2": float(r2s[i])}
                             for i, l in enumerate(layers)]
                    del Xp
                Xp, yp = X, y
            itr, _ = train_test_split(np.arange(len(yp)), train_size=args.train_frac,
                                      random_state=args.n_seeds - 1)
            r2s = batched_fit_eval(Xp[:, itr], yp[itr], X0, y0, device)
            rows += [{"hmm": hmm_name, "param": label, "layer": l,
                      "train_seed": args.n_seeds - 1, "test_seed": 0, "R2": float(r2s[i])}
                     for i, l in enumerate(layers)]
            del Xp, X0; gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=header_needed, index=False)
            header_needed = False
            best = pd.DataFrame(rows).groupby("layer")["R2"].mean()
            print(f"{hmm_name} [{label}]: best cross R2 {best.max():.3f} (L{int(best.idxmax())})",
                  flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
