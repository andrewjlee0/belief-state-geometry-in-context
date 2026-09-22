"""Next-token probability (NTP) and log-NTP probes, and the model-free NTP-to-belief baselines.

Paper section: Section 5.2, the paragraphs "Belief state decodability cannot be explained by next-token probability or
log next-token probability representations" and "NTP and log-NTP are linearly decodable from residual stream
activations", with Figure 5 (bottom), and Appendix L with Figures 47 to 52.

Claim: "Belief state decodability cannot be explained by next-token probability or log next-token probability
representations." In the paper's words, "If the R² from activations exceeds the R² from the ground truth NTP/log-NTP,
then the activations carry belief state information that cannot be explained by these candidate objects alone."
A second claim of the same section is that "NTP and log-NTP are linearly decodable from residual stream activations."

Experiment: For every parametrization and seed, the late window of the sequence (positions 15,000 onward) provides
activations at every layer together with the ground-truth belief states, the NTP p = eta M of Equation 5, and
log-NTP. Three probes per layer are fit with the paper's interleaved 20 percent training split and scored by
held-out R²: activations to beliefs, activations to NTP, and activations to log-NTP. Two model-free baselines per
parametrization quantify how much belief information NTP and log-NTP themselves carry: a linear regression from
the ground-truth NTP to the belief state and one from the ground-truth log-NTP to the belief state, fit on the late
windows of all 10 sequences pooled together.

Saved Outputs: ntp_probes_<model>.csv with the columns hmm, param, layer, seed, target, and R2, where target is one of
act→beliefs, act→ntp, act→log_ntp, ntp→beliefs, and log_ntp→beliefs. The last two are the pooled model-free baselines,
repeated in every layer and seed row of their parametrization.

How the code works: The model is loaded once. For each parametrization, a first loop over the seeds computes the
belief states and NTP without the model, pools their late windows, and fits the two baselines in closed form by
pseudo-inverse with a ones column for the bias, recording their in-sample R². A second loop over the seeds samples
the same sequences, runs the model once per sequence, extracts the late-window activations at every layer, splits
the positions with train_test_split(random_state=seed), and at every layer fits the three probes with one shared
pseudo-inverse and scores them on the held-out positions. Every row of ntp_probes_<model>.csv carries hmm, param,
layer, seed, target (act->beliefs, act->ntp, act->log_ntp, ntp->beliefs, or log_ntp->beliefs), and R2. The two
baseline values are repeated in every (layer, seed) row so that the notebook can draw them as constant lines.
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.metrics.probes import fit_and_evaluate_multi
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
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")
    all_rows = []

    # Loop over the requested families (all of them by default) and every parametrization of each family.
    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        pbar = tqdm(total=len(cfg["params"])*args.n_seeds, desc=hmm_name)

        for param in cfg["params"]:
            label = cfg["label_fn"](param)

            # The HMM: per-token transition matrices, their stacked array, and the stationary distribution used as the
            # initial belief.
            T = cfg["fn"](*param)
            T_stack = np.stack(T)
            pi = stationary_distribution(T)

            # Model-free baselines. The late windows of all seeds are pooled, and a linear regression with bias from the
            # ground-truth NTP (and separately from log-NTP) to the belief state is fit and scored in-sample. These are
            # the constant lines of Figure 5 (bottom). For Mess3 the emission matrix is invertible, so the NTP baseline
            # reaches R² = 1.
            all_b, all_ntp = [], []
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp = next_token_probs(beliefs, T)
                all_b.append(beliefs[args.probe_start:])
                all_ntp.append(ntp[args.probe_start:])
            b_pool = np.concatenate(all_b)
            ntp_pool = np.concatenate(all_ntp)
            log_ntp_pool = np.log(ntp_pool + 1e-12)

            # NTP to beliefs: a ones column is appended for the bias term and the least-squares weights come from the
            # pseudo-inverse.
            X_ntp = np.hstack([ntp_pool, np.ones((len(ntp_pool), 1))])
            W_ntp = np.linalg.pinv(X_ntp) @ b_pool
            pred_ntp = X_ntp @ W_ntp
            ss_r = ((b_pool - pred_ntp)**2).sum()
            ss_t = ((b_pool - b_pool.mean(0))**2).sum()
            r2_ntp_to_b = 1.0 - ss_r / ss_t

            # log-NTP to beliefs: the same regression with log-NTP as the regressor (1e-12 is added before the log).
            X_log = np.hstack([log_ntp_pool, np.ones((len(log_ntp_pool), 1))])
            W_log = np.linalg.pinv(X_log) @ b_pool
            pred_log = X_log @ W_log
            ss_r = ((b_pool - pred_log)**2).sum()
            ss_t = ((b_pool - b_pool.mean(0))**2).sum()
            r2_logntp_to_b = 1.0 - ss_r / ss_t

            del all_b, all_ntp, b_pool, ntp_pool, log_ntp_pool

            # Probes from activations, one sequence per seed. The sequence is sampled again with the same seed, so it is
            # the same sequence whose beliefs entered the pooled baselines above.
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp = next_token_probs(beliefs, T)

                # Write the sequence as space-separated letters, locate the model token carrying each HMM token, and keep the
                # late window of belief states, NTP, and log-NTP as probe targets.
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                y_beliefs = beliefs[args.probe_start:n_matched]
                n_late = len(y_beliefs)
                y_ntp = ntp[args.probe_start:n_matched]
                y_log_ntp = np.log(y_ntp + 1e-12)
                late_pos = pos_indices[args.probe_start:n_matched]
                if n_late == 0: pbar.update(1); continue

                # One chunked forward pass with KV caching; the residual stream at every layer is kept for the late positions only.
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)

                # The paper's split (Section 5.1): a random 20 percent of the late positions train the probes and the remaining
                # 80 percent score them. The split is seeded by the sequence seed, as in every probe script.
                idx_tr, idx_te = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=seed)
                tgts_np = {"act→beliefs": y_beliefs, "act→ntp": y_ntp, "act→log_ntp": y_log_ntp}

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
                    # The two baselines are stored in every (layer, seed) row so that the notebook can plot them next to the
                    # layer curves.
                    all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":"ntp→beliefs","R2":r2_ntp_to_b})
                    all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":"log_ntp→beliefs","R2":r2_logntp_to_b})

                del acts
                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)
        pbar.close()

        # Write ntp_probes_<model>.csv after each family (the file is rewritten with all rows accumulated so far).
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, f"ntp_probes_{ms}.csv"), index=False)
    print(f"Done. {len(all_rows)} rows.")

if __name__ == "__main__": main()
