"""In-context prediction accuracy: the KL divergence between the HMM's next-token probabilities and the LLM's.

Paper section: Section 4 (Evaluating in-context prediction accuracy) with Figure 2, and Appendices G and H with
Figures 17 to 28.

Claim: "In-context prediction converges" and "LLMs outperform low-order HMM predictors". In the paper's words,
"Figure 2 shows that the KL falls rapidly over the first 5,000-10,000 tokens, and then plateaus, during which we say
that the model's predictions have converged", and for Qwen 3.5 9B "the converged KL sits well below that of the
0-HMM and 1-HMM, and matches that of the k-HMM at k = 4.2-11.0 for the four main-text parametrizations".

Experiment: For every parametrization of every family and each of 10 seeds, a 20,000-token sequence is sampled from
the HMM, written as space-separated letters, and run through the model once. At every position the KL divergence
from the HMM's ground-truth next-token distribution, extended by zeros to the model's full vocabulary, to the model's
full-vocabulary next-token distribution is computed (Section 4.1). The same KL is computed for two k-HMM baselines
of Section 2.3: the 1-HMM, which predicts from the most recent token only, and the 0-HMM, which predicts the
stationary token distribution.

Result: In every model the KL falls over the first 5,000 to 10,000 tokens and then plateaus below the 0-HMM and
1-HMM baselines. For Qwen 3.5 9B the KL converged over the last 5,000 tokens is 0.010 to 0.018 nats for the four
main-text parametrizations, which matches a k-HMM at k between 4.2 and 11.0. Gemma 4 E2B is the exception and is
no more accurate than the 1-HMM on 6 of 10 Wing and 4 of 10 Strata parametrizations.

How the code works: The model is loaded once. For each parametrization the script builds the transition matrices,
a lookup table with the 1-HMM's next-token distribution for each previous token, and the 0-HMM's next-token row.
For each seed it samples the sequence, tokenizes it, matches HMM tokens to model positions, runs one chunked forward
pass with KV caching that keeps only the final hidden states, and computes the exact Bayesian belief state and the
ground-truth next-token probabilities at every position. The model's KL is computed from the final hidden states by
src.model_utils.compute_fullvocab_kl, whose softmax runs over the whole vocabulary. Each KL curve is stored as a
rolling mean over 100 positions, and all rows are written to kl_<model>.csv with the columns position, KL, source
(LLM, Order-1, or Order-0), hmm, param, and seed.
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.metrics.kl import kl_divergence
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked, compute_fullvocab_kl

def main():
    # Command-line options. The defaults are the paper's settings: sequences of 20,000 tokens, 10 seeds per
    # parametrization, and forward passes in chunks of 4,096 tokens. probe_start is accepted for uniformity with the
    # probe scripts and is not used by this script.
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load the model once and derive the short model key used in the output file name (for example qwen35_9b).
    wrapper, tokenizer = load_model(args.model, device)
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")
    kl_chunks = []

    # Loop over the requested families (all of them by default) and every parametrization of each family.
    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        n_tok = cfg["n_tokens"]
        pbar = tqdm(total=len(cfg["params"])*args.n_seeds, desc=f"{hmm_name} KL")
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            # The HMM: the per-token transition matrices T^(x) (Equation 1), their stacked array, and the stationary
            # distribution pi that serves as the initial belief state.
            T = cfg["fn"](*param)
            T_stack = np.stack(T)
            pi = stationary_distribution(T)

            # The 1-HMM baseline (Section 2.3 with k = 1) predicts the next token from the most recent token only. Its
            # next-token distribution after previous token zp is the 1-HMM belief reached from the stationary prior by
            # observing zp, mapped to token probabilities, so the baseline is a lookup table with one row per previous token.
            T_o1 = cfg["order_one_fn"](*param)
            pi_o1 = stationary_distribution(T_o1)
            T_o1_stack = np.stack(T_o1)
            ntp_o1_lut = np.zeros((n_tok, n_tok))
            for zp in range(n_tok):
                b = pi_o1 @ T_o1_stack[zp]
                b /= b.sum()
                for zn in range(n_tok): ntp_o1_lut[zp, zn] = (b @ T_o1_stack[zn]).sum()

            # The 0-HMM baseline (k = 0) predicts the stationary token distribution of the order-0 construction at every
            # position. Mess3 has no order-0 constructor because its stationary token distribution is uniform, which is
            # exactly the fallback row.
            if cfg["order_zero_fn"] is not None:
                T_o0 = cfg["order_zero_fn"](*param)
                pi_o0 = stationary_distribution(T_o0)
                ntp_o0_row = next_token_probs(pi_o0.reshape(1,-1), T_o0)[0]
            else:
                ntp_o0_row = np.full(n_tok, 1.0/n_tok)

            # One sequence per seed. Sampling is seeded, so every script in the repository sees the same sequence for a
            # given parametrization and seed.
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)

                # Write the sequence as space-separated letters and locate the model token that carries each HMM token.
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))

                # One chunked forward pass over the whole sequence with KV caching. No layer hooks are needed here because
                # only the final hidden states are kept; the next-token distributions are computed from them below.
                _, hidden_cat = extract_activations_chunked(
                    wrapper, input_ids, [], np.array([], dtype=int),
                    args.chunk_size, device, collect_hidden=True)

                # Ground truth: the exact Bayesian belief state at every position (Equation 3), the next-token probabilities it
                # implies (Equation 5), and the two baselines evaluated on the same tokens.
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp_true = next_token_probs(beliefs, T)
                ntp_o1_arr = ntp_o1_lut[tokens]
                ntp_o0_arr = np.broadcast_to(ntp_o0_row, ntp_true.shape).copy()

                # KL(HMM || LLM) at every matched position, with the model's softmax taken over its full vocabulary. The
                # baselines' KL is computed over the HMM tokens, which is exact for them because they place no mass elsewhere.
                kl_llm = compute_fullvocab_kl(wrapper.model, hidden_cat, pos_indices[:n_matched], n_matched, ntp_true, tok_ids, device, family=wrapper.family)
                kl_o1 = kl_divergence(ntp_true[:n_matched], ntp_o1_arr[:n_matched])
                kl_o0 = kl_divergence(ntp_true[:n_matched], ntp_o0_arr[:n_matched])
                del hidden_cat

                # Store each curve as a rolling mean over 100 positions, the convention of Figure 2. The row labeled with
                # position p averages the KL at positions p-99 through p, so the first row is position 100.
                for vals, source in [(kl_llm,"LLM"),(kl_o1,"Order-1"),(kl_o0,"Order-0")]:
                    if len(vals) <= 100: continue
                    cs = np.cumsum(vals)
                    rm = (cs[100:] - cs[:-100]) / 100
                    kl_chunks.append(pd.DataFrame({"position":np.arange(100,100+len(rm)),"KL":rm,"source":source,"hmm":hmm_name,"param":label,"seed":seed}))

                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)
        pbar.close()

    # Write every row to kl_<model>.csv.
    kl_df = pd.concat(kl_chunks, ignore_index=True) if kl_chunks else pd.DataFrame()
    kl_df.to_csv(os.path.join(args.output_dir, f"kl_{ms}.csv"), index=False)
    print(f"Done. {len(kl_df)} rows.")

if __name__ == "__main__": main()
