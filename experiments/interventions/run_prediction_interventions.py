"""Patching and steering the belief-state subspace at full context, scored by the KL divergence from the injected belief's
next-token probabilities to the model's prediction.

Paper section: Section 7 (Controlling predictions via the belief state) with Figure 7 and the right panel of Figure 1 (b),
Appendix E with Figure 16, and Appendix P with Figures 71 to 76.

Claim: "Belief state interventions systematically shift predictions" and "The injected belief largely overrides
conflicting context." In the paper's words, "Across k suffix lengths, the target's KL (solid) falls well below the
original KL (dotted). This suggests that injecting a target belief state into the subspace moves the LLM's predictions
toward the target's and away from the ones that its true history implies."

Experiment: Steering and patching as defined in Section 6.1, applied at the last k positions of the sequence for
k in {1, 5, 10}. In the paper's words, "we measure the KL divergence between the intervened LLM's predictions at the
final token position N and the ground-truth NTP implied by the final target belief state or the final original belief
state." The targets are past-consistent (a belief history continued from the sequence's own belief over another
sequence's suffix) and past-inconsistent (another sequence's belief states), and the random-belief control injects
symmetric-Dirichlet beliefs but is scored against the past-inconsistent target's NTP.

Result: The KL to the target falls well below the KL to the original belief for every k, and in later layers it is on
the order of the unmodified model's KL. The random control's KL does not drop. The shift toward the target grows with
k, and the past-consistent condition outperforms the past-inconsistent one at k = 1 with an advantage that disappears
by k = 10 (Appendix P). The results hold for nearly all HMMs across all LLMs.

How the code works: For each parametrization, the 10 sequences and their belief states are computed up front, and the
donor of each sequence is the next seed's sequence. For each sequence, the encoder (activations to belief) and the
decoder (belief to activations) of Section 6.1 are fit per layer by least squares with bias on a seeded random
20 percent of the late-window activations obtained from one full chunked pass. The measurement position is the last
held-out late position, and the tail is the K = max(k) tokens ending there. The prefix before the tail is run once
with KV caching, and every variant is a K-token pass on a deep copy of that cache: the clean pass (checked against
the full pass), the patched pass, whose hook at layer l overwrites the residual stream at the last k tail rows with
the decoded target beliefs, and the steered pass, whose hook adds the decoded target minus the decoded true belief.
The prediction at the last position is read from the final hidden state through the unembedding matrix. Every row
records the exact full-vocabulary KL of Section 4 from the target's NTP and from the factual NTP to the model's
prediction (the ground-truth NTP extended by zeros over the rest of the vocabulary), the same KLs restricted to the
HMM tokens (the columns with the suffix _hmm), the probability mass the model puts outside the HMM tokens, and the KL
between the target and factual NTPs. Rows go to prediction_interventions_<model>.csv.
"""
import argparse, copy, gc, os, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix)
from src.metrics.probes import fit_probe, predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

# Numeric helpers: the HMM-token KL, the seeded split, the probe fit, the belief filter, and the model's next-token
# distribution over the HMM tokens.
def _kl(p, q):
    """KL(p || q) over the HMM-token simplex. p,q: (..., n_tokens)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum(-1))

def _split(n, train_frac, seed):
    """Seeded random train/test split of n positions: the first round(n * train_frac) entries of a seeded permutation
    train and the rest test."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(n * train_frac))
    return perm[:n_tr], perm[n_tr:]

def _fit(X, Y, device):
    """fit_probe with a CPU fallback (torch.linalg.pinv is unsupported on MPS)."""
    try:
        return fit_probe(X, Y, use_bias=True).to(device)
    except (RuntimeError, NotImplementedError):
        return fit_probe(X.cpu(), Y.cpu(), use_bias=True).to(device)

def _continue_beliefs(start, toks, T_stack):
    """Run the Bayesian filter for |toks| steps from belief `start`.

    `start` is a prefix-end belief (the source's belief at i-k for past-consistent,
    or the donor's for past-inconsistent) and `toks` are the shared donor suffix b.
    Returns the k running beliefs (k, n_states); the last is the injected belief at N.
    """
    b = np.asarray(start, dtype=np.float64).copy()
    out = np.zeros((len(toks), b.shape[0]))
    for s, tok in enumerate(toks):
        b = b @ T_stack[int(tok)]
        z = b.sum()
        if z > 0:
            b = b / z
        out[s] = b
    return out

def _llm_ntp(last_hidden_row, lm_head_w, tok_ids, cap):
    """Truncate-to-HMM-tokens next-token distribution at the measure position.

    last_hidden_row: (B, d) final-position hidden states.
    Returns (B, n_tokens) renormalised over HMM tokens (paper Sec. 4 convention).
    """
    h = last_hidden_row.float().to(lm_head_w.device)
    with torch.no_grad():
        logits = h @ lm_head_w[tok_ids].float().T      # (B, n_tokens) only
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap     # element-wise; Gemma
        return torch.softmax(logits, dim=-1).cpu().numpy()

# The intervention conditions of Section 7: the round trip (patching with the sequence's own decoded belief, a check),
# past-consistent and past-inconsistent targets, and the random-belief control.
PATCH_CONDS = ["round_trip", "past_consistent", "past_inconsistent", "random"]
STEER_CONDS = ["past_consistent", "past_inconsistent", "random"]

# The model's next-token distribution with the softmax over the full vocabulary (the convention of Section 4), returned
# as the log-probabilities of the HMM tokens and the log of their total mass.
def _llm_ntp_fullvocab(last_hidden_row, lm_head_w, tok_ids, cap):
    """Section-4 convention: softmax over the FULL vocabulary (same arithmetic as src.model_utils.compute_fullvocab_kl:
    fp32 logits, Gemma softcap applied element-wise, logsumexp over the whole vocabulary). Returns the UNnormalised
    probabilities of the HMM tokens (B, n_tokens) and the off-alphabet mass (B,). KL(p_true || this) with p_true
    extended by zeros is the paper's Section-4 KL."""
    h = last_hidden_row.float().to(lm_head_w.device)
    with torch.no_grad():
        hmm_logits = h @ lm_head_w[tok_ids].float().T
        if cap is not None:
            hmm_logits = torch.tanh(hmm_logits / cap) * cap
        lse = torch.full((len(h),), float("-inf"), device=h.device)
        for i in range(0, lm_head_w.shape[0], 16384):
            partial = h @ lm_head_w[i:i + 16384].float().T
            if cap is not None:
                partial = torch.tanh(partial / cap) * cap
            lse = torch.logaddexp(lse, torch.logsumexp(partial, dim=-1))
            del partial
        log_q = (hmm_logits - lse.unsqueeze(-1)).double().cpu().numpy()      # log of the UNnormalised HMM-token probabilities
    log_in = np.logaddexp.reduce(log_q, axis=-1)                              # log of the in-alphabet mass
    return log_q, log_in

# Exact KL from a distribution on the HMM tokens to the model's full-vocabulary distribution, in log space.
def _kl_logq(p, log_q):
    """Exact full-vocabulary KL(p || q) in log space: p lives on the HMM tokens (extended by zeros elsewhere),
    log_q are the log-probabilities of the HMM tokens under the full-vocabulary softmax. No clipping of q."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return float((p * (np.log(p) - log_q)).sum(-1))

# The KV cache of the prefix, computed once per sequence.
def _prefix_cache(wrapper, input_ids, prefix_len, chunk_size, device):
    """Chunked KV-cached forward over input_ids[:, :prefix_len]; returns the cache."""
    past = None
    for start in range(0, prefix_len, chunk_size):
        end = min(start + chunk_size, prefix_len)
        with torch.no_grad():
            out = wrapper.forward(input_ids[:, start:end].to(device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        del out
    return past

# One pass over the K tail tokens on a copy of the prefix cache, with an optional patching or steering hook at one
# layer and optional capture of the residual stream at other layers.
def _tail_pass(wrapper, tail_ids, cache, layer=None, offsets=None, value=None, capture_layers=(), mode="patch"):
    """Run the K tail tokens on a COPY of `cache`. If `layer` is given, intervene on the residual at
    that layer at rows `offsets` of the tail: mode "patch" overwrites with `value` (k, d), mode "steer"
    adds `value`. Returns (last-position hidden state (1, d), {l: (K, d) fp32 residuals at capture_layers})."""
    c = copy.deepcopy(cache)
    caps, hooks = {}, []
    if layer is not None:
        def hk(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if mode == "patch":
                h[0, offsets, :] = value.to(h.dtype)
            else:
                h[0, offsets, :] = h[0, offsets, :] + value.to(h.dtype)
            return None                                    # keep the mutated tensor
        hooks.append(wrapper.get_layer(layer).register_forward_hook(hk))
    for l in capture_layers:
        def mk(li):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                caps[li] = h[0].detach().float()
            return fn
        hooks.append(wrapper.get_layer(l).register_forward_hook(mk(l)))
    with torch.no_grad():
        out = wrapper.forward(tail_ids, past_key_values=c, use_cache=True)
    for h in hooks:
        h.remove()
    lh = out.last_hidden_state[:, -1, :].detach()
    del out, c
    return lh, caps

# One sequence: probes, measurement position, prefix cache, clean baseline, targets, and every intervention.
def run_sequence(wrapper, tokenizer, cfg, T_matrices, pi, beliefs_all, tokens_all, seed,
                 layers, k_values, tok_ids, lm_head_w, cap, M, args, device, pbar):
    n_states = cfg["n_states"]
    T_stack = np.stack(T_matrices)
    tokens = tokens_all[seed]
    beliefs = beliefs_all[seed]
    prompt = tokens_to_prompt(tokens, cfg["token_names"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, _ = match_positions(input_ids, tok_ids)
    n_matched = min(len(tokens), len(pos_indices))
    if len(pos_indices) != len(tokens):
        print(f"WARNING alignment: {len(pos_indices)} matched positions vs {len(tokens)} tokens "
              f"({cfg['_name']} {cfg['_label']} s{seed})", flush=True)
    if n_matched <= args.probe_start + 50:
        pbar.update(len(layers))
        return []

    # The encoder (activations to belief) and the decoder (belief to activations) of Section 6.1, one pair per layer,
    # fit by least squares with bias on the training positions of the late window, from one full chunked pass.
    late_idx = np.arange(args.probe_start, n_matched)
    late_pos = pos_indices[late_idx]
    y_late = beliefs[late_idx]
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | fit enc/dec", refresh=True)
    acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)
    idx_tr, idx_te = _split(len(late_idx), args.train_frac, seed)
    y_tr = torch.tensor(y_late[idx_tr], device=device, dtype=torch.float32)
    enc, dec = {}, {}
    for l in layers:
        X = acts[l]
        if X.numel() == 0:
            continue
        Xtr = X[idx_tr].float()
        enc[l] = _fit(Xtr, y_tr, device)
        dec[l] = _fit(y_tr, Xtr, device)
    layers = [l for l in layers if l in dec]

    # The measurement position: the last held-out late position of this sequence (at least K = max(k) positions in,
    # so that the intervened tail lies inside the held-out set). The tail is the K HMM tokens ending there.
    K = max(k_values)
    eligible = np.sort(late_idx[idx_te])
    eligible = eligible[eligible >= K]
    i = int(eligible[-1])
    mp = int(pos_indices[i])
    tail_hmm = np.arange(i - K + 1, i + 1)
    assert np.array_equal(pos_indices[tail_hmm], np.arange(mp - K + 1, mp + 1)), "tail positions not contiguous"
    prefix_len = mp - K + 1
    tail_ids = input_ids[:, prefix_len:mp + 1].to(device)

    # The clean tail residuals from the full chunked pass, the reference for the cache check below.
    row_of = {int(h): r for r, h in enumerate(late_idx)}
    tail_rows = [row_of[int(h)] for h in tail_hmm]
    ref_tail = {l: acts[l][tail_rows].float() for l in layers}
    del acts
    gc.collect()
    torch.cuda.empty_cache()

    # The prefix before the tail is run once with KV caching, and every variant below is a K-token pass on a copy of
    # that cache. The clean tail pass checks that the cached computation reproduces the full pass.
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | prefix cache", refresh=True)

    cache = _prefix_cache(wrapper, input_ids, prefix_len, args.chunk_size, device)
    lh_c, caps = _tail_pass(wrapper, tail_ids, cache, capture_layers=layers)
    tail_check = max(float((caps[l] - ref_tail[l]).abs().max() / (ref_tail[l].abs().max() + 1e-6)) for l in layers)
    print(f"{cfg['_name']} {cfg['_label']} s{seed}: measure hmm idx {i} (model pos {mp}), prefix {prefix_len}, "
          f"cached-vs-full tail residual max rel diff {tail_check:.2e}", flush=True)

    # The unmodified model's prediction at the measurement position, scored against the factual NTP (the baseline_kl
    # columns and the gray curve of Figure 7). The two logit paths must agree on the distribution restricted to the HMM
    # tokens.
    p_clean_hmm = _llm_ntp(lh_c, lm_head_w, tok_ids, cap)[0]                   # restricted to the HMM tokens (the *_hmm columns)
    log_qc, log_in_c = _llm_ntp_fullvocab(lh_c, lm_head_w, tok_ids, cap)
    log_qc = log_qc[0]
    off_clean = float(1.0 - np.exp(log_in_c[0]))
    # the two logit paths must agree on the restricted distribution (well-conditioned check, independent of the mass scale)
    assert np.abs(p_clean_hmm - np.exp(log_qc - log_in_c[0])).max() < 1e-5, "restricted vs full-vocab logits disagree (clean)"
    p_factual = beliefs[i] @ M
    baseline_kl = _kl_logq(p_factual, log_qc)                                   # full-vocabulary KL (Section-4 convention), exact
    baseline_kl_hmm = _kl(p_factual, p_clean_hmm)                               # the same KL restricted to the HMM tokens
    rows = [dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed, layer=-1, k=0, intervention="none",
                 condition="unmodified", draw=0, kl_to_target=baseline_kl, kl_to_factual=baseline_kl,
                 kl_to_pi_target=float("nan"), baseline_kl=baseline_kl, context="full", tail_check=tail_check,
                 kl_to_target_hmm=baseline_kl_hmm, kl_to_factual_hmm=baseline_kl_hmm, kl_to_pi_target_hmm=float("nan"),
                 baseline_kl_hmm=baseline_kl_hmm, off_mass=off_clean, tgt_kl_to_factual=0.0)]

    # Targets for each k (the sequence conditions of Section 6.1). The donor is the next seed's sequence. Past-consistent:
    # the Bayesian filter continued from this sequence's belief at i - k over the donor's k suffix tokens. Past-
    # inconsistent: the donor sequence's own belief states at the same positions. Random: symmetric-Dirichlet draws
    # (rng seed + 999).
    rng = np.random.default_rng(seed + 999)
    donors = [(seed + j) % args.n_seeds for j in range(1, 2)]
    donors = [d for d in donors if d != seed] or [seed]
    ds = donors[0]
    targets = {}
    for k in k_values:
        b_toks = tokens_all[ds][i - k + 1:i + 1]
        targets[("past_consistent", k)] = _continue_beliefs(beliefs[i - k], b_toks, T_stack)
        targets[("past_inconsistent", k)] = beliefs_all[ds][i - k + 1:i + 1]
        targets[("random", k)] = rng.dirichlet(np.ones(n_states), size=k)

    # Interventions at every layer, for every k and every condition.
    for l in layers:
        pbar.set_postfix_str(f"{cfg['_name']} s{seed} | layer {l}", refresh=True)
        We, Wd = enc[l], dec[l]
        ct = caps[l]  # (K, d) clean tail at layer l
        for k in k_values:
            pi_ntp = targets[("past_inconsistent", k)][-1] @ M
            plan = []
            rt = predict_probe(ct[-k:], We, use_bias=True).cpu().numpy()          # enc(h) round trip
            plan.append(("round_trip", rt, rt[-1] @ M, None))
            plan.append(("past_consistent", targets[("past_consistent", k)], targets[("past_consistent", k)][-1] @ M, None))
            plan.append(("past_inconsistent", targets[("past_inconsistent", k)], pi_ntp, None))
            plan.append(("random", targets[("random", k)], targets[("random", k)][-1] @ M, pi_ntp))
            offsets = torch.arange(K - k, K, dtype=torch.long, device=device)
            if args.steer_ref == "true":                                             # paper Sec. 6.1: emb(eta_target) - emb(eta(x_1:t))
                bt_true = torch.tensor(beliefs[i - k + 1:i + 1], device=device, dtype=torch.float32)
                src_img = predict_probe(bt_true, Wd, use_bias=True)                  # emb(true belief) at the last k positions (k, d)
            else:                                                                    # alternative reference: the decoded belief emb(enc(h_clean))
                src_img = predict_probe(predict_probe(ct[-k:], We, use_bias=True), Wd, use_bias=True)

            # Patching overwrites the residual stream at the last k tail rows with dec(target); steering adds dec(target) minus
            # the reference (Section 6.1). Each variant is one tail pass on a copy of the prefix cache, read out at the last
            # position.
            for mode in args.interventions:
                for cond, bseq, p_tgt, p_alt in plan:
                    if mode == "steer" and cond == "round_trip":
                        continue                                                    # the round trip is a patching condition only
                    bt = torch.tensor(np.asarray(bseq), device=device, dtype=torch.float32)
                    dec_t = predict_probe(bt, Wd, use_bias=True)                    # (k, d) = dec(target)
                    value = dec_t if mode == "patch" else dec_t - src_img
                    lh, _ = _tail_pass(wrapper, tail_ids, cache, layer=l, offsets=offsets, value=value, mode=mode)

                    # Scoring: the KL from the target's NTP and from the factual NTP to the model's prediction, over the full vocabulary
                    # (columns without suffix) and restricted to the HMM tokens (the *_hmm columns), plus the off-alphabet mass and the
                    # KL between the target and factual NTPs.
                    p_llm_hmm = _llm_ntp(lh, lm_head_w, tok_ids, cap)[0]
                    log_qi, log_in_i = _llm_ntp_fullvocab(lh, lm_head_w, tok_ids, cap)
                    log_qi = log_qi[0]
                    off_llm = float(1.0 - np.exp(log_in_i[0]))
                    assert np.abs(p_llm_hmm - np.exp(log_qi - log_in_i[0])).max() < 1e-5, "restricted vs full-vocab logits disagree"
                    kt, kf = _kl_logq(p_tgt, log_qi), _kl_logq(p_factual, log_qi)
                    kt_h, kf_h = _kl(p_tgt, p_llm_hmm), _kl(p_factual, p_llm_hmm)
                    rows.append(dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed, layer=l, k=k,
                                     intervention=mode, condition=cond, draw=0,
                                     kl_to_target=kt, kl_to_factual=kf,
                                     kl_to_pi_target=(_kl_logq(p_alt, log_qi) if p_alt is not None else float("nan")),
                                     baseline_kl=baseline_kl, context="full", tail_check=tail_check,
                                     kl_to_target_hmm=kt_h, kl_to_factual_hmm=kf_h,
                                     kl_to_pi_target_hmm=(_kl(p_alt, p_llm_hmm) if p_alt is not None else float("nan")),
                                     baseline_kl_hmm=baseline_kl_hmm, off_mass=off_llm,
                                     tgt_kl_to_factual=_kl(p_tgt, p_factual)))
        pbar.update(1)
    del cache, caps, ref_tail
    gc.collect()
    torch.cuda.empty_cache()
    return rows

def main():
    # Command-line options. The paper's runs use --all_params with the default sequences, seeds, split, and k values;
    # scripts/run_interventions.sh adds --dtype float32 --sdp_backend math for the Gemma 4 checkpoints.
    P = argparse.ArgumentParser(description="Patching and steering the belief-state subspace at full context (Section 7)")
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--params", nargs="+", default=None, help="explicit param labels (else REPRESENTATIVES)")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10])
    P.add_argument("--layers", type=int, nargs="+", default=None)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--device", default="cuda")
    P.add_argument("--tag", default="")
    P.add_argument("--interventions", nargs="+", default=["patch", "steer"], choices=["patch", "steer"])
    P.add_argument("--smoke", action="store_true")
    P.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"],
                   help="model dtype. The paper's runs use float16 for Qwen and Llama and float32 for Gemma 4, whose float16 "
                        "forward pass is numerically unstable at the position level (E4B in float32 needs a 96 GB card)")
    P.add_argument("--steer_ref", default="true", choices=["true", "decoded"],
                   help="reference of the steering vector emb(target) - emb(reference): 'true' uses the true belief eta(x_1:t) at each "
                        "intervened position (the definition of Section 6.1); 'decoded' uses the decoded belief enc(h_clean) instead")
    P.add_argument("--sdp_backend", default="default", choices=["default", "math"],
                   help="math disables the flash, memory-efficient, and cudnn attention kernels so that every attention call uses the "
                        "reference kernel. The paper's Gemma 4 runs use it, because the sliding-window layers of Gemma 4 E2B "
                        "otherwise hit a defect of the memory-efficient kernel")
    P.add_argument("--attn_impl", default="default", choices=["default", "eager", "sdpa"],
                   help="override the attention implementation after loading. Not used by the paper's runs; eager attention is an "
                        "alternative workaround for the Gemma 4 E2B kernel defect")
    args = P.parse_args()

    # Smoke mode is a plumbing test: short sequences, two seeds, two k values, one family, and two layers.
    if args.smoke:
        args.seq_len = min(args.seq_len, 800)
        args.probe_start = 200
        args.n_seeds = 2
        args.k_values = [1, 5]
        args.families = args.families[:1]

    os.makedirs(args.output_dir, exist_ok=True)

    # Model loading: float16 through the shared loader, or another dtype with the same loading code.
    if args.dtype == "float16":
        wrapper, tokenizer = load_model(args.model, args.device)
    else:                                                    # same loading as src.model_utils.load_model, different dtype
        import src.model_utils as _mu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _cfg = _mu.MODEL_CONFIGS.get(args.model, {})
        _family = _cfg.get("family", _mu._detect_family(args.model))
        _tok = os.environ.get("HF_TOKEN")
        _kw = dict(dtype=getattr(torch, args.dtype), device_map="auto")
        if _tok: _kw["token"] = _tok
        if _cfg.get("sdpa", _family != "gemma"): _kw["attn_implementation"] = "sdpa"
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=_tok)
        _model = AutoModelForCausalLM.from_pretrained(args.model, **_kw)
        _model.eval()
        wrapper = _mu.ModelWrapper(_model, _family)
        print(f"Loaded {args.model} ({_family}) in {args.dtype}: {wrapper.n_layers} layers, d={wrapper.hidden_size}", flush=True)

    if args.sdp_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("sdpa kernels restricted to MATH", flush=True)
    if args.attn_impl != "default":
        wrapper.model.set_attn_implementation(args.attn_impl)
    print(f"attention implementation: {wrapper.model.config._attn_implementation}", flush=True)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = sorted(set([wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    lm_head_w = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)

    # Parametrizations per family: all of them, explicit labels, or the representative.
    fam_params = {}
    for f in args.families:
        cfg = HMMS[f]
        if args.all_params:
            fam_params[f] = cfg["params"]
        elif args.params:
            fam_params[f] = [p for p in cfg["params"] if cfg["label_fn"](p) in args.params]
        else:
            fam_params[f] = [REPRESENTATIVES[f]]
    total = sum(len(v) for v in fam_params.values()) * args.n_seeds * len(layers)
    pbar = tqdm(total=total, desc="fullctx " + "+".join(args.interventions), smoothing=0.05)
    out_csv = os.path.join(args.output_dir, f"prediction_interventions_{ms}{args.tag}.csv")
    all_rows = []
    for hmm_name, params in fam_params.items():
        cfg = HMMS[hmm_name]
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        for param in params:
            label = cfg["label_fn"](param)
            cfg_p = {**cfg, "_name": hmm_name, "_label": label}
            T_matrices = cfg["fn"](*param)
            T_stack = np.stack(T_matrices)
            pi = stationary_distribution(T_matrices)
            M = emission_matrix(T_matrices)

            # All sequences and belief states of this parametrization are computed up front, because each sequence's donor is
            # the next seed's sequence.
            beliefs_all, tokens_all = {}, {}
            for s in range(args.n_seeds):
                tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=s).astype(np.int64)
                tokens_all[s] = tk
                beliefs_all[s] = full_bayesian_beliefs(tk, T_stack, pi)
            tqdm.write(f"===== {hmm_name} {label} =====")
            for seed in range(args.n_seeds):
                all_rows += run_sequence(wrapper, tokenizer, cfg_p, T_matrices, pi, beliefs_all, tokens_all,
                                         seed, layers, args.k_values, tok_ids, lm_head_w, cap, M, args, device, pbar)
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    pbar.close()
    print(f"Done. {len(all_rows)} rows -> {out_csv}", flush=True)

if __name__ == "__main__":
    main()
