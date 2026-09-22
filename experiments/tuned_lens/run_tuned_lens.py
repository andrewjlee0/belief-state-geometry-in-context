"""Per-layer tuned lenses toward the HMM's ground-truth next-token probabilities, toward control targets, and toward
the model's own output, together with the untrained logit lens.

Paper section: Section 8 (Decoding predictions from intermediate layers) with Figure 8, and Appendix Q with
Figures 77 to 94.

Claim: "Residual stream activations carry HMM predictive structure" and "Early-exit predictive accuracy covaries with
belief state decodability." In the paper's words, "A tuned lens trained toward the ground-truth NTP attains low KL
across most layers, falling below the KL of the LLM's own predictions, whereas tuned lenses trained toward shuffled
and random NTP perform substantially worse", and "The tuned lens is most accurate at layers where the belief state is
most decodable."

Experiment: In the paper's words, "For a fixed layer, an affine transformation is learned to align the residual stream
activations such that the application of LayerNorm, the unembedding matrix and softmax, best approximates a target
distribution." The target is the ground-truth NTP, with shuffled and random controls, and, for the canonical tuned
lens of Appendix Q.1, the model's own output distribution. Following Section 5, the lenses are trained on a random
20 percent of the last 5,000-token window and tested on the held-out 80 percent, and every lens is scored by the KL
divergence from its target and from the ground truth at each layer.

Result: The lens trained toward the ground-truth NTP attains low KL across most layers, below the KL of the model's
own predictions, while the shuffled and random lenses are substantially worse, for all HMMs and all LLMs (Appendix
Q.2). The per-layer KL covaries with 1 minus the belief probe's R² across layers (Figure 8, bottom, and Appendix Q.3),
and the canonical lens shows the same covariation (Appendix Q.1).

How the code works: For each parametrization, the late-window activations of the 10 sequences are extracted at
every layer and pooled, with the ground-truth NTP at the same positions and the seed of each position. The model's
own distribution over the HMM tokens is the readout of the last layer's activations. A seeded random 20 percent of
the pooled positions trains every lens and the remaining 80 percent scores every lens. The readout of a lens is the
model's final norm followed by the HMM-token rows of the unembedding matrix and a softmax, so the lens's
distribution and every KL are restricted to the HMM tokens. Per layer and target, an affine translator initialized at
the identity is trained with Adam to minimize the KL from the target to the lens's distribution, in minibatches whose
order is not seeded, so repeated runs differ by training noise. The logit lens is the untrained readout. Every lens is
scored on the held-out positions by the per-position KL to its own target (kl_self), to the ground-truth NTP (kl_hmm),
and to the model's own distribution (kl_final), averaged within each held-out sequence, giving the rows (hmm, param,
layer, lens, seed, kl_self, kl_hmm, kl_final) of tunedlens_<model>.csv.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# TF32 matrix multiplications, for speed.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix)
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

# Helpers: the model's final norm, the split, the KL, the cross-parametrization donor, the lens readout, and the
# translator training.
def _final_norm(wrapper):
    """The model's final norm module (applied after the last decoder block, before
    lm_head). Family-aware; raise a clear error if the attribute path differs."""
    m = wrapper.model
    cands = ([m.model.language_model] if wrapper.family == "gemma" else [m.model])
    for base in cands + [m.model]:
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(base, attr):
                return getattr(base, attr)
    raise AttributeError("Could not locate the final norm on this model; adjust _final_norm().")

# Seeded random split of the pooled positions: the first round(n * train_frac) entries of a seeded permutation
# train and the rest test.
def _split(n, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(n * train_frac))
    return perm[:n_tr], perm[n_tr:]

def _kl(p, q):
    """KL(p || q) per row, over the concept-token simplex. p,q: (...,C)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return (p * np.log(p / q)).sum(-1)

def _cross_donor_by_ntpkl(cfg, param, seqs, src_beliefs, M, ps):
    """Cross-control donor = the SAME-family parametrization whose optimal next-token
    distribution is most DIVERGENT from the source's, measured on the SOURCE sequences.

    For each candidate donor p', filter the *source* token sequence under p' to get the
    donor's beliefs, map to NTPs, and take mean KL(source_NTP || donor_NTP) over the
    pooled window (positions >= ps) across all sequences; pick the argmax (furthest).
    Parameter-free (no probe / regression / R2); directional (source tokens, donor NTP),
    so no symmetrization. Returns (donor_param, mean_kl)."""
    plist = cfg["params"]
    self_lbl = cfg["label_fn"](param)
    best, best_kl = None, -1.0
    for p in plist:
        if cfg["label_fn"](p) == self_lbl:
            continue
        dT = cfg["fn"](*p)
        dstack = np.stack(dT)
        dpi = stationary_distribution(dT)
        dM = emission_matrix(dT)
        tot, cnt = 0.0, 0
        for tokens, sb in zip(seqs, src_beliefs):
            if len(tokens) <= ps:
                continue
            sp = np.clip(sb[ps:] @ M, 1e-12, None)                          # source NTP
            dq = np.clip(full_bayesian_beliefs(tokens, dstack, dpi)[ps:] @ dM, 1e-12, None)  # donor NTP, SAME seq
            tot += (sp * np.log(sp / dq)).sum()
            cnt += sp.shape[0]
        mean_kl = tot / max(cnt, 1)
        if mean_kl > best_kl:
            best_kl, best = mean_kl, p
    return best, best_kl

# The lens readout of Section 8.1: the model's final norm, then the HMM-token rows of the unembedding matrix (and
# Gemma's final logit softcap), so that the softmax over these logits is a distribution over the HMM tokens.
def _readout(h, norm, Wc, cap, norm_dtype):
    """Concept logits = unembed_concept( final_norm(h) ). h:(B,d) fp32 -> (B,C) fp32.
    norm runs in the model's dtype; Wc is the concept slice of lm_head.weight (C,d)."""
    normed = norm(h.to(norm_dtype)).float()
    logits = normed @ Wc.float().T
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits

# Translator training: an affine map initialized at the identity, trained with Adam in minibatches drawn in an
# unseeded random order to minimize the KL from the target distribution to the lens's distribution.
def _train_translator(X_tr, target_tr, norm, Wc, cap, norm_dtype, d, device,
                      epochs, lr, batch):
    """Affine translator (init identity) trained to minimise KL(target || lens)."""
    T = torch.nn.Linear(d, d, bias=True).to(device)
    with torch.no_grad():
        T.weight.copy_(torch.eye(d, device=device))
        T.bias.zero_()
    opt = torch.optim.Adam(T.parameters(), lr=lr)
    n = X_tr.shape[0]
    logt = torch.log(target_tr.clamp_min(1e-12))
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits = _readout(T(X_tr[idx]), norm, Wc, cap, norm_dtype)
            logq = F.log_softmax(logits, dim=-1)
            loss = (target_tr[idx] * (logt[idx] - logq)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return T

# The pipeline for one parametrization: pool the late-window activations of all sequences, build the targets, train one
# translator per (layer, target), and score every lens on the held-out positions.
def run_param(wrapper, tokenizer, cfg, param, layers, tok_ids, Wc, norm, cap,
              norm_dtype, args, device, pbar):
    T_matrices = cfg["fn"](*param)
    T_stack = np.stack(T_matrices)
    pi = stationary_distribution(T_matrices)
    M = emission_matrix(T_matrices)
    label = cfg["label_fn"](param)
    ps = args.probe_start
    controls = args.controls

    # The 10 sequences of this parametrization and their exact belief states (no model involved).
    seqs, src_beliefs = [], []
    for seed in range(args.n_seeds):
        tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=seed).astype(np.int64)
        seqs.append(tk)
        src_beliefs.append(full_bayesian_beliefs(tk, T_stack, pi))

    # Cross-parametrization control: the same-family parametrization whose NTP diverges most from the source's on the
    # source sequences (the donor's beliefs are filtered from the source tokens).
    dstack = dpi = dM = None
    if "cross" in controls:
        dparam, dkl = _cross_donor_by_ntpkl(cfg, param, seqs, src_beliefs, M, ps)
        if dparam is None:
            raise ValueError("cross control needs >=2 parametrizations (use --all_params)")
        dT = cfg["fn"](*dparam)
        dstack = np.stack(dT)
        dpi = stationary_distribution(dT)
        dM = emission_matrix(dT)

    # Late-window activations at every layer from one chunked forward pass per sequence, pooled over the sequences,
    # with the ground-truth NTP, the control targets, and the seed of every position.
    acts_pool = {l: [] for l in layers}
    oracle_pool, o1_pool, cross_pool, seed_pool = [], [], [], []
    for seed in range(args.n_seeds):
        tokens = seqs[seed]
        beliefs = src_beliefs[seed]  # reuse the source seq + beliefs
        prompt = tokens_to_prompt(tokens, cfg["token_names"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
        pos, _ = match_positions(input_ids, tok_ids)
        n = min(len(tokens), len(pos))
        pos = pos[:n]
        if n <= ps:
            continue
        acts, _ = extract_activations_chunked(wrapper, input_ids, layers, pos[ps:n],
                                              args.chunk_size, device)
        for l in layers:
            acts_pool[l].append(acts[l].detach().cpu())
        oracle_pool.append((beliefs[ps:n] @ M).astype(np.float32))
        seed_pool.append(np.full(n - ps, seed, dtype=np.int64))   # which seq each position came from
        if "order1" in controls:                                 # order-1 (1-HMM) NTP
            tk = tokens[ps:n]
            ob = np.einsum("s,nsd->nd", pi, T_stack[tk])
            ob /= ob.sum(1, keepdims=True)
            o1_pool.append((ob @ M).astype(np.float32))
        if "cross" in controls:                                  # donor HMM's NTP on these tokens
            cb = full_bayesian_beliefs(tokens, dstack, dpi)[ps:n]
            cross_pool.append((cb @ dM).astype(np.float32))
        del acts
        torch.cuda.empty_cache()

    for l in layers:
        acts_pool[l] = torch.cat(acts_pool[l], dim=0)
    oracle = np.concatenate(oracle_pool, 0)
    N, C = oracle.shape

    # The split of Section 8.1: a random 20 percent of the pooled positions trains every translator and the remaining
    # 80 percent scores every lens. The held-out positions keep their sequence id for the per-sequence averages.
    tr, te = _split(N, args.train_frac, 42)
    seed_te = np.concatenate(seed_pool, 0)[te]                # seq id for each held-out position

    # The model's own output distribution over the HMM tokens: the readout of the last layer's activations. It is the
    # target of the canonical lens (Appendix Q.1) and the reference of the kl_final column.
    last = layers[-1]
    with torch.no_grad():
        h = acts_pool[last].to(device)
        final = F.softmax(_readout(h, norm, Wc, cap, norm_dtype), dim=-1).cpu().numpy()
        del h
        torch.cuda.empty_cache()

    # Target distribution of each trained lens (Section 8.1). tuned_hmm is the HMM's ground-truth NTP and tuned_concept
    # is the model's own distribution (the canonical lens of Appendix Q.1). The controls mirror those of the belief
    # probes: shuffle is the ground-truth NTP with the positions permuted (rng 77777), random is an i.i.d. symmetric
    # Dirichlet distribution over the HMM tokens at every position (rng 88888), order1 is the 1-HMM's NTP given the
    # previous token, and cross is the donor parametrization's NTP on the same tokens.
    targets = {"tuned_concept": final, "tuned_hmm": oracle}
    if "shuffle" in controls:
        targets["shuffle"] = oracle[np.random.default_rng(77777).permutation(N)]
    if "random" in controls:
        targets["random"] = np.random.default_rng(88888).dirichlet(np.ones(C), size=N).astype(np.float32)
    if "order1" in controls:
        targets["order1"] = np.concatenate(o1_pool, 0)
    if "cross" in controls:
        targets["cross"] = np.concatenate(cross_pool, 0)

    d = wrapper.hidden_size
    rows = []
    # Per layer: the untrained logit lens, then one trained translator per target.
    for l in layers:
        Xtr = acts_pool[l][tr].to(device)
        Xte = acts_pool[l][te].to(device)
        pbar.set_postfix_str(f"{cfg['_name']} {label} | L{l}")

        variants = {}
        # The logit lens: the readout of the raw activations, whose own target is the model's distribution.
        with torch.no_grad():
            variants["logit"] = (F.softmax(_readout(Xte, norm, Wc, cap, norm_dtype), -1).cpu().numpy(), final)
        for name, tgt in targets.items():
            tgt_tr = torch.from_numpy(np.ascontiguousarray(tgt[tr])).to(device)
            Tr = _train_translator(Xtr, tgt_tr, norm, Wc, cap, norm_dtype, d, device,
                                   args.tl_epochs, args.tl_lr, args.tl_batch)
            with torch.no_grad():
                probs = F.softmax(_readout(Tr(Xte), norm, Wc, cap, norm_dtype), -1).cpu().numpy()
            variants[name] = (probs, tgt)
            del Tr
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_postfix_str(f"{cfg['_name']} {label} | L{l} {name}")

        # Score every lens on the held-out positions by three per-position KLs, averaged within each held-out sequence so
        # that the notebook can draw confidence intervals over sequences.
        for name, (probs, tgt) in variants.items():
            ks = _kl(tgt[te], probs)        # per-position KL to this lens's own target
            kh = _kl(oracle[te], probs)     # per-position KL to the real HMM oracle
            kf = _kl(final[te], probs)      # per-position KL to the model's own NTP
            for sd in np.unique(seed_te):   # one row per held-out sequence -> replicates for CIs
                m = seed_te == sd
                rows.append(dict(
                    hmm=cfg["_name"], param=label, layer=l, lens=name, seed=int(sd),
                    kl_self=float(ks[m].mean()),
                    kl_hmm=float(kh[m].mean()),
                    kl_final=float(kf[m].mean()),
                ))
        del Xtr, Xte
        torch.cuda.empty_cache()
    del acts_pool
    gc.collect()
    torch.cuda.empty_cache()
    return rows

def main():
    # Command-line options. The paper's runs use --all_params --controls shuffle random order1 cross with the default
    # sequences (20,000 tokens, 10 seeds, late window from 15,000), split (20 percent), and translator training
    # (20 epochs, learning rate 1e-5, batches of 512).
    P = argparse.ArgumentParser(description="Per-layer tuned lens over HMM data")
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--param_index", type=int, default=-1, help="select one param per family")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--probe_start", type=int, default=15000,
                   help="positions from probe_start onward are used (the late window of Section 5)")
    P.add_argument("--train_frac", type=float, default=0.2,
                   help="20/80 split shared by translator training and the belief probe")
    P.add_argument("--controls", nargs="*", default=["shuffle", "random"],
                   choices=["shuffle", "random", "order1", "cross"],
                   help="control targets to also fit a lens to (mirror the probe controls)")
    P.add_argument("--layers", type=int, nargs="+", default=None)
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--tl_epochs", type=int, default=20)
    P.add_argument("--tl_lr", type=float, default=1e-5)
    P.add_argument("--tl_batch", type=int, default=512)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true", help="tiny local plumbing test")
    args = P.parse_args()

    # Smoke mode is a plumbing test: short sequences, four seeds, three epochs, one family, and two layers.
    if args.smoke:
        args.seq_len = 400
        args.n_seeds = 4
        args.probe_start = 100
        args.tl_epochs = 3
        args.families = args.families[:1]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model. Its parameters are frozen; only the translators are trained.
    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)               # only translators are trained

    # The model's final norm, the unembedding matrix, and Gemma's final logit softcap.
    norm = _final_norm(wrapper)
    norm_dtype = next(norm.parameters()).dtype if any(True for _ in norm.parameters()) else \
                 next(wrapper.model.parameters()).dtype
    lm_W = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    # Parametrizations per family: all of them with --all_params, one by index, or the representative.
    def params_for(fam):
        c = HMMS[fam]
        if args.param_index >= 0:
            return [c["params"][args.param_index]] if args.param_index < len(c["params"]) else []
        return c["params"] if args.all_params else [REPRESENTATIVES[fam]]

    # Progress accounting: one translator training per (layer, target).
    n_trained = 2 + len(args.controls)            # tuned_concept, tuned_hmm, + each control
    units_per_param = len(layers) * n_trained     # one translator training per (layer, target)
    total = sum(len(params_for(f)) for f in args.families if f in HMMS) * units_per_param
    pbar = tqdm(total=total, desc="tuned-lens (layer x target)")
    all_rows = []
    for fam in args.families:
        cfg = HMMS.get(fam)
        if not cfg:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        Wc = lm_W[tok_ids].detach()                          # (C, d) concept unembed
        for param in params_for(fam):
            cfg = {**cfg, "_name": fam}
            pbar.set_postfix_str(f"{fam} {cfg['label_fn'](param)}")
            all_rows += run_param(wrapper, tokenizer, cfg, param, layers, tok_ids,
                                  Wc, norm, cap, norm_dtype, args, device, pbar)

            # Write tunedlens_<model>.csv after each parametrization (rewritten with all rows accumulated so far).
            pd.DataFrame(all_rows).to_csv(
                os.path.join(args.output_dir, f"tunedlens_{ms}.csv"), index=False)
    pbar.close()
    print("Done.")

if __name__ == "__main__":
    main()