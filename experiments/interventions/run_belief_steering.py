"""Steering the belief-state subspace and decoding, with frozen probes, the log-NTP, the belief's delta-component, and a
random direction's component at every later layer.

Paper section: Section 6 (Controlling representations via the belief state) with Figure 6, and Appendices M, N, and O
with Figures 53 to 70.

Claim: "Steering the belief state systematically changes log-NTP decodability" and "Belief interventions have
privileged effects beyond the NTP subspace." In the paper's words, "when steering at layers beyond roughly the first
third of the network, the log-NTP implied by the target belief is highly decodable at subsequent probing layers", and
"the target delta-component is installed in subsequent layers, particularly when steering in the final half of the
LLM, whereas intervening before induces significant self-repair."

Experiment: Section 6.1. An affine embedding emb_l from belief states to activations is fit per layer and per
sequence by linear regression. Steering replaces the activation h_l,t by h_l,t + Delta_l,t with
Delta_l,t = emb_l(eta_target,t) - emb_l(eta(x_1:t)) at the last k = 5,000 positions. Three steers are run. The first
steers toward the belief states of a past-inconsistent sequence, and linear probes trained before steering decode the
target's log-NTP at every later layer. The second steers along a direction delta in the kernel of the emission matrix,
which leaves NTP and log-NTP unchanged, and probes decode the delta-component of the belief. The third steers along a
random unit direction v with the same per-position magnitude and is read out by projecting the activations onto v.

Saved Outputs: belief_steering_<tag><model>.csv in --output_dir, where the tag records the mode, the donor, and the
split (the paper's runs write belief_steering_story_<donor>_rsplit_<model>.csv). The columns are hmm, param, seed,
intervene_layer (-1 for the before-steering rows), readout_layer, condition (the donor), intervention (none or steer),
phase (before or after), probe (clean, frozen, retrain, or fixed_dir), source_kind, target_kind (belief, hidden, ntp,
log_ntp, or rand_proj), ref (orig or new), metric, value (an R²), n_train, n_test, M_rank, and M_cond.

How the code works: For each parametrization, the 10 sequences and their belief states are computed up front. For each
sequence the intervened positions are the last k = 5,000 HMM tokens, split into 1,000 training and 4,000 held-out
positions (--split random is the paper's interleaved split from train_test_split(random_state=seed)). The prefix
before the intervened window is run once with KV caching, and the suffix is run on a copy of that cache to capture
the clean activations at every layer at the intervened positions. The embedding emb_l (belief to activations) and
its encoder (activations to belief) are fit per layer by least squares with bias on the training positions. The
donor sets the targets: past_inconsistent takes the next seed's belief states at the same positions, ntp_matched
moves every belief along the unit kernel direction delta of the emission matrix by 90 percent of the largest step
that keeps it a probability distribution (same NTP, different belief), and random_matched keeps those targets but
replaces the direction of the injected vector by a fixed random unit vector with the same per-position norms.
Frozen probes are fit on the clean training activations to the original belief, NTP, log-NTP, and delta-coordinate
at every layer. For every steering layer l, the steering vector is added at the intervened positions in a suffix
pass from a fresh copy of the prefix cache, the activations are captured at every readout layer r >= l, and the
frozen probes are scored on the held-out positions against the original and the target values. The rows go to
belief_steering_story_<donor>_rsplit_<model>.csv. scripts/run_interventions.sh runs one job per family and merges
them into belief_steering_<donor>_<model>.csv, which the notebook reads.
"""
import argparse, copy, gc, os, sys, zlib
import numpy as np, pandas as pd, torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix, next_token_probs)
from src.metrics.probes import fit_probe, predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

# Numeric helpers: the HMM-token KL and the probe fit with a CPU fallback.
def _kl(p, q):
    """KL(p || q) over the HMM-token simplex. p,q: (..., n_tokens)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum(-1))

def _fit(X, Y, device):
    """fit_probe with a CPU fallback (torch.linalg.pinv is unsupported on MPS)."""
    try:
        return fit_probe(X, Y, use_bias=True).to(device)
    except (RuntimeError, NotImplementedError):
        return fit_probe(X.cpu(), Y.cpu(), use_bias=True).to(device)

# The three decode targets. The steered source is the belief state in the paper's runs.
SRC = ["belief", "ntp", "log_ntp"]
# The sequence condition of Section 6: past-inconsistent targets, the belief states of another sequence.
CONDS = ["past_inconsistent"]
EPS = 1e-12

# Coefficient of determination pooled over the output dimensions.
def _r2(pred, target):
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    if len(pred) < 2:
        return float("nan")
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

# The belief, NTP, or log-NTP values of a belief sequence, depending on the steered source.
def _vals(src, belief_seq, M):
    b = np.asarray(belief_seq, np.float64)
    if src == "belief":
        return b
    ntp = b @ M
    return ntp if src == "ntp" else np.log(ntp + EPS)

# The delta direction of Section 6.1: a unit vector in the left null space of the emission matrix M. Moving a belief
# along it changes the belief but not the NTP. The kernel is one-dimensional for Arch, Wing, and Strata, and empty
# for Mess3, whose emission matrix is invertible.
def _kerM_dir(M, tol=1e-8):
    """Left-null-space direction delta of M (delta @ M = 0), or None if M is full row-rank.
    Since emission rows sum to 1 (M @ 1 = 1), delta @ M = 0 => delta @ 1 = 0, so moving a belief
    along delta keeps it on the simplex AND leaves the next-token probs (ntp = b @ M) unchanged."""
    U, S, _ = np.linalg.svd(np.asarray(M, np.float64))
    rank = int((S > S.max() * tol).sum()) if S.size and S.max() > 0 else 0
    if rank >= M.shape[0]:
        return None                                      # rank-full (e.g. Mess3): no kernel
    return U[:, rank]                                     # (n_states,) singular vec, ~0 singular value

# The ntp_matched targets: each belief moved along delta by 90 percent of the largest step that keeps it a
# probability distribution, in whichever sign allows the larger move.
def _kerM_donor(b, M, frac=0.9):
    """NEW belief with the SAME ntp as `b` but a DIFFERENT belief: move each row maximally along
    ker(M) within the simplex. 'fix prediction, change belief'. Returns None if M is full-rank."""
    d = _kerM_dir(M)
    if d is None:
        return None
    b = np.asarray(b, np.float64)
    out = b.copy()
    for i in range(len(b)):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -b[i] / d                                # b_i + t*d >= 0  =>  bounds on t
        t_hi = np.min(np.where(d < 0, r, np.inf))        # largest +t keeping b' >= 0
        t_lo = np.max(np.where(d > 0, r, -np.inf))       # most negative t
        t = t_hi if abs(t_hi) >= abs(t_lo) else t_lo     # take the larger-magnitude belief move
        if np.isfinite(t):
            out[i] = b[i] + frac * t * d
    return out

# Steered pass without the prefix cache (--no_cache): a chunked KV-cached pass over the full sequence with the
# steering hook at one layer and capture hooks at the readout layers.
def _chunked_steered_readout(wrapper, input_ids, layer, steer_pos, steer_val,
                             readout_layers, chunk_size, device, dtype):
    """Chunked KV-cached forward over the FULL sequence with steering at `layer`.

    steer_pos: sorted abs positions to steer (the last k). steer_val: (len, d) the
    delta ADDED at `layer` (row i -> steer_pos[i]). Captures residuals at every
    read-out layer >= `layer` at steer_pos. Returns {lr: (len, d) cpu fp32}."""
    seq_len = input_ids.shape[1]
    rl = [lr for lr in readout_layers if lr >= layer]
    cap = {lr: [] for lr in rl}
    pos2row = {int(p): i for i, p in enumerate(steer_pos)}
    past_kv = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end].to(device)
        in_chunk = [p for p in steer_pos if start <= p < end]
        hooks = []
        if in_chunk:
            local = torch.tensor([p - start for p in in_chunk], device=device, dtype=torch.long)
            rows = [pos2row[int(p)] for p in in_chunk]
            delta = steer_val[rows].to(device=device, dtype=dtype)        # (m, d)

            def steer_hook(module, inp, out, _local=local, _delta=delta):
                h = out[0] if isinstance(out, tuple) else out
                h[0, _local, :] = h[0, _local, :] + _delta
                return None
            hooks.append(wrapper.get_layer(layer).register_forward_hook(steer_hook))

            def mk(li, _local=local):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[li].append(h[0, _local, :].detach().float().cpu())
                return fn
            for lr in rl:                       # steer hook registered first => capture sees steered
                hooks.append(wrapper.get_layer(lr).register_forward_hook(mk(lr)))
        with torch.no_grad():
            out = wrapper.forward(chunk, past_key_values=past_kv, use_cache=True)
        for h in hooks:
            h.remove()
        past_kv = out.past_key_values
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del past_kv
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {lr: torch.cat(v, 0) for lr, v in cap.items() if v}

# Prefix-KV caching. The intervention only touches the last k positions, so the KV cache of the prefix before the
# intervened window is the same for the clean pass and for every steered pass. It is computed once per sequence, and
# each pass processes only the suffix from a copy of it. The result equals the plain full pass (--no_cache).
def _clone_cache(kv):
    """Deep-copy a KV cache so a steered suffix pass can't mutate the shared prefix.
    copy.deepcopy is version-agnostic (clones the underlying tensors) — robust to the
    transformers cache API changing (DynamicCache internals differ across versions)."""
    if kv is None:
        return None
    if isinstance(kv, tuple):
        return tuple((k.clone(), v.clone()) for (k, v) in kv)
    return copy.deepcopy(kv)

def _run_chunks(wrapper, ids, past, chunk_size, device, hook_fn=None):
    """Run `ids` left-to-right in chunks from cache `past`; hook_fn(start,end,chunk_len)
    may register hooks per chunk and returns a list of handles to remove. Returns the
    final cache."""
    for start in range(0, ids.shape[1], chunk_size):
        end = min(start + chunk_size, ids.shape[1])
        handles = hook_fn(start, end) if hook_fn else []
        with torch.no_grad():
            out = wrapper.forward(ids[:, start:end].to(device), past_key_values=past, use_cache=True)
        for h in handles:
            h.remove()
        past = out.past_key_values
        del out
    return past

def _prefix_and_clean_suffix(wrapper, input_ids, suffix_start, local_pos, capture_layers,
                             chunk_size, device):
    """Clean pass: cache the prefix [0, suffix_start) KV (all layers), then process the
    suffix capturing clean acts at suffix-local positions `local_pos`. Returns
    (prefix_kv, suffix_ids, {layer: (len(local_pos), d) clean acts})."""
    prefix_kv = _run_chunks(wrapper, input_ids[:, :suffix_start], None, chunk_size, device)
    prefix_kv = _clone_cache(prefix_kv)                   # preserve prefix-only for reuse
    suffix_ids = input_ids[:, suffix_start:]
    cap = {l: [] for l in capture_layers}

    def hooks(start, end):
        loc = [p for p in local_pos if start <= p < end]
        if not loc:
            return []
        idx = torch.tensor([p - start for p in loc], device=device, dtype=torch.long)
        hs = []
        for l in capture_layers:
            def mk(li, _idx=idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[li].append(h[0, _idx, :].detach().float().cpu())
                return fn
            hs.append(wrapper.get_layer(l).register_forward_hook(mk(l)))
        return hs

    _run_chunks(wrapper, suffix_ids, _clone_cache(prefix_kv), chunk_size, device, hooks)
    acts = {l: torch.cat(v, 0) for l, v in cap.items() if v}
    return prefix_kv, suffix_ids, acts

def _steered_suffix(wrapper, suffix_ids, prefix_kv, layer, local_pos, steer_val,
                    readout_layers, chunk_size, device, dtype):
    """Steered suffix pass from a fresh copy of the cached prefix. Steers at `layer`
    and captures read-out layers >= `layer` at `local_pos`. Returns {lr: (len, d)}."""
    rl = [lr for lr in readout_layers if lr >= layer]
    cap = {lr: [] for lr in rl}
    pos2row = {int(p): i for i, p in enumerate(local_pos)}

    def hooks(start, end):
        loc = [p for p in local_pos if start <= p < end]
        if not loc:
            return []
        idx = torch.tensor([p - start for p in loc], device=device, dtype=torch.long)
        delta = steer_val[[pos2row[int(p)] for p in loc]].to(device=device, dtype=dtype)
        hs = []

        def steer(module, inp, out, _idx=idx, _delta=delta):
            h = out[0] if isinstance(out, tuple) else out
            h[0, _idx, :] = h[0, _idx, :] + _delta
            return None
        hs.append(wrapper.get_layer(layer).register_forward_hook(steer))
        for lr in rl:                                     # steer hook first => capture sees steered
            def mk(li, _idx=idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[li].append(h[0, _idx, :].detach().float().cpu())
                return fn
            hs.append(wrapper.get_layer(lr).register_forward_hook(mk(lr)))
        return hs

    _run_chunks(wrapper, suffix_ids, _clone_cache(prefix_kv), chunk_size, device, hooks)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {lr: torch.cat(v, 0) for lr, v in cap.items() if v}

# One sequence: the clean pass, the embeddings, the targets, and the steering and decoding loop of the selected mode.
def run_sequence(wrapper, tokenizer, cfg, T_matrices, beliefs_all, tokens_all, seed,
                 layers, k, tok_ids, M, args, device, dtype, pbar):
    n_states = cfg["n_states"]
    beliefs = beliefs_all[seed]
    ntp_all = next_token_probs(beliefs, T_matrices)
    prompt = tokens_to_prompt(tokens_all[seed], cfg["token_names"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, _ = match_positions(input_ids, tok_ids)
    n_matched = min(len(tokens_all[seed]), len(pos_indices))
    if n_matched <= k + 50:
        return []
    if args.donor in ("ntp_matched", "random_matched") and _kerM_dir(M) is None:
        pbar.update(len(layers))
        return []  # full-rank M: no ntp-preserving move

    intv_idx = np.arange(n_matched - k, n_matched)             # the last k HMM-token indices
    # Train/test split of the intervened window. --split random is the paper's protocol (Section 5.1): an interleaved
    # random 20 percent of the positions for fitting, via train_test_split(random_state=seed), and the rest for scoring.
    # --split block instead fits on the first n_train positions and scores after a gap of split_gap positions.
    rng = np.random.default_rng(seed)
    if args.split == "random":
        # The paper's split.
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(k), train_size=args.n_train, random_state=seed)
        tr = np.sort(tr)
        te = np.sort(te)
    else:
        tr = np.arange(args.n_train)
        te = np.arange(args.n_train + args.split_gap, k)

    steer_abs = pos_indices[intv_idx]                          # absolute model positions of intervened
    suffix_start = int(steer_abs[0])                          # prefix [0, suffix_start) is never steered
    local_pos = (steer_abs - suffix_start).astype(int)       # positions within the suffix

    # Clean pass. The prefix before the intervened window is run once and its KV cache is kept for every steered pass;
    # the suffix is run on a copy of that cache, capturing the clean activations at every layer at the intervened
    # positions. --no_cache runs plain full passes instead (same result, slower).
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | clean pass", refresh=True)
    prefix_kv = suffix_ids = None
    if args.no_cache:
        acts, _ = extract_activations_chunked(wrapper, input_ids, layers, steer_abs,
                                              args.chunk_size, device)
    else:
        prefix_kv, suffix_ids, acts = _prefix_and_clean_suffix(
            wrapper, input_ids, suffix_start, local_pos, layers, args.chunk_size, device)
    # The embedding of Section 6.1, emb_l (source to activations), and its encoder enc_l (activations to source), both
    # affine maps fit by least squares on the training positions of this sequence, one pair per layer. The source is the
    # belief state in the paper's runs (--source belief); ntp and log_ntp are alternative sources for the reciprocal
    # experiment (steer the NTP, decode the belief).
    src_tr = torch.tensor(_vals(args.source, beliefs[intv_idx[tr]], M),
                          device=device, dtype=torch.float32)
    src_true_all = torch.tensor(_vals(args.source, beliefs[intv_idx], M),          # true SOURCE at every intervened position
                                device=device, dtype=torch.float32)                # (steering reference, paper Sec. 6.1)
    enc, emb, clean_intv = {}, {}, {}
    for l in layers:
        X = acts.get(l)
        if X is None or X.numel() == 0:
            continue
        X = X.to(device).float()                                # cached path captures to CPU; move to GPU
        enc[l] = _fit(X[tr], src_tr, device)                    # act -> source (last-5k train)
        emb[l] = _fit(src_tr, X[tr], device)                    # source -> act
        clean_intv[l] = X                                       # (k, d) clean acts at all last-k pos
    del acts
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    layers = [l for l in layers if l in enc]

    # Targets at the intervened positions: the original belief, NTP, and log-NTP, and the donor belief. For ntp_matched
    # and random_matched the donor is the original belief moved along the kernel direction of the emission matrix (same
    # NTP, different belief); for past_inconsistent it is the next seed's belief states at the same positions.
    orig = {"belief": beliefs[intv_idx], "ntp": ntp_all[intv_idx]}
    orig["log_ntp"] = np.log(orig["ntp"] + EPS)
    if args.donor in ("ntp_matched", "random_matched"):    # ker(M): same ntp, different belief
        # random_matched has the same targets and probes as ntp_matched; only the direction of the injected vector is
        # randomized in the steering loop below.
        new_bel = {args.donor: _kerM_donor(beliefs[intv_idx], M)}
        conds = [args.donor]
    else:
        donor = (seed + 1) % args.n_seeds
        donor = donor if donor != seed else seed
        new_bel = {"past_inconsistent": beliefs_all[donor][intv_idx]}
        conds = CONDS

    # Every row records the steering layer, the readout layer, the probe kind, the target kind, and whether it is scored
    # against the original or the new value.
    rows = []
    base_info = dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed)

    if args.full_story:
        # The paper's mode (--full_story). Before steering, fresh probes are fit on the clean activations. After steering
        # at each layer, the frozen probe of Section 6.1 (fit on the clean activations to the original targets and applied
        # unchanged) is scored on the held-out positions. Targets: the full belief, the hidden coordinate (the belief's
        # component along the kernel direction, which the NTP cannot see), NTP, and log-NTP, each against the original and
        # the new (target) values, for every (steering layer, readout layer) pair. The probe column labels the rows.
        cond = conds[0]
        nb = new_bel[cond]

        # The unit kernel direction and the targets: belief, NTP, log-NTP, and the hidden coordinate (the belief's component
        # along delta), for the original and the new beliefs.
        dhat = _kerM_dir(M)                                    # unit left-null vec, or None (rank-full)
        def _mk(b):
            b = np.asarray(b, np.float64)
            ntp = b @ M
            d = {"belief": b, "ntp": ntp, "log_ntp": np.log(ntp + EPS)}
            if dhat is not None:
                d["hidden"] = (b @ dhat).reshape(-1, 1)        # scalar coord NTP is blind to
            return d
        targ = {"orig": _mk(beliefs[intv_idx]), "new": _mk(nb)}
        TGT = list(targ["orig"].keys())
        def _aug(X): return torch.cat([X, torch.ones(X.shape[0], 1, device=device)], 1)
        def _pinv(Xa):
            try: return torch.linalg.pinv(Xa)
            except (RuntimeError, NotImplementedError): return torch.linalg.pinv(Xa.cpu()).to(device)
        # Frozen probes: fit on the clean training activations to the original targets, one per (readout layer, target).
        # These are the probes of Section 6.1 that are applied unchanged after steering.
        Pf = {(r, tk): _fit(clean_intv[r][tr],
                            torch.tensor(targ["orig"][tk][tr], device=device, dtype=torch.float32), device)
              for r in layers for tk in TGT}
        # Before steering: a fresh probe on the clean activations, scored against the original and the new targets on
        # the held-out positions.
        for r in layers:
            P = _pinv(_aug(clean_intv[r][tr]))
            Xte = _aug(clean_intv[r][te])
            for tk in TGT:
                for ref in ("orig", "new"):
                    Ytr = torch.tensor(targ[ref][tk][tr], device=device, dtype=torch.float32)
                    pred = (Xte @ (P @ Ytr)).cpu().numpy()
                    rows.append({**base_info, "intervene_layer": -1, "readout_layer": int(r),
                                 "condition": cond, "intervention": "none", "phase": "before",
                                 "probe": "clean", "source_kind": args.source, "target_kind": tk,
                                 "ref": ref, "metric": "clean_R2", "value": _r2(pred, targ[ref][tk][te]),
                                 "n_train": len(tr), "n_test": len(te)})
        # After steering: one steered pass per steering layer, scored with the frozen probes. The random_matched control
        # replaces the direction of the injected vector by a fixed random unit vector u_rand (one draw per parametrization
        # and seed) while keeping the per-position norms of the delta steer (random direction steering, Section 6.1). Its
        # readout is the projection of the activations onto u_rand, stored as fixed_dir rows.
        u_rand = None
        if args.donor == "random_matched":
            g = np.random.default_rng([seed, zlib.crc32(cfg["_label"].encode())])
            u_rand = torch.tensor(g.standard_normal(clean_intv[layers[0]].shape[1]),
                                  device=device, dtype=torch.float32)
            u_rand = u_rand / u_rand.norm()

        # Steering at layer l: the vector of Section 6.1, emb_l(new source) minus emb_l(true source) at every intervened
        # position (or minus emb_l(enc_l(h)) with --steer_ref decoded), added at the intervened positions in a steered suffix
        # pass. The activations are captured at every readout layer r >= l.
        for l in layers:
            new_src = torch.tensor(_vals(args.source, nb, M), device=device, dtype=torch.float32)
            emb_new = predict_probe(new_src, emb[l])
            src_val = (predict_probe(src_true_all, emb[l]) if args.steer_ref == "true"       # the reference emb(eta(x_1:t)) of Section 6.1
                       else predict_probe(predict_probe(clean_intv[l], enc[l]), emb[l]))     # the alternative reference emb(enc(h))
            delta = (emb_new - src_val).detach()
            norms = delta.norm(dim=1, keepdim=True)                 # per-position ker-steer magnitudes
            if u_rand is not None:
                delta = (norms * u_rand[None, :]).detach()          # same energy, random direction
            if args.no_cache:
                cap = _chunked_steered_readout(wrapper, input_ids, l, steer_abs, delta, layers, args.chunk_size, device, dtype)
            else:
                cap = _steered_suffix(wrapper, suffix_ids, prefix_kv, l, local_pos, delta, layers, args.chunk_size, device, dtype)

            # Scoring at every readout layer: the frozen probe against the original and the new targets on the held-out
            # positions.
            for r, A in cap.items():
                Ate = A[te].to(device).float()
                if u_rand is None:
                    Atr = A[tr].to(device).float()
                    P = _pinv(_aug(Atr))
                    Ate_a = _aug(Ate)
                for tk in TGT:
                    for ref in ("orig", "new"):
                        if u_rand is None:
                            Ytr = torch.tensor(targ[ref][tk][tr], device=device, dtype=torch.float32)
                            pr = (Ate_a @ (P @ Ytr)).cpu().numpy()
                            rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                         "condition": cond, "intervention": "steer", "phase": "after",
                                         "probe": "retrain", "source_kind": args.source, "target_kind": tk,
                                         "ref": ref, "metric": "retrain_R2", "value": _r2(pr, targ[ref][tk][te]),
                                         "n_train": len(tr), "n_test": len(te)})
                        pf = predict_probe(Ate, Pf[(r, tk)]).cpu().numpy()            # frozen
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                     "condition": cond, "intervention": "steer", "phase": "after",
                                     "probe": "frozen", "source_kind": args.source, "target_kind": tk,
                                     "ref": ref, "metric": "frozen_R2", "value": _r2(pf, targ[ref][tk][te]),
                                     "n_train": len(tr), "n_test": len(te)})
                if u_rand is not None:
                    # Fixed-direction readout: the steered activations projected onto u_rand, scored against the clean
                    # activations' own projection (orig) and against that projection plus the injected magnitudes (new).
                    # At the steering layer itself the new value is exact by construction.
                    pa = (Ate @ u_rand).cpu().numpy()
                    base = (clean_intv[r][te] @ u_rand).cpu().numpy()
                    step = norms[te, 0].cpu().numpy()
                    for ref, lab in (("orig", base), ("new", base + step)):
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                     "condition": cond, "intervention": "steer", "phase": "after",
                                     "probe": "fixed_dir", "source_kind": args.source,
                                     "target_kind": "rand_proj", "ref": ref, "metric": "fixed_R2",
                                     "value": _r2(pa, lab), "n_train": len(tr), "n_test": len(te)})
            del cap
            if device.type == "cuda": torch.cuda.empty_cache()
            pbar.update(1)
        del prefix_kv, suffix_ids
        if device.type == "cuda": torch.cuda.empty_cache()
        return rows

    if args.clean_baseline:
        # Alternative mode (--clean_baseline): only the before-steering rows, from fresh probes on the clean activations
        # with the same donor targets and split as a steered run. No steered passes.
        cond = conds[0]
        nb = new_bel[cond]
        new = {"belief": nb, "ntp": nb @ M}
        new["log_ntp"] = np.log(new["ntp"] + EPS)
        targ = {"orig": orig, "new": new}
        for r in layers:
            Xtr = clean_intv[r][tr]
            Xte = clean_intv[r][te]
            Xtr_aug = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=device)], 1)
            Xte_aug = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=device)], 1)
            try:
                P = torch.linalg.pinv(Xtr_aug)                     # one pinv, reused across all 6 targets
            except (RuntimeError, NotImplementedError):
                P = torch.linalg.pinv(Xtr_aug.cpu()).to(device)
            for tk in SRC:
                for ref in ("orig", "new"):
                    tgt = targ[ref][tk]
                    Ytr = torch.tensor(tgt[tr], device=device, dtype=torch.float32)
                    pred = (Xte_aug @ (P @ Ytr)).cpu().numpy()
                    rows.append({**base_info, "intervene_layer": -1, "readout_layer": int(r),
                                 "condition": cond, "intervention": "none", "phase": "before",
                                 "source_kind": args.source, "target_kind": tk, "ref": ref,
                                 "metric": "clean_R2", "value": _r2(pred, tgt[te]),
                                 "n_train": len(tr), "n_test": len(te)})
        pbar.update(len(layers))
        del prefix_kv, suffix_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    if args.frozen:
        # Alternative mode (--frozen): frozen probes only, fit on the clean training activations to the original targets
        # and read on the clean (before) and steered (after) held-out activations.
        cond = conds[0]
        nb = new_bel[cond]
        new = {"belief": nb, "ntp": nb @ M}
        new["log_ntp"] = np.log(new["ntp"] + EPS)
        targ = {"orig": orig, "new": new}
        Pf = {(l, tk): _fit(clean_intv[l][tr],
                            torch.tensor(orig[tk][tr], device=device, dtype=torch.float32), device)
              for l in layers for tk in SRC}
        # Before steering: the frozen probes on the clean activations (independent of the steering layer).
        for r in layers:
            for tk in SRC:
                pc = predict_probe(clean_intv[r][te], Pf[(r, tk)]).cpu().numpy()
                for ref in ("orig", "new"):
                    rows.append({**base_info, "intervene_layer": -1, "readout_layer": int(r),
                                 "condition": cond, "intervention": "none", "phase": "before",
                                 "source_kind": args.source, "target_kind": tk, "ref": ref,
                                 "metric": "frozen_R2", "value": _r2(pc, targ[ref][tk][te]),
                                 "n_train": len(tr), "n_test": len(te)})
        # After steering at each layer: the same frozen probes on the steered activations.
        for l in layers:
            new_src = torch.tensor(_vals(args.source, nb, M), device=device, dtype=torch.float32)
            emb_new = predict_probe(new_src, emb[l])
            src_val = (predict_probe(src_true_all, emb[l]) if args.steer_ref == "true"       # the reference emb(eta(x_1:t)) of Section 6.1
                       else predict_probe(predict_probe(clean_intv[l], enc[l]), emb[l]))     # the alternative reference emb(enc(h))
            delta = (emb_new - src_val).detach()
            if args.no_cache:
                cap = _chunked_steered_readout(wrapper, input_ids, l, steer_abs, delta,
                                               layers, args.chunk_size, device, dtype)
            else:
                cap = _steered_suffix(wrapper, suffix_ids, prefix_kv, l, local_pos, delta,
                                      layers, args.chunk_size, device, dtype)
            for r, A in cap.items():
                Ate = A[te].to(device).float()
                for tk in SRC:
                    pa = predict_probe(Ate, Pf[(r, tk)]).cpu().numpy()
                    for ref in ("orig", "new"):
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                     "condition": cond, "intervention": "steer", "phase": "after",
                                     "source_kind": args.source, "target_kind": tk, "ref": ref,
                                     "metric": "frozen_R2", "value": _r2(pa, targ[ref][tk][te]),
                                     "n_train": len(tr), "n_test": len(te)})
            del cap
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pbar.update(1)
        del prefix_kv, suffix_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    for l in layers:
        for cond in conds:
            nb = new_bel[cond]                                  # (k, n_states) donor belief
            # Default mode (without --full_story): the steering vector at layer l is emb(new source) minus the reference,
            # and the probes are fit on the steered training activations.
            new_src = torch.tensor(_vals(args.source, nb, M), device=device, dtype=torch.float32)
            emb_new = predict_probe(new_src, emb[l])
            src_val = (predict_probe(src_true_all, emb[l]) if args.steer_ref == "true"       # the reference emb(eta(x_1:t)) of Section 6.1
                       else predict_probe(predict_probe(clean_intv[l], enc[l]), emb[l]))     # the alternative reference emb(enc(h))
            delta = (emb_new - src_val).detach()                # (k, d)

            if args.no_cache:
                cap = _chunked_steered_readout(wrapper, input_ids, l, steer_abs, delta,
                                               layers, args.chunk_size, device, dtype)
            else:
                cap = _steered_suffix(wrapper, suffix_ids, prefix_kv, l, local_pos, delta,
                                      layers, args.chunk_size, device, dtype)
            new = {"belief": nb, "ntp": nb @ M}
            new["log_ntp"] = np.log(new["ntp"] + EPS)
            for lr, A in cap.items():                           # A: (k, d) cpu, read-out >= l
                Atr = A[tr].to(device).float()
                Ate = A[te].to(device).float()
                # One pseudo-inverse of the augmented training activations (ones column for the bias) is shared by all six
                # targets; it equals fit_probe per target.
                Atr_aug = torch.cat([Atr, torch.ones(Atr.shape[0], 1, device=device)], 1)
                Ate_aug = torch.cat([Ate, torch.ones(Ate.shape[0], 1, device=device)], 1)
                try:
                    P = torch.linalg.pinv(Atr_aug)
                except (RuntimeError, NotImplementedError):
                    P = torch.linalg.pinv(Atr_aug.cpu()).to(device)
                for tk in SRC:
                    for ref, tgt in (("orig", orig[tk]), ("new", new[tk])):
                        Ytr = torch.tensor(tgt[tr], device=device, dtype=torch.float32)
                        pred = (Ate_aug @ (P @ Ytr)).cpu().numpy()   # reuses the single pinv
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(lr),
                                     "condition": cond, "intervention": "steer",
                                     "source_kind": args.source, "target_kind": tk, "ref": ref,
                                     "metric": "retrain_R2", "value": _r2(pred, tgt[te]),
                                     "n_train": len(tr), "n_test": len(te)})
            del cap
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pbar.update(1)
    del prefix_kv, suffix_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows

def main():
    # Command-line options. The paper's runs are --full_story --split random with the default k = 5,000, n_train = 1,000,
    # 10 seeds, all layers, and one of the three donors; scripts/run_interventions.sh passes them, with --sdp_backend math
    # for the Gemma 4 checkpoints.
    P = argparse.ArgumentParser(description="Belief-state steering with frozen probes (Section 6)")
    P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Strata", "Wing"])
    P.add_argument("--params", nargs="+", default=None, help="explicit param labels (else REPRESENTATIVES)")
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--k", type=int, default=5000, help="final tokens intervened + decoded (the last 5k)")
    P.add_argument("--n_train", type=int, default=1000,
                   help="number of training positions among the last k; the embedding, the encoder, and every probe are fit "
                        "on them and the rest are held out")
    P.add_argument("--split_gap", type=int, default=100,
                   help="decorrelating gap (>=tau) between the train block and the test block")
    P.add_argument("--source", choices=["belief", "ntp", "log_ntp"], default="belief",
                   help="subspace to STEER; decode targets are always belief/ntp/log_ntp. "
                        "ntp/log_ntp = the reciprocal test (steer ntp -> decode belief)")
    P.add_argument("--donor", choices=["past_inconsistent", "ntp_matched", "random_matched"],
                   default="past_inconsistent",
                   help="the target beliefs. past_inconsistent takes the next seed's belief states at the same positions. "
                        "ntp_matched moves each belief along the kernel direction of the emission matrix, which keeps the NTP "
                        "and changes the belief; it needs a rank-deficient emission matrix (Arch, Wing, Strata), and Mess3 is "
                        "skipped. random_matched is the control of ntp_matched: the same targets and probes, but the injected "
                        "vector is a fixed random unit direction (one draw per parametrization and seed) with the delta "
                        "steer's per-position norms; it runs with --full_story only")
    P.add_argument("--frozen", action="store_true",
                   help="frozen probes only: fit probes from the clean training activations to belief, NTP, and log-NTP, and "
                        "read the same probes on the clean activations (before) and the steered activations (after)")
    P.add_argument("--full_story", action="store_true",
                   help="the paper's mode: before-steering rows from fresh probes on the clean activations, and after-steering "
                        "rows from the frozen probes, for the belief, its kernel coordinate, NTP, and log-NTP, "
                        "against the original and the target values, at every (steering layer, readout layer) pair")
    P.add_argument("--clean_baseline", action="store_true",
                   help="before-steering rows only: fresh probes on the clean training activations, scored on the held-out "
                        "clean activations against the original and the target values, with no steered passes")
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--layers", type=int, nargs="+", default=None, help="intervention layers (default: all)")
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--no_cache", action="store_true",
                   help="disable prefix-KV caching (plain full passes; for diff validation)")
    P.add_argument("--device", default="cuda")
    P.add_argument("--split", choices=["block", "random"], default="block",
                   help="train/test split of the intervened window. random is the paper's protocol: an interleaved random "
                        "20 percent via train_test_split(random_state=seed), with an rsplit_ tag in the output file name. block "
                        "fits on the first n_train positions and scores after a gap of split_gap positions")
    P.add_argument("--smoke", action="store_true")
    P.add_argument("--sdp_backend", default="default", choices=["default", "math"],
                   help="math disables the flash, memory-efficient, and cudnn attention kernels so that every attention call uses the "
                        "reference kernel. The paper's Gemma 4 runs use it, because the sliding-window layers of Gemma 4 E2B "
                        "otherwise hit a defect of the memory-efficient kernel")
    P.add_argument("--steer_ref", default="true", choices=["true", "decoded"],
                   help="reference of the steering vector emb(new) - emb(reference): 'true' uses the true source value at each "
                        "intervened position (the definition of Section 6.1); 'decoded' uses the decoded value enc(h_clean) instead")
    args = P.parse_args()

    # Attention kernel selection, and the consistency check of the random_matched control.
    if args.sdp_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("sdpa kernels restricted to MATH", flush=True)
    if args.donor == "random_matched" and not args.full_story:
        P.error("--donor random_matched requires --full_story (the control is defined for the story run)")

    # Smoke mode is a plumbing test: short sequences, a 1,000-position window, two seeds, and two layers.
    if args.smoke:
        args.seq_len = 4000
        args.k = 1000
        args.n_train = 200
        args.n_seeds = 2
        if not args.all_params:
            args.families = args.families[:1]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model and read its dtype, which the steering vectors are cast to.
    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    dtype = next(wrapper.model.parameters()).dtype
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    # Progress accounting, then the output file name: belief_steering_<tag><model>.csv, where the tag records the mode,
    # the donor, and the split. The paper's runs (--full_story --split random) produce
    # belief_steering_story_<donor>_rsplit_<model>.csv. An existing file is resumed: finished (hmm, param, seed) triples
    # are skipped.
    combos = sum((len(HMMS[f]["params"]) if args.all_params else (len(args.params) if args.params else 1))
                 for f in args.families if f in HMMS)
    pbar = tqdm(total=combos * args.n_seeds * len(layers), desc="decode_destroy", smoothing=0.05)
    tag = "" if args.source == "belief" else f"{args.source}_"
    if args.donor == "ntp_matched":
        tag = "kerm_"
    if args.frozen:
        tag = "swap_"
    if args.clean_baseline:
        tag = "cleanbase_"
    if args.full_story:
        tag = f"story_{args.donor}_"
    if args.split == "random":
        tag += "rsplit_"
    out_csv = os.path.join(args.output_dir, f"belief_steering_{tag}{ms}.csv")
    all_rows, done = [], set()
    if os.path.exists(out_csv):                              # RESUME: keep finished (hmm,param,seed)
        prev = pd.read_csv(out_csv)
        all_rows = prev.to_dict("records")
        done = set(zip(prev["hmm"], prev["param"], prev["seed"]))
        tqdm.write(f"resume: {len(done)} (hmm,param,seed) already done -> skipping")

    # Per parametrization: all sequences and belief states are computed up front, because the past-inconsistent donor of
    # each sequence is the next seed's sequence. The rank and condition number of the emission matrix are stored with
    # every row.
    for hmm_name in args.families:
        cfg = HMMS.get(hmm_name)
        if not cfg:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        params = (cfg["params"] if args.all_params else
                  [pp for pp in cfg["params"] if cfg["label_fn"](pp) in args.params] if args.params else [REPRESENTATIVES[hmm_name]])
        for param in params:
            label = cfg["label_fn"](param)
            cfg = {**cfg, "_name": hmm_name, "_label": label}
            T_matrices = cfg["fn"](*param)
            T_stack = np.stack(T_matrices)
            pi = stationary_distribution(T_matrices)
            M = emission_matrix(T_matrices)
            sv = np.linalg.svd(M, compute_uv=False)
            mrank = int((sv > sv.max() * 1e-8).sum())
            mcond = float(sv.max() / sv[sv > 0].min()) if (sv > 0).any() else float("inf")
            beliefs_all, tokens_all = {}, {}
            for s in range(args.n_seeds):
                tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=s).astype(np.int64)
                tokens_all[s] = tk
                beliefs_all[s] = full_bayesian_beliefs(tk, T_stack, pi)
            tqdm.write(f"===== {hmm_name} {label} | M rank {mrank}/{M.shape[0]} cond {mcond:.2f} =====")
            for seed in range(args.n_seeds):
                if (hmm_name, label, seed) in done:          # already computed -> skip (resume)
                    pbar.update(len(layers))
                    continue
                rows = run_sequence(wrapper, tokenizer, cfg, T_matrices, beliefs_all, tokens_all,
                                    seed, layers, args.k, tok_ids, M, args, device, dtype, pbar)
                for r in rows:
                    r["M_rank"] = mrank
                    r["M_cond"] = mcond
                all_rows += rows
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    pbar.close()
    print("Done.")

if __name__ == "__main__":
    main()
