"""Model loading, prompt formatting, position matching, chunked activation extraction with KV caching, and the
full-vocabulary KL divergence.

Paper section: Section 3 (Experimental setup: the LLMs and the formatting and tokenization) and Section 4.1.

Claim: In the paper's words, "We assign the letters F, Q, and V (or F and Q for two-token families) and format each
sequence as a space-separated string, e.g. " F Q V F Q ...". This formatting ensures that the tokenizer splits the
sequence into single-letter tokens, each prefixed by a space. We verify that every LLM's tokenizer correctly aligns
HMM tokens and LLM tokens."

Experiment: This module runs no experiment. Every experiment script loads its model, formats its sequences, matches
HMM tokens to model positions, and extracts the residual stream through these functions.

Saved Outputs: None. This module writes no files.

How the code works: MODEL_CONFIGS lists the six checkpoints of the paper with their architecture family. ModelWrapper
exposes the decoder layers of the Qwen, Llama, and Gemma architectures uniformly and runs the decoder without the
language-model head. load_model loads a checkpoint in float16 with scaled-dot-product attention (except for Gemma)
and returns the wrapper and the tokenizer. tokens_to_prompt writes a sequence as space-separated letters, get_tok_ids
looks up the model token of each letter with its leading space, and match_positions returns the model positions
that carry an HMM token and which token each carries. extract_activations_chunked runs the sequence in chunks with
the KV cache, registers forward hooks on the requested layers for the chunks that contain requested positions, keeps
the residual stream at those positions (the output of the block, which is the activation after the MLP and before
the next block's normalization, Section 2.4), and optionally the final hidden states. compute_fullvocab_kl turns
final hidden states into log-probabilities of the HMM tokens under a softmax over the whole vocabulary (in blocks of
vocabulary rows, with Gemma's final logit softcap) and returns KL(p_true || p_model) at every position, where p_true
lives on the HMM tokens.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The six checkpoints of the paper and their architecture family. Gemma is loaded with its default attention
# implementation; the others use scaled-dot-product attention.
MODEL_CONFIGS = {
    "Qwen/Qwen3.5-9B":        {"family": "qwen",  "sdpa": True},
    "Qwen/Qwen3.5-4B":        {"family": "qwen",  "sdpa": True},
    "meta-llama/Llama-3.1-8B": {"family": "llama", "sdpa": True},
    "meta-llama/Llama-3.2-3B": {"family": "llama", "sdpa": True},
    "google/gemma-4-E4B":      {"family": "gemma", "sdpa": False},
    "google/gemma-4-E2B":      {"family": "gemma", "sdpa": False},
}

# Architecture family of a checkpoint outside the table, from its name.
def _detect_family(name):
    for key in ["gemma", "llama", "qwen"]:
        if key in name.lower(): return key
    return "generic"

# A uniform view of the decoder: its layers (the residual stream is read at their outputs), its hidden size, and a
# forward pass of the decoder alone with an optional KV cache.
class ModelWrapper:
    def __init__(self, model, family):
        self.model = model
        self.family = family
        if family == "gemma":
            self._layers = model.model.language_model.layers
            self.hidden_size = getattr(model.config, 'text_config', model.config).hidden_size
        else:
            self._layers = model.model.layers
            self.hidden_size = model.config.hidden_size
        self.n_layers = len(self._layers)
    def get_layer(self, i): return self._layers[i]
    def forward(self, input_ids, past_key_values=None, use_cache=True):
        return self.model.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)

# Load a checkpoint in float16 (with the Hugging Face token from HF_TOKEN for gated models) and its tokenizer.
def load_model(model_name, device="cuda"):
    cfg = MODEL_CONFIGS.get(model_name, {})
    family = cfg.get("family", _detect_family(model_name))
    token = os.environ.get("HF_TOKEN")
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if token: kw["token"] = token
    if cfg.get("sdpa", family != "gemma"): kw["attn_implementation"] = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    wrapper = ModelWrapper(model, family)
    print(f"Loaded {model_name} ({family}): {wrapper.n_layers} layers, d={wrapper.hidden_size}")
    return wrapper, tokenizer

# The prompt of Section 3: the token letters joined by spaces, with a leading space.
def tokens_to_prompt(tokens, token_names, sep=" "):
    return sep + sep.join(token_names[t] for t in tokens)

# The model positions that carry an HMM token, in order, and the HMM token index at each of them.
def match_positions(input_ids, tok_ids):
    ids = input_ids[0].cpu().numpy()
    tok_id_map = {tid: zi for zi, tid in enumerate(tok_ids)}
    pos, tok = [], []
    for i, tid in enumerate(ids):
        if tid in tok_id_map: pos.append(i); tok.append(tok_id_map[tid])
    return np.array(pos), np.array(tok)

# The model token id of each letter with its leading space.
def get_tok_ids(tokenizer, token_names):
    return [tokenizer.encode(f" {n}", add_special_tokens=False)[-1] for n in token_names]

# The residual stream at the requested layers and positions from one pass over the sequence in chunks with the KV
# cache. Hooks are registered only for chunks that contain requested positions, and each hook keeps the output of
# its block at those positions.
def extract_activations_chunked(wrapper, input_ids, layers, positions, chunk_size=4096, device="cuda", collect_hidden=False):
    """Extract residual stream activations. If collect_hidden=True, also return last hidden state."""
    seq_len = input_ids.shape[1]
    pos_set = set(positions.tolist()) if len(positions) > 0 else set()
    past_kv = None
    acts = {l: [] for l in layers}
    all_hidden = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end].to(device)
        chunk_positions = set(range(start, end))
        need_hooks = bool(chunk_positions & pos_set) if pos_set else False
        hooks = []
        chunk_acts = {}

        if need_hooks:
            for l in layers:
                def make_hook(li):
                    def fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        chunk_acts[li] = h[0]
                    return fn
                hooks.append(wrapper.get_layer(l).register_forward_hook(make_hook(l)))

        with torch.no_grad():
            out = wrapper.forward(chunk, past_key_values=past_kv, use_cache=True)

        for h in hooks: h.remove()

        if need_hooks:
            chunk_range = np.arange(start, end)
            needed = np.isin(chunk_range, positions)
            if needed.any():
                idx = torch.tensor(np.where(needed)[0], device=device)
                for l in layers: acts[l].append(chunk_acts[l][idx])

        if collect_hidden:
            all_hidden.append(out.last_hidden_state[0].half().cpu())

        past_kv = out.past_key_values
        del out, chunk_acts
        torch.cuda.empty_cache()

    del past_kv
    torch.cuda.empty_cache()

    for l in layers:
        acts[l] = torch.cat(acts[l], dim=0).float() if acts[l] else torch.empty(0)
    hidden_cat = torch.cat(all_hidden, dim=0) if collect_hidden else None
    return acts, hidden_cat

# The KL of Section 4.1 at every matched position: the log-softmax over the whole vocabulary is computed in blocks of
# 2,000 vocabulary rows, the HMM tokens' probabilities are read off it, and KL(p_true || p_model) is taken with p_true
# extended by zeros outside the HMM tokens.
def compute_fullvocab_kl(model, hidden_cat, pos_indices, n_matched, ntp_true, tok_ids, device="cuda", family="qwen"):
    """Compute KL(HMM || LLM) using lm_head over full vocabulary for proper normalization."""
    from .metrics.kl import kl_divergence
    W = model.lm_head.weight
    n = min(n_matched, len(ntp_true))
    ntp_llm = np.zeros((n, len(tok_ids)))
    
    # Gemma uses logit softcapping
    cap = None
    if family == "gemma":
        cap = getattr(getattr(model.config, 'text_config', model.config), 'final_logit_softcapping', None)
    
    batch_size = 512
    for b_start in range(0, n, batch_size):
        b_end = min(b_start + batch_size, n)
        h = hidden_cat[pos_indices[b_start:b_end]].to(device).float()
        
        hmm_logits = h @ W[tok_ids].float().T
        if cap is not None: hmm_logits = torch.tanh(hmm_logits / cap) * cap
        
        lse = torch.full((len(h),), float('-inf'), device=device)
        for i in range(0, W.shape[0], 2000):
            partial = h @ W[i:i+2000].float().T
            if cap is not None: partial = torch.tanh(partial / cap) * cap
            lse = torch.logaddexp(lse, torch.logsumexp(partial, dim=-1))
            del partial
        
        log_probs = hmm_logits - lse.unsqueeze(-1)
        ntp_llm[b_start:b_end] = log_probs.exp().detach().cpu().numpy()
        del h, hmm_logits, lse, log_probs
        torch.cuda.empty_cache()
    
    return kl_divergence(ntp_true[:n], ntp_llm)