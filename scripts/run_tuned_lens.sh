#!/bin/bash
# Runner for Section 8: per-layer tuned lenses toward the HMM's next-token probabilities, the control targets, and
# the model's own output, for every model.
#
# Paper section: Section 8 (Decoding predictions from intermediate layers) with Figure 8, and Appendix Q with
# Figures 77 to 94.
#
# Claim: A tuned lens trained toward the ground-truth NTP attains low KL across most layers, and its per-layer KL
# covaries with the belief probe's R².
#
# Experiment: experiments/tuned_lens/run_tuned_lens.py on the four families and all 10 parametrizations per model,
# with the shuffled, random, order-1, and cross-parametrization control lenses.
#
# Saved Outputs: results/tunedlens_<model>.csv for each model, read by notebooks/04_tuned_lens.ipynb.
#
# How the script works: It loops over the six checkpoints of the paper (override with MODELS="...") and writes to
# results/ (override with RESULTS_DIR=...). Usage: bash scripts/run_tuned_lens.sh
set -eu
cd "$(dirname "$0")/.."
MODELS="${MODELS:-Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B google/gemma-4-E4B google/gemma-4-E2B}"
RESULTS_DIR="${RESULTS_DIR:-results}"
for M in $MODELS; do
    python experiments/tuned_lens/run_tuned_lens.py --model "$M" --families Mess3 Arch Wing Strata --all_params --controls shuffle random order1 cross --output_dir "$RESULTS_DIR"
done
