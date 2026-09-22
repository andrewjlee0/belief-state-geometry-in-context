#!/bin/bash
# Runner for Section 4: the KL divergence between the HMM's next-token probabilities and the LLM's, for every model.
#
# Paper section: Section 4 (Evaluating in-context prediction accuracy) with Figure 2, and Appendices G and H with
# Figures 17 to 28.
#
# Claim: In-context prediction converges, and the converged KL of the LLMs sits well below that of the 0-HMM and
# 1-HMM baselines.
#
# Experiment: experiments/prediction/run_kl.py on the four families, all 10 parametrizations, and 10 seeds per model.
#
# Result: results/kl_<model>.csv for each model, read by notebooks/01_prediction.ipynb.
#
# How the script works: It loops over the six checkpoints of the paper (override with MODELS="...") and writes to
# results/ (override with RESULTS_DIR=...). Usage: bash scripts/run_prediction.sh
set -eu
cd "$(dirname "$0")/.."
MODELS="${MODELS:-Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B google/gemma-4-E4B google/gemma-4-E2B}"
RESULTS_DIR="${RESULTS_DIR:-results}"
for M in $MODELS; do
    python experiments/prediction/run_kl.py --model "$M" --families Mess3 Arch Wing Strata --output_dir "$RESULTS_DIR"
done
