#!/bin/bash
# Runner for Section 5: belief-state probes and geometries, NTP and log-NTP probes, k-suffix probes, transfer probes,
# and early-context probes, for every model.
#
# Paper section: Section 5 (Linearly probing the belief state geometry) with Figures 3, 4, and 5, Appendix D with
# Figures 11 to 15, and Appendices I to L with Figures 29 to 52.
#
# Claim: Belief states are linearly decodable from the residual stream, the decoded geometry is the belief geometry
# of the HMM, and the belief information exceeds what the two controls, short-suffix beliefs, NTP, and log-NTP can
# explain.
#
# Experiment: The five probe scripts of experiments/probes/ on the four families, all 10 parametrizations, and 10
# seeds per model. The early-context probes take each parametrization's best layer from the belief-probe results.
#
# Result: results/r2_<model>.csv, geom_<model>.npz, geompool_{all,insample,ctl}_<model>.npz, ntp_probes_<model>.csv,
# ksuffix_probes_<model>.csv, transfer_probes_<model>.csv, transfer_probes_gt_<model>.csv, and
# results/early_context/early_context_<model>__<family>__<label>.npz, read by notebooks/02_probes.ipynb.
#
# How the script works: It loops over the six checkpoints of the paper (override with MODELS="...") and writes to
# results/ (override with RESULTS_DIR=...). Usage: bash scripts/run_probes.sh
set -eu
cd "$(dirname "$0")/.."
MODELS="${MODELS:-Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B google/gemma-4-E4B google/gemma-4-E2B}"
RESULTS_DIR="${RESULTS_DIR:-results}"
# The short model key used in the result file names, for example qwen35_9b.
model_short() { local n="${1##*/}"; echo "$n" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr -d '.'; }
for M in $MODELS; do
    ms=$(model_short "$M")
    python experiments/probes/run_belief_probes.py    --model "$M" --families Mess3 Arch Wing Strata --output_dir "$RESULTS_DIR"
    python experiments/probes/run_ntp_probes.py       --model "$M" --families Mess3 Arch Wing Strata --output_dir "$RESULTS_DIR"
    python experiments/probes/run_ksuffix_probes.py   --model "$M" --families Mess3 Arch Wing Strata --output_dir "$RESULTS_DIR"
    python experiments/probes/run_transfer_probes.py  --model "$M" --families Mess3 Arch Wing Strata --output_dir "$RESULTS_DIR"
    # The early-context probes of Figure 15 use each parametrization's best layer from r2_<model>.csv.
    python experiments/probes/run_early_context_probes.py --model "$M" --all_params --early_len 5000 --windows 5000:1000 \
        --layers best --r2_csv "$RESULTS_DIR/r2_${ms}.csv" --no_acts --output_dir "$RESULTS_DIR/early_context"
done
