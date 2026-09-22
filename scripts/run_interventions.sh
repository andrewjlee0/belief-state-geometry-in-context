#!/bin/bash
# Runner for Sections 6 and 7: belief steering with frozen probes, and patching and steering of the prediction, for
# every model.
#
# Paper section: Section 6 with Figure 6 and Appendices M, N, and O with Figures 53 to 70; Section 7 with Figure 7,
# Appendix E with Figure 16, and Appendix P with Figures 71 to 76.
#
# Claim: Steering the belief state changes the decodable log-NTP and the belief's NTP-invisible component at later
# layers, and patching or steering the belief subspace shifts the model's prediction toward the injected belief.
#
# Experiment: experiments/interventions/run_belief_steering.py once per (model, donor, family) with the paper's
# settings (--full_story --split random), where the past-inconsistent donor runs on all four families and the delta
# and random-direction donors on the three families whose emission matrix has a kernel; the parts are merged per
# donor. Then experiments/interventions/run_prediction_interventions.py on all parametrizations. The Gemma 4
# checkpoints run with the reference attention kernel, and in float32 for the prediction interventions.
#
# Saved Outputs: results/belief_steering_<donor>_<model>.csv and results/prediction_interventions_<model>.csv, read by
# notebooks/03_interventions.ipynb.
#
# How the script works: It loops over the six checkpoints of the paper (override with MODELS="...") and writes to
# results/ (override with RESULTS_DIR=...). Usage: bash scripts/run_interventions.sh
set -eu
cd "$(dirname "$0")/.."
MODELS="${MODELS:-Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B google/gemma-4-E4B google/gemma-4-E2B}"
RESULTS_DIR="${RESULTS_DIR:-results}"
# The short model key used in the result file names, for example qwen35_9b.
model_short() { local n="${1##*/}"; echo "$n" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr -d '.'; }
for M in $MODELS; do
    ms=$(model_short "$M")
    # The Gemma 4 checkpoints use the reference attention kernel (and float32 for the prediction interventions).
    EXTRA6=""; EXTRA7=""
    case "$M" in google/gemma*) EXTRA6="--sdp_backend math"; EXTRA7="--dtype float32 --sdp_backend math";; esac
    # Section 6: one job per donor and family, then the parts are merged into belief_steering_<donor>_<model>.csv.
    for DONOR in past_inconsistent ntp_matched random_matched; do
        FAMS="Mess3 Arch Wing Strata"; [ "$DONOR" != "past_inconsistent" ] && FAMS="Arch Wing Strata"
        for FAM in $FAMS; do
            OUT="$RESULTS_DIR/belief_steering_parts/${ms}_${DONOR}_${FAM}"; mkdir -p "$OUT"
            python experiments/interventions/run_belief_steering.py --model "$M" --families "$FAM" --all_params --donor "$DONOR" \
                --full_story --split random $EXTRA6 --output_dir "$OUT"
        done
        python3 - "$RESULTS_DIR" "$ms" "$DONOR" <<'PY'
import sys, glob, os, pandas as pd
R, ms, donor = sys.argv[1:4]
parts = sorted(glob.glob(os.path.join(R, "belief_steering_parts", f"{ms}_{donor}_*", "*.csv")))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv(os.path.join(R, f"belief_steering_{donor}_{ms}.csv"), index=False)
print(f"{ms} {donor}: {len(parts)} parts -> {len(df):,} rows")
PY
    done
    # Section 7
    # Section 7: patching and steering of the prediction at full context, all parametrizations.
    python experiments/interventions/run_prediction_interventions.py --model "$M" --families Mess3 Arch Wing Strata --all_params $EXTRA7 --output_dir "$RESULTS_DIR"
done
