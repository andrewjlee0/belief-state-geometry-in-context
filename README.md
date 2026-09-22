# Large Language Models Develop Belief State Geometry In-Context

Daniel Balcells\*, Andrew Jun Lee\*, Chirag Rastogi\*, Paul M. Riechers, Adam Shai, and Xavier Poncini†

\*These authors contributed equally and are listed alphabetically.

†Correspondence to xponcini@gmail.com.

This repository contains the code, the figures, and the analysis notebooks behind the paper, which is available on arXiv as [arXiv:2609.17376](https://arxiv.org/abs/2609.17376).

## Abstract

Large language models (LLMs) trained on next-token prediction exhibit remarkable in-context learning (ICL) abilities, yet the representations that support ICL remain poorly understood. We consider such representations in a controlled setting: prompting LLMs with data emitted from hidden Markov models (HMMs) and probing for the corresponding belief state, the posterior distribution over the HMM's hidden states given the observed token history. Across six open-source LLMs prompted with data from 40 HMMs selected for non-trivial belief structure, we find that belief states are linearly decodable from residual stream activations, with peak probe R²-values from 0.83–0.99 across HMM and LLM combinations, ranging from early to late layers. To establish functional relevance, we intervene directly on the probe-identified subspace via patching and steering, resulting in downstream prediction quality on the order of the untampered model, while controls degrade performance substantially. Together, these results provide representation-level evidence that ICL in open-source LLMs approximates optimal Bayesian prediction over a context-inferred generative model. More broadly, our findings extend prior results linking input-distribution structure to activation geometry: from toy networks trained explicitly on HMM data to production-scale LLMs.

## Setup

```bash
pip install torch transformers accelerate numpy pandas scikit-learn scipy numba tqdm matplotlib seaborn jupyter huggingface_hub sentencepiece protobuf
export HF_HOME=/path/to/model/cache
export HF_TOKEN=hf_...          # required for Llama
```

## Replicating Results

All figures of the paper are saved in `figures/`.

To reproduce a section of the paper:
- Run a "runner" script in `scripts/` to execute all experiments of the section you want, and save the results files to `results/`. 
- Then, open and run the corresponding notebook to load those files, and generate and save the figures to `figures/`.

All runner scripts and notebooks:

1. **Section 4.** `bash scripts/run_prediction.sh`, then `notebooks/01_prediction.ipynb`.
2. **Section 5.** `bash scripts/run_probes.sh`, then `notebooks/02_probes.ipynb`.
3. **Sections 6 and 7.** `bash scripts/run_interventions.sh`, then `notebooks/03_interventions.ipynb`.
4. **Section 8.** `bash scripts/run_tuned_lens.sh`, then `notebooks/04_tuned_lens.ipynb`.

Edit the flags in the runner scripts to change the settings. Run a script with `--help` to see the defaults and allowed values.

Note that a single model needs roughly one to two GPU-hours per probe script and several GPU-hours per intervention script on an 80 GB card.

## Directories

The repository has six directories.

- `src/`: shared functions
  - `hmm/definitions.py`: the transition matrices of the four HMM families and their order-1 and order-0 approximations
  - `hmm/core.py`: stationary distributions, sequence sampling, belief states, next-token probabilities, and k-suffix beliefs
  - `metrics/probes.py`: the least-squares probes with a bias term
  - `metrics/kl.py`: the KL divergence
  - `model_utils.py`: model loading, prompt formatting, position matching, activation extraction, and the full-vocabulary KL
- `configs/`
  - `hmm_configs.py`: the 40 HMMs (four families with ten parametrizations each) and the representative parametrization of each family
- `experiments/`
  - Explained below
- `scripts/`
  - Explained above
- `notebooks/`: create figures
  - `01_prediction.ipynb`, `02_probes.ipynb`, `03_interventions.ipynb`, `04_tuned_lens.ipynb`
- `figures/`
  - Location of figures

## Experiments

All experiments share the following experimental design:

1. Sample a 20,000-token sequence from the HMM and compute the exact Bayesian belief state at every position.
2. Write the sequence as space-separated single letters and run the model once with the whole sequence as context.
3. Keep the final 5,000-token window, where the model's predictions have converged.
4. Where a probe is involved, fit OLS on a random 20 percent of that window and score the probe by R² on the other 80 percent, with the split seeded by the sequence seed.

We describe each experiment file below. All files save results in `results/`.

### Section 4. In-context prediction accuracy

How well do the models predict HMM data at all?

- **Filename.** `experiments/prediction/run_kl.py`
- **Explanation.** At every position, compute the KL divergence from the HMM's next-token probabilities to the model's, over the full vocabulary.
- **Saved Outputs.** `kl_<model>.csv`
- **Corresponding Figures.** Figure 2, the left panel of Figure 1 (b), and Figures 17 to 28.

### Section 5. Linear probes for the belief state

How well are belief states linearly decodable from activations?

#### Belief-state probes and geometries

- **Filename.** `experiments/probes/run_belief_probes.py`
- **Explanation.** At every layer, fit probes from the activations to the belief state, with shuffled-belief and random-belief controls. At each parametrization's best layer, record the decoded geometries for the (1) held-out test set, pooled over the ten sequences, (2) probes fit to all 5,000 final positions, and (3) the two controls.
- **Saved Output.** `r2_<model>.csv`, `geom_<model>.npz`, `geompool_all_<model>.npz`, `geompool_insample_<model>.npz`, and `geompool_ctl_<model>.npz`
- **Corresponding Figures.** Figure 3, the middle panel of Figure 1 (b), Figures 11 to 14, and Figures 29 to 34.

#### NTP and log-NTP probes

- **Filename.** `experiments/probes/run_ntp_probes.py`
- **Explanation.** At every layer, fit probes from the activations to the next-token probabilities (NTP) and to the log next-token probabilities (log-NTP), and fit the model-free baselines that regress the belief state on the ground-truth NTP and log-NTP.
- **Saved Output.** `ntp_probes_<model>.csv`
- **Corresponding Figures.** The bottom of Figure 5 and Figures 47 to 52.

#### k-suffix probes

- **Filename.** `experiments/probes/run_ksuffix_probes.py`
- **Explanation.** For k from 1 to 20, fit probes to the belief computed from only the last k tokens, using activations on sequences from the full HMM, from its order-1 approximation, and from its order-0 approximation.
- **Saved Output.** `ksuffix_probes_<model>.csv`
- **Corresponding Figures.** The top of Figure 5 and Figures 41 to 46.

#### Transfer probes

- **Filename.** `experiments/probes/run_transfer_probes.py`
- **Explanation.** For every pair of parametrizations within a family, train a probe on one parametrization's activations to decode the other's belief states on the same sequence, and do the same for ground-truth beliefs.
- **Saved Output.** `transfer_probes_<model>.csv` and `transfer_probes_gt_<model>.csv`
- **Corresponding Figures.** Figure 4 and Figures 35 to 40.

#### Early-context probes

- **Filename.** `experiments/probes/run_early_context_probes.py`
- **Explanation.** At each parametrization's best layer from the belief-probe results, fit probes on 1,000 random positions among the first 5,000 tokens and store their predictions on the remaining early positions.
- **Saved Output.** `early_context/early_context_<model>__<family>__<param>.npz`
- **Corresponding Figures.** Figure 15.

### Section 6. Steering the belief state

How do representations change when the belief-probe subspace is steered?

- **Filename.** `experiments/interventions/run_belief_steering.py`
- **Explanation.** Steer the belief-probe subspace at one layer over the last 5,000 positions toward a donor belief, then decode the belief, the NTP, the log-NTP, and the belief's NTP-invisible component at every later layer, using probes fit before the steer. The three donors are `past_inconsistent` (another sequence's beliefs), `ntp_matched` (a move along the kernel direction of the emission matrix, which changes the belief but not the NTP), and `random_matched` (a random direction with the same norms).
- **Saved Output.** `belief_steering_<donor>_<model>.csv`
- **Corresponding Figures.** Figure 6 and Figures 53 to 70.

### Section 7. Patching and steering the prediction

Do predictions change according to the belief-probe subspace when it is patched or steered?

- **Filename.** `experiments/interventions/run_prediction_interventions.py`
- **Explanation.** Patch or steer the belief subspace at the last k positions of the full sequence, for k of 1, 5, and 10, and measure the KL divergence from the injected belief's next-token probabilities to the model's prediction, next to the KL from the original belief's and to a random-belief control.
- **Saved Output.** `prediction_interventions_<model>.csv`
- **Corresponding Figures.** Figure 7, the right panel of Figure 1 (b), Figure 16, and Figures 71 to 76.

### Section 8. The tuned lens

Are the layers where beliefs are decodable also the layers from which the prediction can be read out early?

- **Filename.** `experiments/tuned_lens/run_tuned_lens.py`
- **Explanation.** Per layer, train an affine lens toward the HMM's next-token probabilities, toward shuffled, random, order-1, and cross-parametrization control targets, and toward the model's own output, and score every lens, together with the untrained logit lens, by the KL to its target on held-out positions.
- **Saved Output.** `tunedlens_<model>.csv`
- **Corresponding Figures.** Figure 8 and Figures 77 to 94.

## Figures

Below is a list of the figures that each notebook generates from the saved outputs in `results/`.

- `01_prediction.ipynb`
  - Figure 2: `kl.pdf`
  - Figure 1 (b), left panel: `fig1_kl_wing.svg`
  - Figures 9 and 10 and the ranges of Table 2: `belief_geometry.pdf` and `all_lambda2_vs_entropy.pdf`
  - Figures 17 to 22: `kl_all_<model>.png`
  - Figures 23 to 28: `crossover_hist_<model>.pdf`
- `02_probes.ipynb`
  - Figure 3: `r2_pooled.pdf`
  - Figure 1 (b), middle panel: `fig1_r2_wing.svg`
  - Figures 11 to 15: `geometry_grid_real_pooled_qwen35_9b.pdf`, `geometry_grid_insample_pooled_qwen35_9b.pdf`, `geometry_grid_shuffle_pooled_qwen35_9b.pdf`, `geometry_grid_random_pooled_qwen35_9b.pdf`, and `geometry_grid_early_5k_first250_pooled_qwen35_9b.pdf`
  - Figures 29 to 34: `r2_controls_<model>.pdf`
  - Figure 4 and Figures 35 to 40: `within_r2.pdf` and `within_r2_<model>.pdf`
  - Figure 5, top, and Figures 41 to 46: `redr2.pdf`, `rep_gap_k20_<model>.pdf`, and `rep_gap_theory_seed_<model>.pdf`
  - Figure 5, bottom, and Figures 47 to 52: `obsprob.pdf`, `ntp_gap_<model>.pdf`, `log_ntp_gap_<model>.pdf`, `ntp_gap_theory_seed_ntp_<model>.pdf`, and `ntp_gap_theory_seed_log_ntp_<model>.pdf`
- `03_interventions.ipynb`
  - Figure 6: `xmap_dtarget_logntp_after_frozen_qwen35_9b_rsplit.pdf`, `ker_dtarget_hidden_after_qwen35_9b_rsplit.pdf`, and `randdir_dtarget_qwen35_9b_rsplit.pdf`
  - Figures 53 to 58, 59 to 64, and 65 to 70: `xmap_summary_rawR2_<model>_rsplit.pdf`, `ker_summary_rawR2_<model>_rsplit.pdf`, and `randdir_summary_rawR2_<model>_rsplit.pdf`
  - Figures 7 and 16: `intervene_main_patch.pdf` and `intervene_main_steer.pdf`
  - Figure 1 (b), right panel: `fig1_intervene_wing.svg`
  - Figures 71 to 76: `intervene_all_<model>.pdf`
- `04_tuned_lens.ipynb`
  - Figure 8: `tunedlens.pdf`
  - Figures 77 to 82, bottom panels: `tuned_concept_corr_<model>.pdf`
  - Figures 83 to 88: `tunedlens_all_<model>.pdf`
  - Figures 89 to 94: `tunedlens_corr_slope_<model>.pdf`
