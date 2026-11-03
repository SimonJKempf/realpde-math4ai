# Calibrated Intervals and Distilled Ensembles: A Flower Neural Operator for Airfoil Wakes

Team **Math4AI** (Simon Kempf, LMU Munich), NeurIPS 2026 RealPDE Competition, Track 1 (Sim2Real).
Final leaderboard: rank 12, `final_score` 81.81.

Track 1 gives 20 frames of particle-image-velocimetry (PIV) velocity behind an airfoil on a
32x64 grid and asks for the next 20 frames plus a per-element uncertainty interval. Five subscores
(relative L2, turbulent kinetic energy, mean velocity profile, inference time, and interval
sharpness/coverage) are combined into one number.

## The method in four lines

1. **One-shot Flower operator.** A [Flower](https://github.com/t-muser/flowers) U-Net-style
   neural operator with spatial-warping layers (Muser et al. 2026), with the time axis folded into
   channels so one forward pass maps all 20 input frames to all 20 output frames. No autoregressive
   rollout, so no error accumulation and one kernel launch per window. 0.6 M parameters per member.
2. **Calibrated intervals.** The organizers' default band of ±5 % of the prediction covers under
   40 % of measured elements, and an uncovered element scores zero. Fitted constant half-widths gave
   +5.16 points with no retraining; a small convolutional width head trained on out-of-fold
   residuals to maximise a smooth version of the scorer's reward gave the rest.
3. **Distilled ensemble.** Five teachers trained on all real data, screened by a divergence rule,
   then three students each initialised from its own teacher and trained against the truth and the
   teacher mean. Three students match the five-member teacher ensemble at lower inference cost.
4. **Training recipe.** Simulation pretraining, then real data with a loss that adds the TKE metric
   and an interval-coverage surrogate to relative L2; the model predicts the change from the last
   input frame; shifted-grid sub-pixel phases as the only augmentation.

[METHOD.md](METHOD.md) describes every stage, hyperparameter, selection rule and the ablation
table. The short version of what mattered, cumulative on the development leaderboard:

| change | final_score |
|---|---|
| organizers' FNO baseline, default ±5 % band | 72.64 |
| same FNO, fitted constant interval widths | 77.79 |
| Flower, real data only, constant widths | 79.12 |
| sim pretraining, TKE loss term, spatial widths | 80.26 |
| trained width head on out-of-fold residuals | 80.84 |
| coverage surrogate in the loss | 81.35 |
| ensembling, delta parametrisation, phase mix | 81.66 |
| distilled students, mean anchor, fluctuation scale | **81.81** |

## Repository layout

```
src/pde_competition/   the library: dataset, model wrapper, local composite score
  _vendor/             Flower (MIT), vendored with provenance and licence
scripts/               training, bounds fitting, submission packaging, local scoring, diagnostics
submission/            submission_flower.py, the predict() entry point that gets zipped
solution.py            train() and load_predictor(): the organizers' reproduction interface
kit/                   the organizers' Track 1 starting kit v9 (scoring code, baselines)
tests/                 unit tests, no data needed
weights/               the final submission zip (three fp16 students, width head, bounds)
dev_logs/              per-epoch logs and metadata of the 19 training runs behind the submission
METHOD.md              the method, stage by stage
REPRODUCE.md           how to retrain everything from the raw data release (Docker)
```

## Using the shipped predictor

The zip under `weights/` is byte-identical to the final Codabench upload (sha256 `cc3ef1db…`).
It is self-contained: it needs only `torch` and `numpy`.

```bash
python scripts/predict_zip.py weights/sub38_denseall_k3_cfk3_vinf102_anch010_fp16_gcf_timed.zip x.npy out.npz
```

where `x.npy` holds an input array of shape `(N, 20, 32, 64, 3)` in physical units with channels
`[u, v, p]`; `out.npz` receives `prediction`, `lower` and `upper` of shape `(N, 20, 32, 64, 3)`.

## Retraining

```bash
conda create -n realpde python=3.11 "numpy<2" h5py
conda activate realpde && pip install torch==2.2.2 && pip install -e .
python -c "import solution; solution.train('/path/to/raw', 'output', 'cuda')"
```

`/path/to/raw` is the organizers' raw data release with its `train_real/` and `train_sim/`
subfolders; it is not part of this repository. `train()` subsamples it to the 32x64 grid first,
then runs every training stage and writes the submission zip to `output/`.
Retraining the full pipeline (sim pretraining, 8 teachers, 9 cross-fit folds, a held-out twin,
the width head and 3 students) takes about 1.8 h on one RTX 4090 and under 2 GB of GPU memory.
[REPRODUCE.md](REPRODUCE.md) has the Docker route and the exact numbers of our reproduction.

Tests: `python -m unittest discover tests` (CPU, about a minute).

## Credits and licence

Our code is MIT licensed (see [LICENSE](LICENSE)). The Flower model is by Till Muser, Elias
Spitzer, Matti Lassas, Maarten V. de Hoop and Ivan Dokmanić, *Flowers: A Warp Drive for Neural PDE
Solvers*, arXiv:2603.04430, MIT licensed and vendored unchanged apart from two marked patches
listed in [src/pde_competition/_vendor/README.md](src/pde_competition/_vendor/README.md).
`kit/` is the organizers' starting kit and is theirs.
