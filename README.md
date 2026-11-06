# Calibrated Intervals and Distilled Ensembles: A Flower Neural Operator for Airfoil Wakes

Team **Math4AI** (Simon Kempf, LMU Munich), NeurIPS 2026 RealPDE Competition, Track 1 (Sim2Real).
Final leaderboard: rank 12, `final_score` 81.81.

Track 1 gives 20 frames of particle-image-velocimetry (PIV) velocity behind an airfoil on a
32x64 grid and asks for the next 20 frames plus a per-element uncertainty interval. Five subscores
(relative L2, turbulent kinetic energy, mean velocity profile, inference time, and interval
sharpness/coverage) are combined into one number.

This repository contains the complete solution: the model and training code, the shipped weights,
a one-command retraining pipeline, and the logs of the runs behind the submission.

## Findings

**The intervals are where the points are.** The organizers' default band of ±5 % of the
prediction covers under 40 % of measured elements, and an element outside its interval scores
zero. Replacing the band with constant half-widths fitted on held-out trajectories, with no change
to the model, gave +5.16 points. All modelling work after that, over two months, added about +4.

**A learned width head beats constant widths, if it is trained out of fold.** A small CNN maps
per-element features (prediction, input statistics, position, frame index) to lower and upper
offsets, trained to maximise a smooth version of the scorer's reward on residuals of cross-fit
models that never saw the row being predicted. Fitting on in-sample residuals makes the intervals
too narrow on the hidden test rows.

**One forward pass for all 20 frames.** A [Flower](https://github.com/t-muser/flowers) U-Net-style
neural operator (Muser et al. 2026) with time folded into channels maps the input window directly to
the output window. No autoregressive rollout, no error accumulation, one kernel launch per window,
and 0.6 M parameters per member. Autoregressive variants, temporal lifting and wider models did not
help.

**Distillation keeps the ensemble gain with fewer members.** Five teachers trained on all real
data are screened by a divergence rule; three students, each initialised from its own teacher, are
trained against the truth and the teacher mean. The three students match the five teachers on the
leaderboard at three fifths of the member evaluations per call.

**The loss should contain the metrics.** Adding the TKE metric and a smooth interval-coverage
term to the relative-L2 loss, predicting the change from the last input frame, and training on
the three shifted-grid sub-pixel phases of the real data each moved the accuracy subscores by a
few tenths of a point. The coverage term trains the point predictor to keep its residuals inside a
fixed band; the interval widths themselves come from the separately trained width head. Both
objectives are written out in [METHOD.md §4.1](METHOD.md#41-the-losses-written-out).

Cumulative effect on the development leaderboard, in the order we found them:

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

[METHOD.md](METHOD.md) describes every stage, hyperparameter and selection rule, and lists what
did not help.

## Using the shipped predictor

The zip under `weights/` is byte-identical to the final Codabench upload. It is self-contained
and needs only `torch` and `numpy`. (Because it is the scored artefact, it was left as uploaded:
it carries the einops licence but not the Flower licence, which is in this repository under
`src/pde_competition/_vendor/LICENSE.flower`; zips built with the current `build_submission.py`
include it.)

```bash
python scripts/predict_zip.py weights/sub38_denseall_k3_cfk3_vinf102_anch010_fp16_gcf_timed.zip x.npy out.npz
```

`x.npy` holds an input array of shape `(N, 20, 32, 64, 3)` in physical units with channels
`[u, v, p]`; `out.npz` receives `prediction`, `lower` and `upper` of the same shape. The
`submission.py` inside the zip is the whole inference path, including the fused three-member
forward, the width head and the CUDA-graph fast path.

## Retraining from the raw data

The reference environment is the Dockerfile, which is the competition grader's image plus h5py:

```bash
docker build -t realpde-math4ai .
docker run --gpus all -v /path/to/raw:/data -v /path/to/output:/output realpde-math4ai \
    python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

`/path/to/raw` is the organizers' data release with its `train_real/` and `train_sim/` folders
of HDF5 files, each with top-level datasets `u`, `v` (and `p` for simulation) of shape
`(T, 64, 128)` plus the scalars `re` and `aoa`. `train()` subsamples it to the 32x64 grid, runs
every training stage (sim pretraining, 8 teachers, 9 cross-fit folds, a held-out model, the width
head, 3 students), fits the bounds and writes the submission zip to the output directory. About
1.8 h on one RTX 4090, under 2 GB of GPU memory. [REPRODUCE.md](REPRODUCE.md) has the native
route, the environment, and the numbers a reproduction should land on.

## Using the pieces separately

- **Interval fitting for any predictor.** `scripts/fit_bounds.py` fits per-channel half-widths
  that maximise coverage × exp(−width/σ) on held-out residuals; the objective and the fit are
  independent of the model. `scripts/check_bounds_fit.py` guards against fitting on one row only.
- **Learned interval widths.** `scripts/train_width_head.py` trains the width head on out-of-fold
  residuals from fold checkpoints; `scripts/oof_width_model.py` and `scripts/oof_disagreement.py`
  hold the data collection and the simpler width models it is compared against.
- **The model wrapper.** `src/pde_competition/model.py` wraps the vendored Flower operator for
  window-to-window prediction, with the delta parametrisation, per-window scaling, the TKE and
  coverage loss terms, and the ensemble distillation loss used by `scripts/train.py`.
- **Local scoring.** `scripts/score_local.py` computes the five official subscores for a
  submission zip on held-out trajectories, using the organizers' scoring code in `kit/`.
- **Caveats for reuse as a library.** The submission module is written for a disposable grader
  process: on import it replaces `torch.compile` with an identity function, freezes the garbage
  collector, and selects CUDA whenever available regardless of the device argument. `solution.py`
  builds intermediate zips under the checkout's `submission/dist/` before copying the result to the
  output directory.
- **Post-hoc corrections.** `scripts/variance_inflation.py` is the reference implementation of
  the fluctuation rescaling and the mean anchor applied to the shipped prediction.

## Repository layout

```
src/pde_competition/   dataset, model wrapper, composite score estimate
  _vendor/             Flower (MIT), vendored with provenance and licence
scripts/               the training pipeline and local scoring (see above)
submission/            submission_flower.py, the predict() entry point that gets zipped
solution.py            train() and load_predictor(): the one-command reproduction
kit/                   the organizers' scoring code and vendored einops, from their starting kit
tests/                 unit tests, CPU, no data needed: python -m unittest discover -s tests -t .
weights/               the final submission zip (three students, width head in fp16, bounds)
dev_logs/              per-epoch logs and metadata of the 19 training runs behind the submission
```

## Credits and licence

Our code is MIT licensed (see [LICENSE](LICENSE)). The Flower model is by Till Muser, Elias
Spitzer, Matti Lassas, Maarten V. de Hoop and Ivan Dokmanić, *Flowers: A Warp Drive for Neural PDE
Solvers*, arXiv:2603.04430, MIT licensed and vendored with two marked patches listed in
[src/pde_competition/_vendor/README.md](src/pde_competition/_vendor/README.md). `kit/` holds the
organizers' scoring code and their vendored einops copy, which remain theirs; their baseline
models are not redistributed here.
