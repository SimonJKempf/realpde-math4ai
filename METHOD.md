# Math4AI: Track 1 method

## 1. Summary
A single-pass Flower neural operator (Muser et al. 2026) that maps the 20 input frames to all 20 output
frames at once, trained on all sim data first and then on all real PIV data with a loss that adds the
TKE metric and an interval-coverage surrogate to the relative-L2 objective. The shipped predictor is the
mean of three "student" copies distilled from a five-member teacher ensemble, and the uncertainty bounds
come from a small convolutional width head trained on out-of-fold residuals. The bounds (sps_score) are
what matters most for the final score; among the accuracy terms, the delta parametrisation and the TKE
loss term moved it most.

## 2. Model
- **Architecture and number of parameters:** Flower (`flower_standalone.py`, upstream commit `f1d2cad`,
  MIT), a U-Net-style operator whose layers warp their input along learned displacement fields
  (`grid_sample`) before mixing channels; here with `lifting_dim=128`, `n_levels=2`, 16 heads, 16 groups,
  periodic boundary padding, no dropout. **608,796 parameters** per member. The prediction is the mean of
  **3 members** with identical architecture (the three students in §4).
  A separate width head (`WidthHead` in `scripts/train_width_head.py`, 4 conv layers, width 48, 17 input
  planes, 4 outputs, ~49k parameters) produces the interval offsets.
- **Inputs and outputs, preprocessing and normalization:** input `(N, 20, 32, 64, 3)` in physical units;
  time is folded into channels (60 in, 60 out) so one forward pass produces the whole forecast. Each
  window is divided by its own free-stream velocity scale `mean|u|` over measured points (`window_scale`),
  then standardised with per-channel statistics stored as buffers in the checkpoint. The model predicts
  the **change from the last input frame** (`--delta`), which is added back; the pressure channel is
  returned as zeros for real data. The loss is computed over the two measured channels (u, v); PIV points
  outside the field of view or inside the body are exact zeros in the target and are treated as such.
- **Starting point:** random initialisation. No official baseline checkpoint is used.

## 3. Data
- **Training data used and how combined:** `train_sim` (all 100 trajectories minus one, see below) is
  used for a first pretraining stage. `train_real` (all 81 released trajectories) is used for every later
  stage. The raw 64x128 PIV fields are strided by 2 to 32x64 at offset (0,0) — the grader's resolution —
  and additionally at the three other sub-pixel offsets (0,1), (1,0), (1,1) ("phases"), which are the same
  flow sampled half a pixel off. During real training, 75 % of every epoch's windows come from those three
  extra phases and 25 % from the scored offset (`--mix-phases 0.75`).
- **Validation split carved out of the training data:** for the shipped models none — teachers and
  students train on every real trajectory (`--train-all`). The epoch to ship (36 of 40 for teachers, 10
  of 10 for students) and every hyper-parameter were chosen earlier on runs that held out the Reynolds
  rows 13950 and 26700; those runs are not part of the package. Inside `train()` two kinds of held-out
  runs are still trained because the *bounds* need them: nine cross-fit models (three folds of six Re
  rows each, three seeds) whose out-of-fold residuals train the width head, and one model holding out
  rows 13950/26700 whose residuals fit the constant fallback widths.
- **Augmentation:** the three shifted-grid phases above. Nothing else (no flips, noise or jitter).
- **Data left out, and why:** `train_sim/13950_15.h5` — its pressure field is broken (values +60..+241
  where every other sim file is O(1)); `train_real/7575_0.h5` is not in the container and is excluded in
  the code as well.

## 4. Training
Windows are 20 input + 20 output frames cut from each trajectory at a stride (5 frames for teachers and
folds, 2 for students, 1 for sim pretraining). All stages use AdamW (weight decay 1e-4) with a one-cycle
schedule (10 % warm-up), batch size 16, `rel_l2` = mean per-window relative L2 over measured points.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1 sim pretrain (1 model, seed 1) | train_sim, all trajectories, stride 1 | rel_l2 | AdamW, one-cycle, peak 1e-3 | 30 epochs | 16 |
| 2 teachers (8 models, seeds 1–8, screened to 5, see below), init from stage 1 | train_real, all trajectories, stride 5, 75 % phase mix | rel_l2 + 0.09·tke_rel_l2 + 0.4·(1 − smooth coverage at half-widths 0.0093/0.0063) | AdamW, one-cycle, peak 3e-3 | 40 epochs, epoch 36 kept | 16 |
| 3 cross-fit folds (9 models: 3 folds × seeds 1 2 3), init from stage 1 | train_real minus six held-out Re rows per fold, stride 5, phase mix | same as stage 2 | same as stage 2 | 40 epochs, epoch 36 kept | 16 |
| 4 held-out twin (1 model, seed 22), init from stage 1 | train_real minus rows 13950 and 26700, stride 5, phase mix | same as stage 2 | same as stage 2 | 40 epochs, epoch 36 kept | 16 |
| 5 width head (seed 1) | out-of-fold residuals of the nine stage-3 models on all 18 real Re rows, stride 20 | smooth SPS reward: −mean sigmoid((h−|r|)/τ)·exp(−(lo+up)/σ), τ annealed from 5e-4 | AdamW (wd 1e-4), one-cycle, peak 2e-3 | 30 epochs | 64 |
| 6 students (3 models, seeds 1 3 4), each init from the same-seed teacher | train_real, all trajectories, stride 2, phase mix | stage-2 loss against the truth + rel_l2 and 0.09·tke_rel_l2 against the mean of the five teachers (`--distill-weight 1 --distill-truth-weight 1 --distill-tke`) | AdamW, one-cycle, peak 3e-4 | 10 epochs, epoch 10 kept | 16 |

### 4.1 The losses, written out

Three objectives do three different jobs. Only the first trains the point predictor.

**Point predictor (stages 2, 3, 4, and the truth part of 6).** For a batch of predictions p and
targets y, over the scored elements i (measured channels u and v, and y_i ≠ 0, which excludes the
airfoil body and the area outside the field of view):

```
L = rel_l2(p, y) + λ_tke · tke_rel_l2(p, y) + λ_cov · L_cov(p, y)

rel_l2      mean over windows of ‖p − y‖₂ / ‖y‖₂
tke_rel_l2  mean over windows of ‖k(p) − k(y)‖₂ / ‖k(y)‖₂,   k = turbulent kinetic energy field
L_cov       (1/N) · Σ_i [ 1 − σ((h_c − |p_i − y_i|) / τ) ],  σ = logistic sigmoid

λ_tke = 0.09,  λ_cov = 0.4,  τ = 0.002,  h_u = 0.0093,  h_v = 0.0063
```

`L_cov` is the smooth version of the scorer's coverage indicator 1{p_i − h ≤ y_i ≤ p_i + h}. The
band is centred on the prediction with a fixed half-width per channel; the target enters only
through the residual. Its gradient with respect to p_i is proportional to
s(1−s)·sign(p_i − y_i) with s = σ((h − |r|)/τ): an L1-type pull toward the target that is
strongest for residuals near the band edge and vanishes for elements already well inside or
hopelessly outside. The half-widths h_c are the constant interval widths fitted for an earlier
submission, so the predictor is trained to be coverable by the band expected to ship. The code is
`coverage_surrogate` in `src/pde_competition/model.py`; the TKE term is `tke_rel_l2` there, a
differentiable port of the organizers' metric.

**Distillation (stage 6).** With t the mean prediction of the five frozen teachers on the same
augmented input the student sees:

```
L_student = L(p, y) + rel_l2(p, t) + λ_tke · tke_rel_l2(p, t)
```

The TKE term against the teacher mean matters because the mean is smoother than any member, and
matching it in relative L2 alone would pull the student's fluctuation energy down.

**Interval widths (stage 5).** The widths are not part of the predictor's loss. The width head
outputs a lower and an upper offset (lo_i, up_i) per element and is trained on out-of-fold
residuals r_i = y_i − p_i to maximise a smooth version of the scorer's interval reward:

```
R = (1/N) · Σ_i  σ((r_i + lo_i) / τ) · σ((up_i − r_i) / τ) · exp(−(lo_i + up_i) / σ_sps)
```

with τ annealed from 5e-4 and σ_sps the scorer's width constant from `kit/scoring.py`. The two
sigmoids are the smooth indicator of p_i − lo_i ≤ y_i ≤ p_i + up_i; the exponential is the scorer's
own sharpness penalty. The constant fallback widths (§5) maximise the exact, non-smooth reward per
channel by a one-dimensional search in `scripts/fit_bounds.py`.

- **How runs or checkpoints are selected:** by fixed rules applied inside `train()` on training data
  only. Epochs are fixed (stage 1: lowest in-sample validation rel_l2 epoch, `best.pth`; stages 2–4:
  epoch 36; stage 6: epoch 10). Teachers: eight seeds (1–8) are trained and screened with the
  divergence rule we used throughout development — a seed is dropped when its epoch-36 in-sample
  rel_l2 exceeds the median of the eight by more than 0.00075 *and* by more than 3 MAD; up to five
  survivors (lowest seed numbers) are the teachers. In the development run seeds 2, 7, 8 were dropped
  (0.0794, 0.0797, 0.0799 vs median 0.0785) and 1, 3, 4, 5, 6 shipped. Students: the three
  lowest-numbered surviving seeds (1, 3, 4 in development), each initialised from its own teacher. A
  student whose final in-sample rel_l2 is ≥ 0.080 or whose validation curve spiked above 0.10 is left out
  of the shipped mean (development values 0.0772–0.0773, max 0.0777; none was dropped). A guard refuses
  to package members whose configs differ.
- **Seeds, and the run-to-run spread:** teachers seeds 1 3 4 5 6 (in-sample rel_l2 at epoch 40:
  0.0782–0.0784 across the five, 0.0793–0.0797 for the two seeds not shipped), students seeds 1 3 4
  (in-sample 0.0772–0.0773). On held-out rows the replicate spread of one member of this recipe is about
  0.0003 in rel_l2 and about 0.01–0.02 points of the local composite estimate.

## 5. Uncertainty bounds
Returned. For each element `lower = p − lo`, `upper = p + up` where `(lo, up)` for u and v are the four
outputs of the width head (softplus, clamped at 0.5) evaluated on 17 per-frame feature planes: the
ensemble-mean prediction u, v; two zeroed disagreement planes (the head is trained "drop-z", so it never
sees the member spread); the window's mean input speed; normalised x, y and frame index; the input
window's temporal mean and std of u, v; the last input frame u, v; the measured fraction; and the
prediction's frame-to-frame |Δu|, |Δv|. The head is trained to maximise a smooth version of the scorer's
reward, coverage × exp(−width/σ), on out-of-fold residuals so that the widths reflect errors on rows the
predictor never trained on. `bounds.json` also carries constant per-channel half-widths fitted on the
held-out twin's residuals (rows 13950 and 26700, stride 10) as the fallback for any channel the head does
not cover (pressure). The head runs in fp16 at inference (prediction unaffected).

## 6. Inference
- **Track 1, per call:** `numpy -> float32 CUDA tensor`; the three members run in one fused forward
  (their weights are stacked, so one kernel launch sequence serves all three) and are averaged; the width
  head runs on the mean; then two post-hoc transforms are applied to the **prediction only**, after the
  bounds are formed: (i) `fluct_scale` 1.02 — the fluctuation about the prediction's own per-pixel time
  mean is multiplied by 1.02 (the ensemble mean is slightly too smooth); (ii) `mean_anchor` 0.10 — the
  prediction's per-pixel time mean is shifted 10 % of the way toward the input window's masked time mean
  (u, v only). Numeric precision is float32 (no TF32) for the model, fp16 for the width head. On CUDA the
  forward+head for each batch shape (1..16 windows) is captured once into a `torch.cuda.CUDAGraph` at
  import and replayed; larger calls are chunked into 16-window graphs. Any failure of graph capture falls
  back to the eager path, which computes the same numbers (checked to 2e-6). No `torch.compile`
  (`TORCHDYNAMO_DISABLE=1`).

## 7. Speed
Model choice (one forward for all 20 output frames, no autoregression), a small model (0.6 M parameters
per member), distillation from five teachers into three students so that k=3 replaces k=5/6 at almost
the same accuracy, CUDA-graph replay of the fused three-member forward plus head (the model is launch-
bound on an A800: one window ≈ 1 ms), the width head in fp16, all one-time work at import, and — because
the development grader timed each call in a fresh process — the cyclic garbage collector is frozen after
import. In this package `load_predictor` re-enables the collector, since one process serves the whole
test set.

## 8. What mattered
Development leaderboard `final_score` (sim2real test set), cumulative:

| change | our upload no. | final_score | Δ |
|---|---|---|---|
| organizers' FNO baseline, default ±5 % band | 1 | 72.64 | |
| same FNO, fitted constant interval widths | 2 | 77.79 | +5.16 (bounds dominate the score) |
| Flower, 60 epochs, real only, constant widths | 5 | 79.12 | +1.33 |
| lifting 128 / 2 levels, stride-1 windows | 8 | 79.44 | +0.32 |
| sim pretraining on all sim, then all real; TKE loss term; spatial widths | 9 | 80.26 | +0.82 |
| trained width head (out-of-fold residuals, extended planes) | 10 | 80.84 | +0.58 |
| TKE weight 0.08 and coverage surrogate 0.4 in the loss | 11 | 81.35 | +0.51 |
| 3-member ensemble | 12 | 81.50 | +0.15 |
| delta parametrisation (predict the change from the last frame), 6 members | 14 | 81.57 | +0.07 |
| shifted-grid phase mix 0.75; CUDA-graph fast path | 15 | 81.66 | +0.09 |
| cross-fit head matched to the recipe; fluct_scale; fp16 head | 17 | 81.73 | +0.07 |
| 3 distilled students in place of 5 teachers (time) | 18 | 81.75 | +0.03 |
| students at window stride 2; mean anchor 0.10; fluct_scale 1.02 | 20 = Codabench id 948680 (this package) | **81.81** | +0.05 |

Things that did not help and are not in the package: rollout/autoregressive training and inference
blends, temporal lifting, wider models (lifting 256), stride-2 *teachers* (unstable at lr 3e-3), more than
three students, 20-epoch students, MC-dropout / energy-score fine-tuning.

## 9. Compute
Development run behind submission 20 (several machines, run in parallel over two days):
sim pretraining 53 min on one TITAN RTX; the five teachers 10.4 min each on a TITAN RTX; the nine folds
16–17 min each on RTX 3090s, three per GPU; the held-out twin 26 min (four per GPU); the width head
about 20 min; the three students 37 min each on an RTX 3090, three per GPU; torch 2.7.1 (cu118) on the
TITAN machines and 2.13 (cu126) on the 3090s. Peak GPU memory per training process < 4 GB.
Reproduction with this package under the image's torch 2.2.2+cu121 on one RTX 4090: 1.79 h wall clock
with the stages run one at a time (`RPDE_FINAL_PACK=1`, the default), peak GPU memory 1.5 GB. See REPRODUCE.md.

## 10. References
- T. Muser, E. Spitzer, M. Lassas, M. V. de Hoop, I. Dokmanić, *Flowers: A Warp Drive for Neural PDE
  Solvers*, arXiv:2603.04430 (2026). Code: https://github.com/t-muser/flowers (MIT; `flower_standalone.py`
  vendored unchanged, licence in `src/pde_competition/_vendor/LICENSE.flower`).
- einops (MIT), vendored copy from the organizers' starting kit under `kit/_vendor/einops`.
- `kit/scoring.py` from the organizers' Track 1 starting kit v9 (used only to read the SPS width constant
  σ and for local scoring).
- PyTorch 2.2.2, NumPy 1.26, h5py 3.11.
