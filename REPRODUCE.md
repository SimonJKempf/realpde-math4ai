# Math4AI: Track 1 training package

This is the reproduction package as delivered to the organizers after the competition, with the
Codabench account details and contact lines removed. `solution.train()` retrains every model behind
the final submission from the raw data release; see [README.md](README.md) for an overview and
[METHOD.md](METHOD.md) for the method.

## Codabench
- Submission id this package reproduces: **948680** (our 20th upload, `sub38_denseall_k3_cfk3_vinf102_anch010_fp16_gcf_timed.zip`,
  sha256 `cc3ef1dbd36217a50f78d729d2e110c4a06c3eaf4db6d403915fc1c704ef1366`, uploaded 2026-09-27)
- final_score of that submission: 81.805455 (rel_l2 94.883388, tke 78.298361, mvpe 93.855544,
  time 91.711667, sps 44.338762)

## How to reproduce
- Build command: `docker build -t <name> .` (no difference).
- Train command: the default,
  `python -c "import solution; solution.train('/data', '/output', 'cuda')"`.
  `train()` runs every stage in `METHOD.md` §4 through the same scripts we used during the competition
  (`scripts/train.py`, `scripts/train_width_head.py`, `scripts/fit_bounds.py`,
  `scripts/build_submission.py`, `submission/rezip_fast.py`); `solution.py` only sequences them. It
  writes `/output/submission/` (the unzipped equivalent of the Codabench zip: `submission.py`, three
  member checkpoints, `width_head.pth`, `bounds.json`, `ensemble.json`, the vendored model code),
  the zip itself, `/output/MANIFEST.json` (sha256 of every delivered file, versions, wall-clock hours),
  `/output/logs/<stage>.log` (one log per training run) and `/output/work/` (the 32x64 data stores and
  every run directory with `log.jsonl` / `meta.json`). `load_predictor('/output', 'cuda')` imports
  `/output/submission/submission.py` and returns an object whose `predict(x)` calls its `predict(x, {})`.
- Optional environment variables: `RPDE_FINAL_PACK` — how many training processes share the GPU at
  once (default 1, one after another; we measured that three at a time on an RTX 4090 was no faster
  and made every real-data model slightly worse, so leave it at 1).
  `RPDE_FINAL_SMOKE=1` — a few-minute run of every stage on a handful of batches, for checking the
  interface; its output is not a model.
- A `/output` that already holds finished stages is resumed (each stage is skipped when its checkpoint
  exists), so an interrupted training can be restarted with the same command.

## Hardware your submitted model was trained on
The run behind submission 20 was spread over three machines over 2026-09-16 .. 2026-09-26 (each stage's
`meta.json` and `log.jsonl` are in `dev_logs/`, with host, torch version, argv and per-epoch metrics):
- GPU model and count: 1x NVIDIA TITAN RTX 24 GB (sim pretraining; the five teachers, one at a time) and
  NVIDIA RTX 3090 24 GB nodes of a Slurm cluster (the nine cross-fit folds, the held-out twin, the width
  head and the three students, three or four processes per GPU).
- CPU: workstation: Intel Xeon (4 loader workers per run); cluster nodes: 2 cores per job.
- Memory: 64 GB requested per cluster job (the data store is ~6 GB in RAM per process); workstation 128 GB.
- Training time on this hardware: sim pretraining 53 min; teachers 8 x 10.4 min; folds 9 x 16.5 min (three
  per GPU); twin 26 min (four per GPU); width head ~20 min; students 3 x 37 min (three per GPU). Sum of
  process times ≈ 6.9 h; GPU-hours ≈ 4.
- Peak GPU memory on this hardware: < 4 GB per training process (three processes fit on one 24 GB card
  with room to spare).

Reproduction of the whole `train()` with this package's code, outside Docker but under the image's
exact torch build (2.2.2+cu121, Python 3.11) on one RTX 4090 24 GB (2026-09-30, `RPDE_FINAL_PACK=1`,
the default): **1.79 h wall clock** (subsampling 11 min over NFS, sim pretraining 18 min, the 18
real-data runs ~3.5 min each, width head ~10 min, students ~5 min each), **peak GPU memory 1.5 GB**,
host memory in use ≤ 63 GB sampled system-wide. On an A800 we expect 2–4 h. The teacher screen kept
seeds 1 2 3 4 7 8 (epoch-36 in-sample rel_l2 0.0781–0.0789; seeds 5 and 6 at 0.0801 were dropped by
the rule), so the teachers were seeds 1 2 3 4 7 and the students 1 2 3 (final in-sample rel_l2
0.0775 / 0.0772 / 0.0774, no spike; the submitted run had 0.0772–0.0773); fitted fallback half-widths
0.00843 / 0.00569 (submitted run 0.00828 / 0.00569). Local five-subscore proxy (`scripts/score_local.py`,
rows 13950/26700, in-sample for all-data models): 84.82 (86.19 / 82.51) for this reproduction vs 84.82
(86.17 / 82.53) for the shipped zip; subscores rel_l2 96.28 vs 96.28, tke 80.65 vs 80.58, mvpe 96.69 vs
96.72, sps 52.23 vs 52.23. Three earlier reproductions with three processes sharing the GPU scored 84.60,
84.68 and 84.69 with every real-data model ~0.001 worse in-sample, which is why the default is 1.
CUDA-graph path live, graph == eager to 1.9e-6.

## Environment
- Base image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` (Python 3.10, torch 2.2.2+cu121, numpy 1.26.4)
- Added packages and versions (pinned the same way in the Dockerfile): `h5py==3.11.0`. Everything else
  is in the image or vendored in the package (`einops` under `kit/_vendor/einops`, Flower under
  `src/pde_competition/_vendor/flower_standalone.py`). `wandb` is not installed and not used
  (`--no-wandb`). `torch.compile` is disabled (`TORCHDYNAMO_DISABLE=1`), matching how every shipped model
  was trained and how the Codabench zip runs.

## Data, weights and code
- Training data used: `/data/train_sim/*.h5` (all files except `13950_15.h5`, whose pressure field is
  corrupt) and `/data/train_real/*.h5` (all files). The raw 64x128 fields are strided by 2 to 32x64 at
  the four sub-pixel offsets; only the (0,0) offset is the grader's grid, the other three are used as
  extra real training windows (see METHOD.md §3). `mean_std_real.pt` is not used; normalisation
  statistics are computed from the training data and stored in the checkpoints.
- Augmentation: the three shifted-grid copies of the real data. Nothing else.
- Official baseline checkpoints used, if any: none. All weights are produced by `train()`.
- Third-party code and its license:
  - Flower model, `src/pde_competition/_vendor/flower_standalone.py`, from
    https://github.com/t-muser/flowers (commit `f1d2cad`), MIT (c) 2026 Till Muser / SADA Laboratory,
    with BSD-3-Clause parts (c) 2024 Polymathic AI — full text in `src/pde_competition/_vendor/LICENSE.flower`.
    Vendored with two minimal, marked patches (`VENDORED FILE, MINIMALLY PATCHED` comments in the file;
    list in `src/pde_competition/_vendor/README.md`): the `border`/`reflection` padding modes made
    reachable from the config, and `n_levels=1` made constructible. The shipped models use `periodic`
    padding and 2 levels, so neither patch changes what they compute.
  - einops, `kit/_vendor/einops`, MIT, as vendored in the organizers' starting kit.
  - `kit/` is the organizers' Track 1 starting kit v9 (scoring code and baseline plumbing); only
    `kit/scoring.py` (the SPS σ constant, local scoring) and `kit/_vendor/einops` are used.
  - Everything else in `src/`, `scripts/`, `submission/`, `tests/` and `solution.py` is ours.

## Randomness
- Seed(s): sim pretraining seed 1; teachers seeds 1–8 (screened to five, see below; 1, 3, 4, 5, 6 in
  the submitted run); cross-fit folds seeds 1, 2, 3 (each of three folds); held-out twin seed 22; width
  head seed 1; students: the three lowest surviving teacher seeds (1, 3, 4 in the submitted run). `torch.manual_seed` and
  `numpy.random.seed` are set per run; cuDNN is left non-deterministic, so a retrained model is a new
  draw from the same recipe (run-to-run spread in METHOD.md §4).
- How runs or checkpoints are selected, if at all: inside `train()`, on training data only, by the rules
  we used during development. Epochs are fixed: stages 2–4 keep epoch 36 of 40, the students epoch 10
  of 10, the sim pretraining its lowest in-sample validation rel_l2 epoch (`best.pth`; those rows are
  training rows). Teacher seeds: all eight are trained; a seed is dropped when its epoch-36 in-sample
  rel_l2 exceeds the median by > 0.00075 and > 3 MAD (a divergence screen), and up to five survivors
  (lowest seed numbers) are kept — in our run seeds 2, 7, 8 were dropped. Students: the three lowest
  surviving seeds; one is left out of the shipped mean only if it diverged (final rel_l2 ≥ 0.080 or a
  spike above 0.10; none did in our run). `/output/MANIFEST.json` records the screened values and the
  seeds that shipped. Everything else is hard-coded in `solution.py`.

## Track 2 only
- Does the method call the LLM gateway: n/a (Track 1 only)

## Notes
- Training log of the run behind the submission: `dev_logs/<run>/log.jsonl` and `meta.json` for all
  19 runs (one line per epoch: train/val rel_l2, val TKE, per-Re validation, learning rate, seconds).
- `python -m unittest discover tests` runs the package's unit tests (431 tests, ~1 min on CPU; two
  temporal-lift tolerance tests are known to fail on Apple MPS only).
- Resource request: none beyond the default (one GPU, 16 cores, 128 GB). Host memory: each training
  process holds the ~6 GB data store; three in parallel plus loaders stay under 40 GB.
