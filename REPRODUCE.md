# Reproducing the final submission

`solution.train()` retrains every model behind our final submission from the organizers' raw data
release and writes the submission zip. This document gives the exact procedure, the environment,
the hardware we used, and the numbers a reproduction should land on. The method itself is in
[METHOD.md](METHOD.md).

## What is reproduced

Codabench submission 948680, our 20th and final upload
(`sub38_denseall_k3_cfk3_vinf102_anch010_fp16_gcf_timed.zip`, sha256 `cc3ef1db…`, the copy under
`weights/`). Its leaderboard scores:

| final_score | rel_l2 | tke | mvpe | time | sps |
|---|---|---|---|---|---|
| 81.805 | 94.883 | 78.298 | 93.856 | 91.712 | 44.339 |

## Procedure

With Docker, from the repository root:

```bash
docker build -t realpde-math4ai .
docker run --gpus all -v /path/to/raw:/data -v /path/to/output:/output realpde-math4ai \
    python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

Without Docker, build the same environment by hand. The project file deliberately leaves the
versions open so the code can be imported under newer torch; the reproduction is defined by these
pins:

```bash
python3.11 -m venv venv && . venv/bin/activate
pip install torch==2.2.2 numpy==1.26.4 h5py==3.11.0
pip install -e .
TORCHDYNAMO_DISABLE=1 python -c "import solution; solution.train('/path/to/raw', 'output', 'cuda')"
```

`/path/to/raw` must contain the release's `train_real/` and `train_sim/` folders of HDF5 files.
`scripts/subsample_data.py` expects each file to hold top-level datasets `u`, `v` (and `p` for
simulation) of shape `(T, 64, 128)` and the scalars `re` and `aoa`; this is the layout of the
release we trained on. The organizers' kit README mentions an older tarball layout with the fields
under `measured_data/`; such files would need to be flattened first. The release is obtained from
the organizers and is not part of this repository.

`train()` runs every stage of METHOD.md §4 through the same scripts we used during the competition
(`scripts/train.py`, `scripts/train_width_head.py`, `scripts/fit_bounds.py`,
`scripts/build_submission.py`, `submission/rezip_fast.py`); `solution.py` only sequences them. It
writes:

- the submission zip, copied to the top of the output directory, and `output/submission/`, the
  same zip unpacked (`submission.py`, three member checkpoints, `width_head.pth`, `bounds.json`,
  `ensemble.json`, the vendored model code);
- `output/MANIFEST.json`: sha256 of every delivered file, library versions, wall-clock hours, and
  the seed-screening values;
- `output/logs/<stage>.log`, one per training run, and `output/work/` with the 32x64 data stores
  and every run directory (`log.jsonl`, `meta.json`).

`load_predictor('output', 'cuda')` imports the produced `submission.py` and returns an object whose
`predict(x)` matches the Codabench interface.

A stage whose checkpoint already exists is skipped, so an interrupted run can be restarted with the
same command.

Known limitations of the wrapper: intermediate zips are built under the checkout's
`submission/dist/` before the result is copied to the output directory, so two concurrent runs from
one checkout would interfere; and the produced `submission.py` selects CUDA whenever it is
available, whatever device string is passed to `load_predictor`.

Environment variables:

- `RPDE_FINAL_PACK=N`: how many training processes share the GPU at once. Default 1. Three at a
  time on an RTX 4090 was no faster and made every real-data model slightly worse.
- `RPDE_FINAL_SMOKE=1`: a few-minute pass through every stage on a handful of batches, to check the
  interface. Its output is not a model.

## Environment

- Base image `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` (Python 3.10, torch 2.2.2+cu121,
  numpy 1.26.4), plus `h5py==3.11.0`. This is the image the competition grader ran.
- Everything else is vendored: Flower under `src/pde_competition/_vendor/`, einops under
  `kit/_vendor/einops`.
- `torch.compile` is disabled (`TORCHDYNAMO_DISABLE=1`), matching how every shipped model was
  trained and how the submission zip runs. Weights & Biases is not used.

## Data

- `train_sim/*.h5`: all files except `13950_15.h5`, whose pressure field is corrupt.
- `train_real/*.h5`: all files.
- The raw 64x128 fields are strided by 2 to 32x64 at the four sub-pixel offsets. Only the (0,0)
  offset is the grader's grid; the other three serve as extra real training windows (METHOD.md §3).
- No official baseline checkpoint is used. Normalisation statistics are computed from the training
  data and stored in the checkpoints.

## Randomness and selection

Seeds: sim pretraining 1; teachers 1 to 8, screened to five; cross-fit folds 1, 2, 3 per fold;
held-out model 22; width head 1; students the three lowest surviving teacher seeds.
`torch.manual_seed` and `numpy.random.seed` are set per run, cuDNN is left non-deterministic, so a
retrained model is a new draw from the same recipe. The run-to-run spread is in METHOD.md §4.

All selection happens inside `train()` on training data only. Epochs are fixed (stages 2 to 4 keep
epoch 36 of 40, the students epoch 10 of 10, the sim pretraining its lowest in-sample validation
rel_l2 epoch). A teacher seed is dropped when its epoch-36 in-sample rel_l2 exceeds the median of
the eight by more than 0.00075 and more than 3 MAD; up to five survivors with the lowest seed
numbers are the teachers. A student is left out of the shipped mean only if it diverged (final
rel_l2 at or above 0.080, or a spike above 0.10). `MANIFEST.json` records the screened values and
the seeds that shipped.

## Hardware and run time

The submitted models were trained between 2026-09-16 and 2026-09-26 on one TITAN RTX 24 GB (sim
pretraining and the teachers) and RTX 3090 24 GB cluster nodes (folds, held-out model, width head,
students; three or four processes per GPU). Each run's `meta.json` and `log.jsonl` under
`dev_logs/` records host, torch version, argv and per-epoch metrics. Summed process time was about
6.9 h, about 4 GPU-hours. Peak GPU memory per training process was under 4 GB.

Our own reproduction of `train()` on one RTX 4090, under the image's torch 2.2.2 and with the
default `RPDE_FINAL_PACK=1`, took 1.79 h wall clock (subsampling 11 min, sim pretraining 18 min,
the 18 real-data runs about 3.5 min each, width head about 10 min, students about 5 min each) with
a peak GPU memory of 1.5 GB and up to 63 GB of host memory in use.

## What a reproduction should land on

Because cuDNN is non-deterministic, a reproduction is a fresh draw from the recipe, not a bitwise
copy. In our reproduction:

- The teacher screen kept seeds 1 2 3 4 7 8 (epoch-36 in-sample rel_l2 0.0781 to 0.0789) and
  dropped 5 and 6 (0.0801), so the teachers were seeds 1 2 3 4 7 and the students 1 2 3. In the
  submitted run the survivors were 1 3 4 5 6 and the students 1 3 4.
- Student in-sample rel_l2: 0.0772 to 0.0775 (submitted run 0.0772 to 0.0773).
- Fitted fallback half-widths: 0.00843 / 0.00569 (submitted run 0.00828 / 0.00569).
- Local five-subscore proxy (`scripts/score_local.py`, rows 13950 and 26700, in-sample for
  all-data models): 84.82 for the reproduction and 84.82 for the shipped zip, with subscores
  rel_l2 96.28 / 96.28, tke 80.65 / 80.58, mvpe 96.69 / 96.72, sps 52.23 / 52.23.
- The CUDA-graph fast path was live and agreed with the eager path to 1.9e-6.

Three earlier reproductions with three processes sharing the GPU scored 84.60, 84.68 and 84.69,
with every real-data model about 0.001 worse in-sample, which is why the default is one process.

## Tests

`python -m unittest discover -s tests -t .` runs the unit tests on CPU in about a minute. No data
is needed. The `-t .` matters: it imports the tests as a package, whose initialiser disables
`torch.compile` before torch is imported; without it, set `TORCHDYNAMO_DISABLE=1` yourself. Two temporal-lift tolerance tests are known to fail on Apple MPS and on recent torch
versions by a few 1e-6; the temporal lift is not used by the shipped models.
