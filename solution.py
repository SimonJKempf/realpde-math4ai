"""RealPDE 2026 Track 1 -- final training package (reproduces Codabench submission #20, sub38).

    train(data_dir, output_dir, device)   trains everything from scratch and writes output_dir/
    load_predictor(output_dir, device)    returns an object whose predict(x) matches Codabench

The method is the sub38 recipe exactly as it was run during the development phase, driven
through the SAME scripts (scripts/train.py, scripts/train_width_head.py, scripts/fit_bounds.py,
scripts/build_submission.py, submission/rezip_fast.py). Nothing is re-implemented here: this
file only sequences the stages, pins their run ids, and relocates data/outputs into the
container's mounts. Every command line below is the one recorded in the corresponding run's
meta.json (experiments/queue/*.sh in the development repository).

Stages, in the order train() runs them (all on the released training data only):

  0  subsample     raw 64x128 h5 -> 32x64 float32 stores: real at all four sub-grid offsets
                   (p00 is what the grader scores; p01/p10/p11 are the same PIV frames sampled
                   half a pixel off, used only as a curriculum mix), and sim at p00.
  1  pretrain      one Flower model on every sim trajectory, 30 epochs, stride 1      -> best.pth
  2  teachers      five all-data real fine-tunes from stage 1 (seeds 1 3 4 5 6), 40 ep -> epoch036
  3  folds         nine cross-fit models of the same recipe, each holding out six Re rows
                   (3 folds x 3 seeds), 40 ep                                        -> epoch036
  4  twin          one held-out model of the same recipe (default holdout 13950/26700), 40 ep,
                   used ONLY to fit the constant fallback widths                      -> epoch036
  5  width head    trained on the nine folds' out-of-fold residuals (all 18 rows)     -> head.pth
  6  students      three stride-2 all-data students distilled from the five teachers, 10 ep
                   (seeds 1 3 4, each initialised from its own-seed teacher)          -> epoch010
  7  package       fit_bounds (twin residuals on Re 13950+26700, stride 10) -> build_submission
                   (k=3 students + width head) -> rezip_fast (fast path, timing, fp16 head,
                   fluct_scale 1.02, mean_anchor 0.10)  -> output_dir/submission/ (unzipped)

Environment knobs (all optional):
  RPDE_FINAL_PACK=N     how many same-recipe runs share the GPU at once. Default 1 (one at a
                        time): on an RTX 4090 three at a time was no faster and shifted every
                        real-data model by ~+0.001 in-sample val (four full runs, 2026-09-30).
  RPDE_FINAL_SMOKE=1    a minutes-long end-to-end run of every stage on a few batches, for
                        checking the interface. Its output is NOT a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent            # /workspace in the image
PY = sys.executable

# ----------------------------------------------------------------------------- the recipe
ARCH = ["--lifting-dim", "128", "--n-levels", "2", "--channels", "3", "--boundary", "periodic",
        "--batch-size", "16"]
# The real-data recipe shared by teachers, folds and the twin (ws_all_m75t09 / cc_cfk3 / cc_stack).
REAL = ARCH + ["--delta", "--mix-phases", "0.75", "--tke-weight", "0.09", "--coverage-weight", "0.4",
               "--stride", "5", "--lr", "3e-3"]
# Teachers: the development run trained EIGHT all-data seeds and screened them with the project's
# divergence rule (directions.md EXR: a member is OUT when its epoch-36 in-sample val rel_l2 exceeds
# the median by more than 0.00075 AND by more than 3 x MAD). Seeds 2, 7, 8 were out; 1 3 4 5 6 were
# the five clean members that shipped. The rule is applied here, inside train(), on the same
# quantity, so a fresh draw is screened the way the shipped one was. The students are the three
# lowest-numbered surviving seeds (1, 3, 4 in the development run), each initialised from its own
# teacher, so a seed that fails the screen never seeds a student either.
TEACHER_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
N_TEACHERS, N_STUDENTS = 5, 3
EXR_ABS, EXR_MAD = 0.00075, 3.0
FOLD_SEEDS = (1, 2, 3)
FOLD_HOLDOUT = {0: (3750, 7575, 11400, 16500, 20325, 24150),
                1: (5025, 8850, 12675, 17775, 21600, 25425),
                2: (6300, 10125, 13950, 19050, 22875, 26700)}
TWIN_SEED = 22
FINAL_NAME = "sub38_denseall_k3_cfk3_vinf102_anch010_fp16_gcf_timed"

SMOKE = os.environ.get("RPDE_FINAL_SMOKE", "") not in ("", "0")
PACK = max(1, int(os.environ.get("RPDE_FINAL_PACK", "1")))

if SMOKE:
    PRE_EPOCHS, REAL_EPOCHS, REAL_PICK, STUD_EPOCHS, HEAD_EPOCHS = 2, 2, 2, 1, 1
    SMALL = ["--max-steps", "4", "--max-val-steps", "2"]
else:
    PRE_EPOCHS, REAL_EPOCHS, REAL_PICK, STUD_EPOCHS, HEAD_EPOCHS = 30, 40, 36, 10, 30
    SMALL = []
EP = f"epoch{REAL_PICK:03d}.pth"
SAVE = sorted({"32", "36", "40", str(REAL_PICK)}, key=int)   # the development runs saved 32/36/40
EP_STUD = f"epoch{STUD_EPOCHS:03d}.pth"
EP_PRE = f"epoch{PRE_EPOCHS:03d}.pth"        # completion marker; the CONSUMED pretrain file is best.pth


# ----------------------------------------------------------------------------- plumbing
class Stage:
    """One subprocess: the command, the file that proves it finished, and its log.
    `done=None` marks a check that must run every time and succeeds on exit code 0."""

    def __init__(self, run_id: str, argv: list[str], done: Path | None = None):
        self.run_id, self.argv, self.done = run_id, argv, done

    def marker(self) -> Path | None:
        return None if self.done is None else self.done.parent / f"DONE.{self.done.name}"

    def finished(self) -> bool:
        """The artefact exists AND the marker _run_batch writes after a clean exit: epochNNN.pth
        is saved mid-run, so the artefact alone would let a restart adopt an interrupted stage."""
        return self.done is not None and self.done.exists() and self.marker().exists()


def _env(work: Path, data_dir: Path, device: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["RPDE_RAW"] = str(data_dir)                  # /data/train_real, /data/train_sim
    env["RPDE_PROCESSED"] = str(work / "processed")
    env["RPDE_OUTPUTS"] = str(work / "outputs")
    env["RPDE_PACK"] = str(PACK)
    # flower_standalone decorates two functions with @torch.compile. On torch 2.2.2 that drives
    # the inductor backend (a C++ compiler and a cache dir the container may not have); with
    # dynamo disabled the functions run eagerly, which is how every shipped model was trained.
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    env["RPDE_DEVICE"] = device                      # honoured by fit_bounds.py (train.py takes --device)                    # per-stage logs stream instead of block-buffering
    return env


def _log(msg: str) -> None:
    print(f"[solution {time.strftime('%H:%M:%S')}] {msg}", flush=True)


PEAK = {"gpu_mb": 0, "host_mb": 0}      # sampled every poll while stages run; reported in MANIFEST.json


def _sample_memory() -> None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        PEAK["gpu_mb"] = max(PEAK["gpu_mb"], *[int(v) for v in out.split() if v.isdigit()], 0)
    except Exception:  # noqa: BLE001 -- no nvidia-smi (CPU smoke), or a slow one: the sample is optional
        pass
    try:
        # Inside a container /proc/meminfo is the HOST's; the cgroup files are this container's.
        for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.current",
                  "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
            if os.path.exists(f):
                PEAK["host_mb"] = max(PEAK["host_mb"], int(open(f).read().strip()) // 2**20)
                break
        else:
            mi = dict(line.split(":")[:2] for line in open("/proc/meminfo") if ":" in line)
            used = (int(mi["MemTotal"].split()[0]) - int(mi["MemAvailable"].split()[0])) // 1024
            PEAK["host_mb"] = max(PEAK["host_mb"], used)
    except Exception:  # noqa: BLE001
        pass


STAGE_CAP_H = float(os.environ.get("RPDE_FINAL_STAGE_CAP_H", "6"))   # no stage takes > ~1.5 h on an A800


def _run_batch(stages: list[Stage], env: dict, logs: Path, pack: int) -> None:
    """Run independent stages, at most `pack` at a time. A stage whose artefact and DONE marker
    exist is skipped (lets an interrupted training resume; a fresh output_dir runs everything).
    A stage running longer than STAGE_CAP_H is killed and counted as failed."""
    todo = [s for s in stages if not s.finished()]
    for s in stages:
        if s.finished():
            _log(f"skip {s.run_id}: {s.done.name} exists")
    running: list[tuple[Stage, subprocess.Popen, float]] = []
    failed: list[str] = []
    while todo or running:
        while todo and len(running) < pack:
            s = todo.pop(0)
            log = open(logs / f"{s.run_id}.log", "a")
            log.write(" ".join(s.argv) + "\n"); log.flush()
            p = subprocess.Popen([PY] + s.argv, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((s, p, time.time()))
            _log(f"start {s.run_id}  (pid {p.pid}, {len(running)}/{pack} on the GPU)")
        time.sleep(5)
        _sample_memory()
        still = []
        for s, p, t0 in running:
            rc = p.poll()
            if rc is None:
                if time.time() - t0 > STAGE_CAP_H * 3600:
                    _log(f"KILLING {s.run_id}: over the {STAGE_CAP_H:g} h stage cap")
                    p.kill(); p.wait(); rc = -9
                else:
                    still.append((s, p, t0)); continue
            mins = (time.time() - t0) / 60
            if rc != 0 or (s.done is not None and not s.done.exists()):
                failed.append(s.run_id)
                tail = (logs / f"{s.run_id}.log").read_text().splitlines()[-15:]
                _log(f"FAILED {s.run_id} rc={rc} after {mins:.1f} min; log tail:\n  " + "\n  ".join(tail))
            else:
                if s.done is not None:
                    s.marker().write_text(f"{s.run_id} rc=0 {mins:.1f} min\n")
                _log(f"done  {s.run_id}  {mins:.1f} min  -> {s.done}")
        running = still
    if failed:
        raise RuntimeError(f"stages failed: {failed} (logs under {logs})")


def _val_curve(run_dir: Path) -> dict:
    """epoch -> in-sample val rel_l2 from the run's log.jsonl (the last launch wins on duplicates)."""
    out = {}
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
            out[int(r["epoch"])] = float(r["val_rel_l2"])
        except (ValueError, KeyError):
            continue
    return out


def _screen_teachers(runs: Path, seeds, epoch: int):
    """EXR on the teachers' epoch-`epoch` in-sample val: OUT when v > median + EXR_ABS AND
    v > median + EXR_MAD * MAD. Returns (kept seeds sorted, {seed: val}, note)."""
    vals = {s: _val_curve(runs / f"ws_all_m75t09_s{s}")[epoch] for s in seeds}
    v = sorted(vals.values())
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    dev = sorted(abs(x - med) for x in v)
    mad = dev[len(dev) // 2] if len(dev) % 2 else 0.5 * (dev[len(dev) // 2 - 1] + dev[len(dev) // 2])
    cut = max(med + EXR_ABS, med + EXR_MAD * mad)
    kept = sorted(s for s in seeds if vals[s] <= cut)
    note = (f"teacher screen at epoch {epoch}: " + " ".join(f"s{s}={vals[s]:.5f}" for s in seeds)
            + f"  median {med:.5f} MAD {mad:.5f} cut {cut:.5f} -> OUT {sorted(set(seeds) - set(kept))}, kept {kept}")
    if len(kept) > N_TEACHERS:
        kept = kept[:N_TEACHERS]                       # lowest seed numbers, as in the development run
        note += f" -> first {N_TEACHERS} by seed: {kept}"
    if len(kept) < N_STUDENTS:
        # Never abort a 24 h training over a screen. Fall back to the N_TEACHERS lowest-val seeds.
        kept = sorted(sorted(seeds, key=vals.get)[:N_TEACHERS])
        note += f" -> FEWER THAN {N_STUDENTS} SURVIVED; falling back to the {N_TEACHERS} lowest-val seeds {kept}"
    return kept, vals, note


def _gate_students(runs: Path, seeds, epochs: int):
    """The SUB38 gate: each student finished, final val < 0.0800 and no spike (max val < 0.100).
    Development values: 0.07721 / 0.07724 / 0.07732, max 0.0777. A student that fails is left out
    of the shipped ensemble (k shrinks) rather than poisoning the mean; if none passes, all ship."""
    ok, note = [], []
    for s in seeds:
        c = _val_curve(runs / f"cc_denseall_s{s}")
        fin, mx = c.get(epochs, float("nan")), max(c.values()) if c else float("nan")
        passed = fin < 0.0800 and mx < 0.100
        note.append(f"s{s}: final {fin:.5f} max {mx:.5f} {'ok' if passed else 'FAIL'}")
        if passed:
            ok.append(s)
    if not ok:
        note.append("NO STUDENT PASSED THE GATE -- shipping all three regardless")
        ok = list(seeds)
    return ok, "student gate: " + "; ".join(note)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _versions() -> dict:
    import numpy, torch
    try:   # nvidia-smi rather than torch.cuda: the parent must not hold a CUDA context for 24 h
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        gpu = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": numpy.__version__, "gpu": gpu, "host": platform.node()}


# ----------------------------------------------------------------------------- train
def train(data_dir, output_dir, device):
    t_start = time.time()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    work = output_dir / "work"
    logs = output_dir / "logs"
    runs = work / "outputs" / "runs"
    for d in (work, logs, runs, work / "processed"):
        d.mkdir(parents=True, exist_ok=True)
    env = _env(work, data_dir, device)
    mode = work / "MODE"                               # a smoke output_dir must never seed a full run
    if mode.exists() and mode.read_text().strip() != ("smoke" if SMOKE else "full"):
        raise RuntimeError(f"{output_dir} holds a {mode.read_text().strip()} run; use a fresh output_dir")
    mode.write_text("smoke" if SMOKE else "full")
    _log(f"train(): data={data_dir} output={output_dir} device={device} pack={PACK} smoke={SMOKE}")
    _log(f"versions: {_versions()}")
    for split in ("train_real", "train_sim"):
        n = len(list((data_dir / split).glob("*.h5")))
        _log(f"{split}: {n} h5 files")
        if n == 0:
            raise FileNotFoundError(f"no .h5 under {data_dir / split}")

    # -- 0. subsample -------------------------------------------------------------------------
    sub = HERE / "scripts" / "subsample_data.py"
    proc = work / "processed"
    _run_batch([
        Stage("subsample_p00", [str(sub)], proc / "train_sim" / "manifest.csv"),
    ], env, logs, 1)
    _run_batch([
        Stage(f"subsample_{ph}", [str(sub), "--phase", ph, "--real-only"],
              proc / f"train_real_{ph}" / "manifest.csv")
        for ph in ("p01", "p10", "p11")
    ], env, logs, 3)

    tr = str(HERE / "scripts" / "train.py")
    common = ["--no-wandb", "--device", device] + SMALL

    # -- 1. sim pretrain ----------------------------------------------------------------------
    pre_id = "ws_pre_allsim_d128_str1"
    pre = runs / pre_id / "best.pth"
    _run_batch([Stage(pre_id, [tr, "--run-id", pre_id, "--name", pre_id, "--split", "train_sim",
                               "--train-all", "--stride", "1", "--epochs", str(PRE_EPOCHS),
                               "--save-epochs", str(PRE_EPOCHS),
                               "--seed", "1", "--lr", "1e-3"] + ARCH + common, runs / pre_id / EP_PRE)],
               env, logs, 1)

    # -- 2/3/4. the real-data recipe: teachers, folds, twin (independent -> packed) -----------
    real = REAL + ["--epochs", str(REAL_EPOCHS), "--init-from", str(pre)] + common
    stages = []
    for s in TEACHER_SEEDS:
        rid = f"ws_all_m75t09_s{s}"
        stages.append(Stage(rid, [tr, "--run-id", rid, "--name", rid, "--seed", str(s), "--train-all",
                                  "--save-epochs", *SAVE] + real,
                            runs / rid / EP))
    for f, held in FOLD_HOLDOUT.items():
        for s in FOLD_SEEDS:
            rid = f"cc_cfk3_fold{f}_s{s}"
            stages.append(Stage(rid, [tr, "--run-id", rid, "--name", rid, "--seed", str(s),
                                      "--holdout-re", *map(str, held),
                                      "--save-epochs", *SAVE] + real,
                                runs / rid / EP))
    twin_id = f"cc_stack_m75t09_s{TWIN_SEED}"
    stages.append(Stage(twin_id, [tr, "--run-id", twin_id, "--name", twin_id, "--seed", str(TWIN_SEED),
                                  "--save-epochs", *SAVE] + real,
                        runs / twin_id / EP))
    _run_batch(stages, env, logs, PACK)
    teacher_seeds, teacher_vals, note = _screen_teachers(runs, TEACHER_SEEDS, REAL_PICK)
    _log(note)
    student_seeds = teacher_seeds[:N_STUDENTS]
    teachers = [str(runs / f"ws_all_m75t09_s{s}" / EP) for s in teacher_seeds]

    # -- 5. width head on the nine folds -------------------------------------------------------
    head = work / "outputs" / "width_head" / "head_cfk3_all_s1.pth"
    _run_batch([Stage("head_cfk3_all_s1",
                      [str(HERE / "scripts" / "train_width_head.py"),
                       "--folds", str(runs / "cc_cfk3_fold*" / EP), "--stride", "20",
                       "--epochs", str(HEAD_EPOCHS), "--drop-z", "--tau", "0.0005", "--width", "48",
                       "--planes", "ext", "--outputs", "2", "--val-re", "--seed", "1",
                       "--name", "head_cfk3_all_s1"], head)], env, logs, 1)

    # -- 6. stride-2 students distilled from the five teachers ---------------------------------
    stages = []
    for s in student_seeds:
        rid = f"cc_denseall_s{s}"
        stages.append(Stage(rid, [tr, "--run-id", rid, "--name", rid, "--seed", str(s)] + ARCH +
                            ["--delta", "--mix-phases", "0.75", "--tke-weight", "0.09",
                             "--coverage-weight", "0.4", "--stride", "2", "--epochs", str(STUD_EPOCHS),
                             "--save-epochs", str(STUD_EPOCHS), "--lr", "3e-4",
                             "--distill-weight", "1.0", "--distill-truth-weight", "1.0", "--distill-tke",
                             "--train-all", "--teacher-ensemble", *teachers,
                             "--init-from", str(runs / f"ws_all_m75t09_s{s}" / EP)] + common,
                            runs / rid / EP_STUD))
    _run_batch(stages, env, logs, PACK)
    shipped_seeds, note = _gate_students(runs, student_seeds, STUD_EPOCHS)
    _log(note)
    students = [runs / f"cc_denseall_s{s}" / EP_STUD for s in shipped_seeds]

    # -- 7. bounds, zip, inference switches ----------------------------------------------------
    sc = HERE / "scripts"
    dist = HERE / "submission" / "dist"
    base_zip = dist / "sub38_denseall_k3.zip"
    final_zip = dist / f"{FINAL_NAME}.zip"
    bounds = students[0].with_name(students[0].stem + ".bounds.json")
    if bounds.exists():
        bounds.unlink()
    _run_batch([Stage("assert_members", [str(sc / "diag" / "assert_members_same_config.py"),
                                          *map(str, students)])], env, logs, 1)
    _run_batch([Stage("fit_bounds", [str(sc / "fit_bounds.py"), "--ckpt", str(runs / twin_id / EP),
                                     "--fit-re", "13950", "26700", "--stride", "10",
                                     "--write-for", str(students[0])], bounds)], env, logs, 1)
    for z in (base_zip, final_zip):
        if z.exists():
            z.unlink()
    _run_batch([Stage("build_submission", [str(sc / "build_submission.py"), "--ckpt", str(students[0]),
                                           "--ensemble", *map(str, students), "--width-head", str(head),
                                           "--name", base_zip.stem], base_zip)], env, logs, 1)
    _run_batch([Stage("rezip_fast", [str(HERE / "submission" / "rezip_fast.py"), str(base_zip), str(final_zip),
                                     "--src", str(HERE / "submission" / "submission_flower.py"),
                                     "--on", "--timing", "--fp16-head", "--fluct-scale", "1.02",
                                     "--mean-anchor", "0.10"], final_zip)], env, logs, 1)

    # -- deliver: the unzipped archive is what load_predictor imports ---------------------------
    sub_dir = output_dir / "submission"
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    with zipfile.ZipFile(final_zip) as z:
        z.extractall(sub_dir)
    shutil.copy2(final_zip, output_dir / final_zip.name)
    manifest = {
        "codabench_submission": {"id": 20, "name": FINAL_NAME, "final_score": 81.805455,
                                 "shipped_zip_sha256": "cc3ef1dbd36217a50f78d729d2e110c4a06c3eaf4db6d403915fc1c704ef1366"},
        "this_zip_sha256": _sha(final_zip), "smoke": SMOKE, "pack": PACK,
        "teacher_vals_at_pick": {f"s{k}": v for k, v in teacher_vals.items()},
        "teacher_seeds": teacher_seeds, "student_seeds": student_seeds, "shipped_student_seeds": shipped_seeds,
        "hours": round((time.time() - t_start) / 3600, 3), "versions": _versions(),
        "peak_gpu_memory_mb_sampled": PEAK["gpu_mb"], "peak_host_memory_used_mb_sampled": PEAK["host_mb"],
        "files": {p.name: _sha(p) for p in sorted(sub_dir.iterdir()) if p.is_file()},
        "bounds": json.loads((sub_dir / "bounds.json").read_text()),
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"wrote {output_dir / final_zip.name} and {sub_dir}/  ({manifest['hours']:.2f} h)")

    # -- self-check: import the delivered module in a clean subprocess and predict two windows -
    _run_batch([Stage("verify_submission", [str(sc / "verify_submission.py"), str(final_zip),
                                            "--python", PY])], env, logs, 1)
    _log("verify_submission PASS; train() finished")


# ----------------------------------------------------------------------------- inference
class Predictor:
    """Wraps the delivered submission.py (the Codabench artefact) in the final-phase interface.
    Import does the one-time work (weights, CUDA-graph capture); predict() is the timed call."""

    def __init__(self, output_dir, device):
        sub_dir = str(Path(output_dir) / "submission")
        if not os.path.exists(os.path.join(sub_dir, "submission.py")):
            raise FileNotFoundError(f"{sub_dir}/submission.py -- run train() first")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        # By file path, under a name of our own: the artefact resolves everything beside itself via
        # __file__ and its own sys.path inserts, and a harness that already holds a module called
        # `submission` (Codabench's ingestion did) must not shadow or be shadowed by it.
        import importlib.util
        spec = importlib.util.spec_from_file_location("rpde_final_submission", os.path.join(sub_dir, "submission.py"))
        submission = importlib.util.module_from_spec(spec)
        sys.modules["rpde_final_submission"] = submission
        spec.loader.exec_module(submission)
        # submission.py freezes and disables the cyclic garbage collector at import (GCF): right for the
        # development grader, which ran every predict() in a fresh process, but this process serves the
        # whole test set, so the collector goes back on. Changes no number the model computes.
        import gc
        gc.enable()
        self._predict = submission.predict
        self.device = device

    def predict(self, input_array):
        return self._predict(input_array, {})


def load_predictor(output_dir, device):
    return Predictor(output_dir, device)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="train, or load and run one prediction")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--predict-only", action="store_true")
    a = ap.parse_args()
    if not a.predict_only:
        train(a.data, a.output, a.device)
    import numpy as np
    p = load_predictor(a.output, a.device)
    x = np.random.randn(2, 20, 32, 64, 3).astype(np.float32); x[..., 2] = 0
    out = p.predict(x)
    print({k: (np.asarray(v).shape, np.asarray(v).dtype) for k, v in out.items()})
