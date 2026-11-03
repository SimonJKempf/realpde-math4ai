"""Eager vs hand-captured CUDA graphs, through the real submission module.

W21 measured torch.compile(mode="reduce-overhead") at 6x on a 4090 (+0.47
final via time_score). The shipped fast path avoids torch.compile (it scored
submission 3 a zero under torch 2.2.2) and captures torch.cuda.CUDAGraph by
hand inside submission_flower.py. This prices THAT path, per window, at the
two call shapes that matter: one window per call and _BATCH windows per call.
Also checks the graph output equals the eager output.

    python scripts/bench_graph_path.py submission/dist/<zip built by build_submission>

T6/E4 SWEEP (2026-09-21), additive -- the invocation above is unchanged:

    python scripts/bench_graph_path.py <zip> --t6 [mode[:perhead] ...]

prices the alternatives to vmap (T6) and the per-member width head (E4) against
the shipped path, ONE MODE PER PROCESS (the grader runs a fresh process per
call, and a reloaded submission module would share a warmed CUDA context and
another mode's graph pool). Every alternative is checked against the shipped
path's own output at 1e-5, the same bar the graph path has to clear.
"""
import importlib.util, os, sys, time, zipfile, tempfile
import numpy as np, torch
sys.path.insert(0, 'kit')
from scoring import score_time
sys.path.insert(0, 'src')
from pde_competition.composite import estimate

# ---------------------------------------------------------------------------
# T6/E4 sweep. Runs BEFORE the classic bench body and exits, so nothing below
# changes for an invocation that does not ask for it.
# ---------------------------------------------------------------------------
_T6_SIZES = tuple(int(s) for s in os.environ.get("RPDE_T6_SIZES", "1,5,8,16").split(","))
_T6_REPS = int(os.environ.get("RPDE_T6_REPS", "50"))
_T6_DEFAULT_MODES = ("vmap", "loop", "streams", "grouped", "fold1",
                     "vmap:mean", "vmap:median", "vmap:q0.75")


def _t6_load(zip_path, tmp):
    """Extract the zip, then OVERWRITE its submission.py with the working tree's.

    This is not a convenience, it is the fix for a trap that has already voided
    one parcel: a built zip carries a FROZEN COPY of submission_flower.py, so a
    new env switch is simply absent from it and every arm of the sweep runs the
    same code while appearing to test different ones. That is exactly how W200's
    ladder and TF32 arms came back null on sub17/sub18. The weights, bounds.json,
    ensemble.json and the vendored model still come from the zip; only the module
    under test is the one on disk.
    """
    zipfile.ZipFile(zip_path).extractall(tmp)
    live = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "submission", "submission_flower.py")
    with open(live) as f:
        src = f.read()
    with open(os.path.join(tmp, "submission.py"), "w") as f:
        f.write(src)
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location("submission", os.path.join(tmp, "submission.py"))
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    return sub


def _t6_inputs(sizes):
    rng = np.random.default_rng(0)
    xs = {}
    for n in sizes:
        a = rng.normal(0, 0.1, (n, 20, 32, 64, 3)).astype(np.float32)
        a[..., 2] = 0
        xs[n] = a
    return xs


def _t6_single(zip_path, mode, perhead, outdir):
    """One mode, one process: time it at every size and save its n=5 output."""
    import json
    os.environ["RPDE_ENSEMBLE_MODE"] = mode
    os.environ["RPDE_HEAD_PER_MEMBER"] = perhead
    # The SHIPPED ladder applies, untouched: n=8 pays a 16-window graph here
    # exactly as it would on the board. Every variant pays the same padding, so
    # the comparison is unaffected and the absolute numbers stay honest.
    tmp = tempfile.mkdtemp()
    t_imp = time.perf_counter()
    sub = _t6_load(zip_path, tmp)
    t_imp = time.perf_counter() - t_imp
    xs = _t6_inputs(_T6_SIZES)
    res = {"mode": mode, "perhead": perhead, "import_s": t_imp,
           "members": 0 if sub._MEMBERS is None else len(sub._MEMBERS),
           "ladder": list(sub._GRAPH_LADDER), "sizes": {}}
    # FIRST call in this process at n=5 -- the honest twin of a grader call.
    t0 = time.perf_counter()
    o5 = sub.predict(xs[5], {})
    torch.cuda.synchronize()
    res["first_call_ms_per_window"] = (time.perf_counter() - t0) * 1000.0 / 5
    res["graph_live"] = bool(sub._GRAPH_OK[0])
    np.savez(os.path.join(outdir, f"{mode}__{perhead or 'none'}.npz"),
             pred=o5["prediction"], width=o5["upper"] - o5["lower"])
    for n, a in xs.items():
        sub.predict(a, {}); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(_T6_REPS):
            sub.predict(a, {})
        torch.cuda.synchronize()
        res["sizes"][str(n)] = (time.perf_counter() - t0) * 1000.0 / _T6_REPS / n
    print("T6JSON " + json.dumps(res), flush=True)


def _t6_sweep(zip_path, modes):
    import json, subprocess
    outdir = tempfile.mkdtemp()
    rows = []
    for spec in modes:
        mode, _, perhead = spec.partition(":")
        env = dict(os.environ)
        cmd = [sys.executable, os.path.abspath(__file__), zip_path,
               "--t6-single", mode, perhead, outdir]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("T6JSON ")]
        if not line:
            print(f"  {spec:<16} FAILED\n{p.stdout[-800:]}\n{p.stderr[-1500:]}")
            continue
        r = json.loads(line[0][7:])
        r["spec"] = spec
        rows.append(r)
        err = [l for l in p.stderr.splitlines() if "[submission]" in l]
        for l in err[:3]:
            print("    " + l)
    if not rows:
        return
    ref = rows[0]
    rdat = np.load(os.path.join(outdir, f"{ref['mode']}__{ref['perhead'] or 'none'}.npz"))
    print(f"\n=== T6/E4 sweep: {os.path.basename(zip_path)} "
          f"k={ref['members']} ladder={ref['ladder']} "
          f"(module = WORKING TREE submission_flower.py, weights from the zip; "
          f"reference = {ref['spec']}, {_T6_REPS} reps, sizes {_T6_SIZES}) ===")
    head = "  ".join(f"n={n:<2}" for n in _T6_SIZES)
    print(f"{'variant':<16}{'first@5':>9} {head}   {'vs ref @5':>10}  {'max|d pred|':>11} {'max|d width|':>12}")
    for r in rows:
        d = np.load(os.path.join(outdir, f"{r['mode']}__{r['perhead'] or 'none'}.npz"))
        dp = float(np.abs(d["pred"] - rdat["pred"]).max())
        dw = float(np.abs(d["width"] - rdat["width"]).max())
        cells = "  ".join(f"{r['sizes'][str(n)]:5.3f}" for n in _T6_SIZES)
        rel = r["sizes"]["5"] / ref["sizes"]["5"]
        print(f"{r['spec']:<16}{r['first_call_ms_per_window']:9.3f} {cells}   "
              f"{rel:9.3f}x  {dp:11.2e} {dw:12.2e}"
              + ("" if max(dp, dw) <= 1e-5 else "   NOT IDENTICAL"))
    print("  (ms per window; 'first@5' is the first predict in a fresh process, "
          "the grader's own pattern. fold1 is a CEILING PROBE and is expected to "
          "differ -- it runs member 0's weights M times.)")
    # What the n=5 steady-state delta is worth on the board, holding the board's
    # own cost model: only the MEMBER term can move, so the saving is capped by it.
    print("  board value of the n=5 delta (board fit t = 0.2417*k + 6.0962 ms):")
    for r in rows[1:]:
        dms = (r["sizes"]["5"] - ref["sizes"]["5"])
        print(f"    {r['spec']:<16} {dms:+7.3f} ms/window locally")


if "--t6-single" in sys.argv:
    i = sys.argv.index("--t6-single")
    _t6_single(sys.argv[1], sys.argv[i + 1], sys.argv[i + 2], sys.argv[i + 3])
    raise SystemExit(0)
if "--t6" in sys.argv:
    i = sys.argv.index("--t6")
    _t6_sweep(sys.argv[1], tuple(sys.argv[i + 1:]) or _T6_DEFAULT_MODES)
    raise SystemExit(0)

S8 = dict(rel_l2=94.678636, tke=75.181389, mvpe=92.891731, time=91.752566, sps=36.824846)
BASE = estimate(**S8)
t_ref = (100.0 / S8['time'] - 1.0) ** 2 * 0.72896     # grader seconds/window for submission 8

zip_path = sys.argv[1]
tmp = tempfile.mkdtemp()
zipfile.ZipFile(zip_path).extractall(tmp)
sys.path.insert(0, tmp)
spec = importlib.util.spec_from_file_location("submission", os.path.join(tmp, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
print(f"device cuda={torch.cuda.is_available()}  zip={zip_path}")

rng = np.random.default_rng(0)
x1 = rng.normal(0, 0.1, (1, 20, 32, 64, 3)).astype(np.float32); x1[..., 2] = 0
x16 = rng.normal(0, 0.1, (16, 20, 32, 64, 3)).astype(np.float32); x16[..., 2] = 0
# intermediate call sizes (audit 2026-09-17): a 2..15-window call pays a padded 16-window graph
# 5 IS THE GRADER'S ACTUAL CALL SIZE (submission 13's Codabench log: "call n=5",
# seventeen times). Every bench before 2026-09-21 measured 1/2/4/8/16 and therefore
# never measured the only size that is scored. RPDE_BENCH_SIZES overrides the list.
_mid = tuple(int(s) for s in os.environ.get("RPDE_BENCH_SIZES", "2,4,5,8").split(","))
xs_mid = {n: rng.normal(0, 0.1, (n, 20, 32, 64, 3)).astype(np.float32) for n in _mid}
for _a in xs_mid.values(): _a[..., 2] = 0

def timeit(fn, x, reps=50):
    fn(x); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / x.shape[0]

for label, env in (("graph path", ""), ("eager", "1")):
    os.environ["RPDE_NO_GRAPH"] = env
    sub._GRAPH_OK[0] = None; sub._graphs.clear()
    o1 = sub.predict(x1, {}); o16 = sub.predict(x16, {})
    y1 = np.concatenate([o1["prediction"], o1["upper"] - o1["lower"]], axis=-1) if "upper" in o1 else o1["prediction"]
    y16 = np.concatenate([o16["prediction"], o16["upper"] - o16["lower"]], axis=-1) if "upper" in o16 else o16["prediction"]
    t1 = timeit(lambda a: sub.predict(a, {}), x1); t16 = timeit(lambda a: sub.predict(a, {}), x16)
    print(f"{label:<11} batch 1: {t1*1000:7.3f} ms/window   batch 16: {t16*1000:7.3f} ms/window   "
          f"graph live={sub._GRAPH_OK[0]}")
    mids = {n: timeit(lambda a: sub.predict(a, {}), xa, reps=20) for n, xa in xs_mid.items()}
    print("            " + "   ".join(f"batch {n}: {t*1000:6.3f} ms/window" for n, t in mids.items()))
    if env == "" and getattr(sub, "_graphs", None):
        for b, entry in sorted(sub._graphs.items()):
            outs = entry[2]
            outs = outs if isinstance(outs, tuple) else (outs,)
            print(f"            graph b={b}: outputs contiguous = "
                  + ", ".join("None" if o is None else str(bool(o.is_contiguous())) for o in outs))
    if env == "":
        yg1, yg16 = y1, y16
    else:
        d1, d16 = np.abs(yg1 - y1).max(), np.abs(yg16 - y16).max()
        # (Since 2026-09-17 the compared array also carries the bound WIDTHS, so a
        # fused head inside the graph is checked against the numpy head as well.)
        # Same weights, same kernels replayed: the only legitimate difference is
        # float32 accumulation order (cuBLAS may choose another algorithm under
        # capture), i.e. ~1e-6 relative on O(0.1-1) fields. 1e-5 absolute is
        # 500x below the narrowest half-width (0.006) and below val rel-L2's
        # resolution. Anything larger means the capture froze shape- or
        # state-dependent behaviour and the path must not ship.
        TOL = 1e-5
        verdict = "PASS" if max(d1, d16) <= TOL else "FAIL -- do not ship the graph path"
        print(f"  max |graph - eager|: batch1 {d1:.2e}  batch16 {d16:.2e}   tolerance {TOL:.0e}: {verdict}")
        for name, tg, te in (("batch 1", tg1, t1), ("batch 16", tg16, t16)):
            ratio = tg / te
            ts = score_time(t_ref * ratio)
            print(f"  {name}: graph/eager {ratio:.2f}x -> if the grader's 5.89 ms scales the same, "
                  f"time_score {ts:.2f}, d(final) {estimate(**dict(S8, time=ts)) - BASE:+.3f}")
    if env == "":
        tg1, tg16 = t1, t16
