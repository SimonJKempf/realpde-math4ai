#!/usr/bin/env python
"""Where does the grader's member-INDEPENDENT 6.1 ms per window actually go? (direction T1)

The board's own cost model, pinned by three board points (subs 13s/17/18 on the
A800), is

    t = 0.2417*k + 6.0962 ms per window,   k = ensemble members

so a member costs 0.24 ms and the member-independent term is 6.1 ms -- 89% of
ens3's total. The grader calls predict() with n=5 windows, 17 times, EACH CALL
IN A FRESH SUBPROCESS (codabench rules note, 2026-09-17). So the fixed term is
~30.5 ms paid once per process, and anything that is one-time-per-process but
sits inside predict() lands squarely in it.

Modes (each is meant to run as its OWN process -- that is the point):

  grader   import (untimed, as on the board) then ONE predict of n windows.
           The honest twin of a grader call. Run it many times => a distribution.
  steady   import then R predicts in a loop: the steady-state cost, i.e. what is
           left after every one-time cost has been paid. grader - steady = the
           per-process tax.
  stages   replicate predict()'s stages in order with host timers and CUDA
           events, first call and steady state, plus per-graph first-replay
           timings (the cudaGraphLaunch upload hypothesis) and a host
           page-fault micro-benchmark.
  kernels  torch.profiler kernel counts and self-CUDA time for the model
           forward and for the width head, separately.
  proto    a prototype fused path (no dead std, bounds formed on the device,
           pinned input staging, pre-uploaded graphs, zero-copy outputs) timed
           exactly like `grader`, to price the proposed reductions end to end
           instead of summing guesses.

    python scripts/predict_profile.py <zip> --mode grader -n 5
"""
import argparse
import importlib.util
import os
import statistics
import sys
import tempfile
import time
import zipfile

import numpy as np


def load(zp, src=None, bounds_fast=None):
    """Extract the zip and import its submission.py.

    src          replace the zip's submission.py with this file (so the tree's
                 working copy can be timed against already-built weights)
    bounds_fast  True/False -> set bounds.json's "fast_predict" key, which is how
                 build_submission.py bakes the T7 fast path in (--bounds <copy>)
    """
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(zp).extractall(tmp)
    if src is not None:
        import shutil
        shutil.copy(src, os.path.join(tmp, "submission.py"))
    if bounds_fast is not None:
        import json
        bp = os.path.join(tmp, "bounds.json")
        with open(bp) as f:
            b = json.load(f)
        b["fast_predict"] = bool(bounds_fast)
        with open(bp, "w") as f:
            json.dump(b, f)
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location("sub", os.path.join(tmp, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    spec.loader.exec_module(m)                       # runs _warm_up(): the untimed region
    return m, (time.perf_counter() - t0) * 1e3


def make_input(n, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.1, (n, 20, 32, 64, 3)).astype(dtype)
    x[..., 2] = 0                                    # p is unmeasured on real PIV
    return x


def ms(t0, t1):
    return (t1 - t0) * 1e3


def banner(sub, args):
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"# zip={os.path.basename(args.zip)} mode={args.mode} n={args.n} "
          f"members={0 if sub._MEMBERS is None else len(sub._MEMBERS)} "
          f"head={'none' if sub._HEAD is None else ('z' if sub._HEAD_USE_Z else 'drop-z')} "
          f"gpu={dev} torch={torch.__version__} "
          f"no_graph={os.environ.get('RPDE_NO_GRAPH','')!r} tf32={os.environ.get('RPDE_TF32','')!r} "
          f"fast={getattr(sub, '_FAST', 'absent')}")


# ---------------------------------------------------------------------------
def mode_grader(sub, args):
    """Exactly what the grader does: one call, fresh process, import untimed."""
    x = make_input(args.n, np.float64 if args.f64 else np.float32)
    t0 = time.perf_counter()
    out = sub.predict(x, {})
    t1 = time.perf_counter()
    assert out["prediction"].shape == (args.n, 20, 32, 64, 3)
    # checksums so two processes (e.g. TF32 on vs off) can be compared for numerics
    print(f"GRADER total_ms={ms(t0,t1):8.3f} per_window_ms={ms(t0,t1)/args.n:7.3f} "
          f"pred_sum={out['prediction'].astype(np.float64).sum():.9e} "
          f"half_sum={(out['upper']-out['lower']).astype(np.float64).sum():.9e}")


def mode_steady(sub, args):
    x = make_input(args.n)
    first = []
    for i in range(args.reps):
        t0 = time.perf_counter()
        sub.predict(x, {})
        t1 = time.perf_counter()
        first.append(ms(t0, t1))
    print(f"STEADY call1={first[0]:8.3f} call2={first[1]:8.3f} call3={first[2]:8.3f} "
          f"median_of_rest={statistics.median(first[3:]):8.3f} "
          f"per_window_median={statistics.median(first[3:])/args.n:7.3f}")
    print("STEADY_ALL " + " ".join(f"{v:.3f}" for v in first))


# ---------------------------------------------------------------------------
def mode_stages(sub, args):
    import torch
    n = args.n
    dev = sub._state["device"]
    cfg = sub._state["cfg"]
    stacked = sub._state["stacked"]
    x = make_input(n)

    # ---- 0. the graphs the module captured at import
    print(f"STAGES graphs_captured={sorted(sub._graphs)} plan={sub._plan(n, sub._BATCH)}")

    plan = sub._plan(n, sub._BATCH)
    assert len(plan) == 1, "this mode assumes one chunk"
    start, size, b = plan[0]
    g, static_x, (s_mean, s_std, s_half, s_ok), pinned = sub._graphs[b]
    assert pinned is not None, "pinned mirrors unavailable; stage timings would not match the board"
    print(f"STAGES chunk size={size} graph_b={b} pinned={'yes' if pinned else 'no'} "
          f"out_shapes mean={tuple(s_mean.shape)} "
          f"std={None if s_std is None else tuple(s_std.shape)} "
          f"half={None if s_half is None else tuple(s_half.shape)}")

    # ---- 1. FIRST replay of each captured graph in this process, timed one by one.
    #         cudaGraphLaunch uploads the instantiated graph on its first launch;
    #         if that is expensive it is a pure per-process tax the grader pays 17x.
    torch.cuda.synchronize()
    reps = []
    for i in range(8):
        t0 = time.perf_counter()
        g.replay()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        reps.append((ms(t0, t1), ms(t1, t2)))
    print("STAGES replay_launch_then_wait_ms " +
          " ".join(f"{a:.3f}/{w:.3f}" for a, w in reps))

    # a *different* graph size, never replayed yet, to confirm it is per-graph
    other = next((bb for bb in sorted(sub._graphs) if bb != b), None)
    if other is not None:
        g2 = sub._graphs[other][0]
        torch.cuda.synchronize()
        t0 = time.perf_counter(); g2.replay(); t1 = time.perf_counter()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        t3 = time.perf_counter(); g2.replay(); t4 = time.perf_counter()
        torch.cuda.synchronize(); t5 = time.perf_counter()
        print(f"STAGES other_graph b={other} first_launch={ms(t0,t1):.3f} first_wait={ms(t1,t2):.3f} "
              f"second_launch={ms(t3,t4):.3f} second_wait={ms(t4,t5):.3f}")

    # ---- 2. pure device time of the fused graph
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    ev0.record(); [g.replay() for _ in range(20)]; ev1.record()
    torch.cuda.synchronize()
    print(f"STAGES graph_device_ms_per_call={ev0.elapsed_time(ev1)/20:.3f} "
          f"per_window={ev0.elapsed_time(ev1)/20/size:.3f}")

    # ---- 3. per-stage host timing, first iteration and steady state
    im = torch.inference_mode(); im.__enter__()      # predict() runs the device part under it

    def run_stages(xx, fresh_out=True, store=None):
        t = {}
        t0 = time.perf_counter()
        arr = np.asarray(xx, dtype=np.float32)
        t1 = time.perf_counter(); t["asarray"] = ms(t0, t1)
        xin = np.ascontiguousarray(arr[..., :cfg["c_in"]])
        t2 = time.perf_counter(); t["contig"] = ms(t1, t2)
        if size < b:
            static_x[size:].zero_()
        static_x[:size].copy_(torch.from_numpy(xin[start:start + size]))
        t3 = time.perf_counter(); t["h2d"] = ms(t2, t3)
        g.replay()
        t4 = time.perf_counter(); t["replay_launch"] = ms(t3, t4)
        for dev_t, host_t in zip((s_mean, s_std, s_half, s_ok), pinned):
            if dev_t is not None:
                host_t.copy_(dev_t, non_blocking=True)
        t5 = time.perf_counter(); t["d2h_launch"] = ms(t4, t5)
        torch.cuda.current_stream().synchronize()
        t6 = time.perf_counter(); t["sync"] = ms(t5, t6)
        h_mean, h_std, h_half, h_ok = pinned
        if fresh_out or store is None:
            mean = np.empty((n, cfg["t_out"], *cfg["resolution"], cfg["c_out"]), np.float32)
            std = np.empty_like(mean) if s_std is not None else None
            half_g = np.empty((n, cfg["t_out"], *cfg["resolution"], 2), np.float32)
        else:
            mean, std, half_g = store
        t7 = time.perf_counter(); t["alloc_out"] = ms(t6, t7)
        mean[start:start + size] = h_mean.numpy()[:size]
        t8 = time.perf_counter(); t["copy_mean"] = ms(t7, t8)
        if std is not None:
            std[start:start + size] = h_std.numpy()[:size]
        t9 = time.perf_counter(); t["copy_std"] = ms(t8, t9)
        half_g[start:start + size] = h_half.numpy()[:size]
        t10 = time.perf_counter(); t["copy_half"] = ms(t9, t10)
        ok = bool(h_ok.item())
        t11 = time.perf_counter(); t["ok_item"] = ms(t10, t11)
        half = np.empty(mean.shape, np.float32)
        half[..., :2] = half_g
        half[..., 2:] = sub._HALF_WIDTH[2:]
        t12 = time.perf_counter(); t["half_assemble"] = ms(t11, t12)
        lower = mean - half
        t13 = time.perf_counter(); t["lower"] = ms(t12, t13)
        upper = mean + half
        t14 = time.perf_counter(); t["upper"] = ms(t13, t14)
        d = {"prediction": mean, "lower": lower, "upper": upper}
        t15 = time.perf_counter(); t["dict"] = ms(t14, t15)
        t["TOTAL"] = ms(t0, t15)
        return t, (mean, std, half_g), d

    keys = ["asarray", "contig", "h2d", "replay_launch", "d2h_launch", "sync", "alloc_out",
            "copy_mean", "copy_std", "copy_half", "ok_item", "half_assemble", "lower",
            "upper", "dict", "TOTAL"]
    first, store, _ = run_stages(x)
    print("STAGES first_call " + " ".join(f"{k}={first[k]:.3f}" for k in keys))
    runs = [run_stages(x)[0] for _ in range(args.reps)]
    med = {k: statistics.median(r[k] for r in runs) for k in keys}
    print("STAGES steady_fresh_alloc " + " ".join(f"{k}={med[k]:.3f}" for k in keys))
    runs2 = [run_stages(x, fresh_out=False, store=store)[0] for _ in range(args.reps)]
    med2 = {k: statistics.median(r[k] for r in runs2) for k in keys}
    print("STAGES steady_reuse_alloc " + " ".join(f"{k}={med2[k]:.3f}" for k in keys))
    im.__exit__(None, None, None)

    # ---- 4. host page-fault micro-benchmark: fresh mmap'd numpy vs reused buffer
    nb = n * 20 * 32 * 64 * 3
    src = np.zeros(nb, np.float32)
    reuse = np.empty(nb, np.float32); reuse[:] = src
    t0 = time.perf_counter()
    for _ in range(args.reps):
        a = np.empty(nb, np.float32); a[:] = src
    t1 = time.perf_counter()
    for _ in range(args.reps):
        reuse[:] = src
    t2 = time.perf_counter()
    print(f"STAGES hostbuf fresh_alloc_fill_ms={ms(t0,t1)/args.reps:.3f} "
          f"reuse_fill_ms={ms(t1,t2)/args.reps:.3f} bytes={nb*4}")

    # ---- 5. device-side split: model forward vs width head vs std
    with torch.inference_mode():
        def ev(fn, r=20):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(r):
                fn()
            e1.record(); torch.cuda.synchronize()
            return e0.elapsed_time(e1) / r

        def wall(fn, r=20):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(r):
                fn()
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            return ms(t0, t1) / r

        y = stacked(static_x)
        mean_t = (y[0] if y.shape[0] == 1 else y.mean(0)).contiguous()
        print(f"STAGES eager_device_ms stacked={ev(lambda: stacked(static_x)):.3f} "
              f"mean={ev(lambda: (stacked(static_x)[0] if y.shape[0]==1 else stacked(static_x).mean(0))):.3f} "
              f"head={ev(lambda: sub._head_half_torch(mean_t, None, static_x, dev)):.3f} "
              f"std={ev(lambda: y.std(0, unbiased=False)) if y.shape[0] > 1 else float('nan'):.3f} "
              f"fused={ev(lambda: sub._fused(stacked, static_x, dev)):.3f}")
        print(f"STAGES eager_wall_ms stacked={wall(lambda: stacked(static_x)):.3f} "
              f"head={wall(lambda: sub._head_half_torch(mean_t, None, static_x, dev)):.3f} "
              f"fused={wall(lambda: sub._fused(stacked, static_x, dev)):.3f}")

        # transfers, isolated (a full-graph-shape copy, i.e. b rows, not n)
        xt = torch.from_numpy(np.ascontiguousarray(make_input(b)))
        print(f"STAGES h2d_pageable_ms={wall(lambda: static_x.copy_(xt)):.3f} "
              f"h2d_bytes={static_x.numel()*4}")
        px = torch.empty_like(xt, pin_memory=True); px.copy_(xt)
        def h2d_pinned():
            static_x.copy_(px, non_blocking=True); torch.cuda.current_stream().synchronize()
        print(f"STAGES h2d_pinned_ms={wall(h2d_pinned):.3f} "
              f"host_memcpy_into_pinned_ms={wall(lambda: px.copy_(xt)):.3f}")
        hm = pinned[0]
        def d2h_pinned():
            hm.copy_(s_mean, non_blocking=True); torch.cuda.current_stream().synchronize()
        print(f"STAGES d2h_pinned_mean_ms={wall(d2h_pinned):.3f} "
              f"d2h_pageable_mean_ms={wall(lambda: s_mean.cpu()):.3f} "
              f"bytes={s_mean.numel()*4}")


# ---------------------------------------------------------------------------
def mode_kernels(sub, args):
    import torch
    from torch.profiler import profile, ProfilerActivity
    dev = sub._state["device"]
    cfg = sub._state["cfg"]
    stacked = sub._state["stacked"]
    n = args.n
    static_x = torch.zeros((n, cfg["t_in"], *cfg["resolution"], cfg["c_in"]), device=dev)

    def report(tag, fn):
        with torch.inference_mode():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
        def dev_us(e):
            for a in ("self_device_time_total", "self_cuda_time_total"):
                if hasattr(e, a):
                    return getattr(e, a)
            return 0.0
        evs = [e for e in p.key_averages() if dev_us(e) > 0]
        tot = sum(dev_us(e) for e in evs) / 10 / 1000.0
        cnt = sum(e.count for e in evs) / 10
        print(f"KERNELS {tag} n_kernel_launches={cnt:.0f} total_self_cuda_ms={tot:.3f}")
        top = sorted(evs, key=lambda e: -dev_us(e))[:12]
        for e in top:
            print(f"KERNELS   {tag} {e.key[:58]:58s} n={e.count/10:6.1f} "
                  f"cuda_ms={dev_us(e)/10/1000:7.4f}")

    with torch.inference_mode():
        y = stacked(static_x)
        mean_t = (y[0] if y.shape[0] == 1 else y.mean(0)).contiguous()
    report("model", lambda: stacked(static_x))
    if sub._HEAD:
        report("head", lambda: sub._head_half_torch(mean_t, None, static_x, dev))
    report("fused", lambda: sub._fused(stacked, static_x, dev))


# ---------------------------------------------------------------------------
def mode_proto(sub, args):
    """Price the proposed reductions ONE AT A TIME, each as the FIRST call in a
    fresh process -- the only honest twin of a grader call.

      v0  faithful replica of the shipped path at the shipped graph size (b=8
          for n=5 pre-ladder), rebuilt here so v0..v4 differ in nothing else
      v1  v0 with the exact graph size (no padding)
      v2  v1 without the member std, which --drop-z never reads
      v3  v2 with lower/upper formed on the device (no numpy half/sub/add)
      v4  v3 with a pinned input staging buffer and the graph pre-uploaded
      v5  v4 returning views of the pinned output buffers (no host copy at all)
    """
    import torch
    v = args.variant
    n = args.n
    dev = sub._state["device"]
    cfg = sub._state["cfg"]
    stacked = sub._state["stacked"]
    half_const = torch.as_tensor(sub._HALF_WIDTH, device=dev)
    b = sub._bucket(n, sub._BATCH) if v == 0 else n

    def fused(sx):
        y = stacked(sx)
        mean = y[0] if y.shape[0] == 1 else y.mean(0)
        std = y.std(0, unbiased=False) if (v <= 1 and y.shape[0] > 1) else None
        hh = sub._head_half_torch(mean, None, sx, dev)             # (N,T,H,W,2)
        if v <= 2:
            out = [mean.contiguous(), None if std is None else std.contiguous(),
                   hh.contiguous()]
        else:
            half = torch.empty_like(mean)
            half[..., :2] = hh
            half[..., 2:] = half_const[2:]
            out = [mean.contiguous(), (mean - half).contiguous(), (mean + half).contiguous()]
        ok = torch.isfinite(out[0]).all()
        return tuple(out) + (ok,)

    static_x = torch.zeros((b, cfg["t_in"], *cfg["resolution"], cfg["c_in"]), device=dev)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(s):
        for _ in range(3):
            fused(static_x)
    torch.cuda.current_stream().wait_stream(s)
    gg = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(gg):
        outs = fused(static_x)
    host = tuple(None if o is None else torch.empty(o.shape, dtype=o.dtype, pin_memory=True)
                 for o in outs)
    pin_x = None
    if v >= 4:
        for h in host:
            if h is not None:
                h.zero_()                             # pre-touch the pinned pages
        pin_x = torch.empty(static_x.shape, dtype=static_x.dtype, pin_memory=True)
        pin_x.zero_()
        gg.replay()                                   # PRE-UPLOAD the graph (untimed)
        torch.cuda.synchronize()

    x = make_input(n)
    t0 = time.perf_counter()
    with torch.inference_mode():
        xin = np.ascontiguousarray(np.asarray(x, dtype=np.float32)[..., :cfg["c_in"]])
        if v >= 4:
            pin_x.copy_(torch.from_numpy(xin))
            static_x.copy_(pin_x, non_blocking=True)
        else:
            if n < b:
                static_x[n:].zero_()
            static_x[:n].copy_(torch.from_numpy(xin))
        gg.replay()
        for d_t, h_t in zip(outs, host):
            if d_t is not None:
                h_t.copy_(d_t, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        if v >= 5:
            res = {"prediction": host[0].numpy(), "lower": host[1].numpy(),
                   "upper": host[2].numpy()}
        elif v >= 3:
            res = {"prediction": host[0].numpy()[:n].copy(),
                   "lower": host[1].numpy()[:n].copy(),
                   "upper": host[2].numpy()[:n].copy()}
        else:
            mean = np.empty((n, *host[0].shape[1:]), np.float32)
            mean[:] = host[0].numpy()[:n]
            if host[1] is not None:
                std = np.empty_like(mean); std[:] = host[1].numpy()[:n]
            half_g = np.empty((n, *host[2].shape[1:]), np.float32)
            half_g[:] = host[2].numpy()[:n]
            half = np.empty(mean.shape, np.float32)
            half[..., :2] = half_g
            half[..., 2:] = sub._HALF_WIDTH[2:]
            res = {"prediction": mean, "lower": mean - half, "upper": mean + half}
        ok = bool(host[-1].item())
    t1 = time.perf_counter()
    assert ok and res["prediction"].shape == (n, 20, 32, 64, 3)
    print(f"PROTO v{v} graph_b={b} total_ms={ms(t0,t1):8.3f} per_window_ms={ms(t0,t1)/n:7.3f}")


# ---------------------------------------------------------------------------
# T7: identity and aliasing checks for the fast predict path.
def _call(sub, x):
    t0 = time.perf_counter()
    out = sub.predict(x, {})
    return out, ms(t0, time.perf_counter())


def mode_dump(sub, args):
    """One call in a fresh process; write prediction/lower/upper for a later cmp."""
    x = make_input(args.n, seed=args.seed)
    out, dt = _call(sub, x)
    np.savez(args.out, prediction=out["prediction"], lower=out["lower"], upper=out["upper"])
    print(f"DUMP -> {args.out} total_ms={dt:8.3f} per_window_ms={dt/args.n:7.3f} "
          f"shapes={out['prediction'].shape}")


def mode_cmp(sub, args):
    """One call in a fresh process; compare element for element against a dump.
    This is the identity proof: it reports max |delta| and the exact-bit count."""
    x = make_input(args.n, seed=args.seed)
    out, dt = _call(sub, x)
    ref = np.load(args.ref)
    worst = 0.0
    for k in ("prediction", "lower", "upper"):
        a, b = ref[k], out[k]
        assert a.shape == b.shape == (args.n, 20, 32, 64, 3), (k, a.shape, b.shape)
        assert b.dtype == np.float32, (k, b.dtype)
        d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        bits = int((a.view(np.uint32) == b.view(np.uint32)).sum())
        worst = max(worst, d)
        print(f"CMP {k:10s} max_abs_delta={d:.3e} bit_identical={bits}/{a.size} "
              f"({100.0*bits/a.size:.4f}%)")
    assert np.all(out["lower"] <= out["upper"]), "lower > upper"
    assert np.all(np.isfinite(out["lower"])) and np.all(np.isfinite(out["upper"]))
    print(f"CMP WORST max_abs_delta={worst:.3e} total_ms={dt:8.3f} "
          f"per_window_ms={dt/args.n:7.3f} verdict={'IDENTICAL' if worst == 0.0 else 'DIFFERS'}")


def mode_twocall(sub, args):
    """The grader promises one call per process. Prove we are correct anyway:
    r1 = predict(A), r2 = predict(B), r3 = predict(A) in ONE process. r1 must be
    untouched by the later calls (no aliasing of a reused buffer) and r3 == r1."""
    xa = make_input(args.n, seed=args.seed)
    xb = make_input(args.n, seed=args.seed + 1)
    r1, t1 = _call(sub, xa)                      # keep r1's OBJECTS, not copies
    snap = {k: v.copy() for k, v in r1.items()}  # what call 1 returned, frozen
    r2, t2 = _call(sub, xb)
    r3, t3 = _call(sub, xa)
    for k in ("prediction", "lower", "upper"):
        same = np.array_equal(r1[k], snap[k])
        print(f"TWOCALL {k:10s} call1_unchanged_after_2_more_calls={same} "
              f"r3_equals_r1={np.array_equal(r3[k], snap[k])} "
              f"r2_differs_from_r1={not np.array_equal(r2[k], snap[k])}")
        assert same, f"ALIASING BUG: call 1's {k} was overwritten by a later call"
        assert np.array_equal(r3[k], snap[k]), f"call 3 on the same input differs in {k}"
        assert not np.array_equal(r2[k], snap[k]), f"call 2 on a different input matched in {k}"
        assert np.all(r2["lower"] <= r2["upper"])
    print(f"TWOCALL ms={t1:.3f} {t2:.3f} {t3:.3f}  OK")


MODES = {"grader": mode_grader, "steady": mode_steady, "stages": mode_stages,
         "kernels": mode_kernels, "proto": mode_proto, "dump": mode_dump,
         "cmp": mode_cmp, "twocall": mode_twocall}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--mode", default="grader", choices=sorted(MODES))
    ap.add_argument("-n", type=int, default=5, help="windows per call (the grader uses 5)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--f64", action="store_true", help="hand predict() a float64 array")
    ap.add_argument("--variant", type=int, default=5, help="proto mode: which reduction to price")
    ap.add_argument("--src", default=None,
                    help="replace the zip's submission.py with this file before importing")
    ap.add_argument("--bounds-fast", dest="bounds_fast", default=None,
                    choices=("0", "1"), help='set bounds.json "fast_predict" (the build-time switch)')
    ap.add_argument("--seed", type=int, default=0, help="input seed for dump/cmp/twocall")
    ap.add_argument("--out", default=None, help="dump mode: .npz to write")
    ap.add_argument("--ref", default=None, help="cmp mode: .npz to compare against")
    args = ap.parse_args()
    sub, import_ms = load(args.zip, args.src,
                          None if args.bounds_fast is None else args.bounds_fast == "1")
    banner(sub, args)
    print(f"# import_ms={import_ms:.1f} (untimed on the board)")
    MODES[args.mode](sub, args)
