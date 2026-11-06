#!/usr/bin/env python
"""Post-hoc variance inflation: does re-inflating the blurred fluctuation field pay?

    python scripts/variance_inflation.py <zip> [--scales 1.0 1.05 1.1 1.2 1.3] [--stride 50]
    python scripts/variance_inflation.py <zip> --scales 1.0 --mean-anchor 0.1 0.25 0.5 [--mean-anchor-source input|persist]
    python scripts/variance_inflation.py <zip> --scales 1.0 --rollout-blend 5 0.5 5 1.0 10 0.5   # S W pairs, 2nd predict()
    python scripts/variance_inflation.py <zip> --scales 1.0 --rollout-ramp 10 1.0 5 0.5        # S W19 pairs, lead ramp
    python scripts/variance_inflation.py <zip> --scales 1.0 --rollout-ramp 10 1.0 --rollout-zip <zip_b>   # 2nd pass from zip_b

Our diagnostics measured the shipped prediction carrying only **65% of the true unsteady energy**,
falling to 46% at the longest lead. That is the conditional-mean blur an L2 objective produces, and it is
the mechanism behind tke_score sitting at 78 while rel_l2 sits at 95.
The expensive fix is a generative or likelihood-trained model. This is the cheap probe of the same target:
take the prediction as it is and scale its fluctuation about its own per-pixel time mean by a constant,

    p' = mean_t(p) + alpha * (p - mean_t(p))

which adds unsteady energy without changing the mean flow, needs no training, and would be a handful of
lines in predict() if it won.
WHY IT MIGHT NOT WIN, and the reason to measure rather than assume: inflating fluctuations that are in the
WRONG PHASE increases the L2 error. BLUR found the correlation falling to 0.51 at lead 19, so a good part of
the missing energy is missing because the phase is unknown, and adding energy there is adding noise.
rel_l2 is worth 0.703 board points per subscore point against tke's 0.167, so the trade is steeply against
inflation: it must buy roughly four tke points per rel_l2 point lost merely to break even.
The theoretical optimum for a pure-noise component is alpha = 1 (the conditional mean IS the L2 optimum);
anything above 1 wins only if the scorer's tke term values energy more than rel_l2 penalises the error --
which is exactly what the composite makes measurable.

Reports every subscore and est_final per held-out row and pooled, for each alpha.
"""
import argparse, pathlib, subprocess, sys, tempfile, zipfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kit"))

REF_TIME = 91.370559


def mean_anchor(pred, x_in, alpha, source="input"):
    """Post-hoc MEAN ANCHOR (diagnostic, 2026-09-24): shift the prediction's per-pixel time mean toward
    the input window's, on channels [u, v] only, bounds untouched by the caller:

        p' = p + alpha * (m_src - mean_t(p))

    m_src is the INPUT window's per-pixel mean over its MEASURED frames only (source="input") -- an input
    element that is exactly 0.0 is an unmeasured PIV hole, the same convention as submission_flower's
    _head_extra / anchor_diagnostic.input_stats -- or the LAST input frame (source="persist"). A pixel with
    no measured frame (no data at all, or a hole in the last frame for "persist") gets NO shift.
    The fluctuation p - mean_t(p) is carried over unchanged; only the level moves. Channel 2 is untouched.
    alpha == 0 early-returns a copy so the identity is BITWISE (p + 0.0 is not: -0.0 + 0.0 = +0.0).

    pred (N, T, H, W, C>=2), x_in (N, T_in, H, W, C>=2), both numpy. Returns (p', shift) with
    shift (N, 1, H, W, 2) the applied per-pixel offset, float32."""
    p = np.asarray(pred, dtype=np.float32)
    x = np.asarray(x_in, dtype=np.float32)
    if source == "persist":
        x = x[:, -1:]
    elif source != "input":
        raise ValueError(f"mean_anchor source must be 'input' or 'persist', got {source!r}")
    meas = (x[..., 0] != 0).astype(np.float32)[..., None]                    # (N, T_in, H, W, 1)
    cnt = meas.sum(axis=1, keepdims=True)                                    # (N, 1, H, W, 1)
    m_src = (x[..., :2] * meas).sum(axis=1, keepdims=True) / np.maximum(cnt, 1.0)
    out = p.copy()
    if float(alpha) == 0.0:
        return out, np.zeros(m_src.shape, dtype=np.float32)
    m_t = p[..., :2].mean(axis=1, keepdims=True)                             # (N, 1, H, W, 2)
    shift = np.where(cnt > 0, np.float32(alpha) * (m_src - m_t), np.float32(0.0)).astype(np.float32)
    out[..., :2] = p[..., :2] + shift
    return out, shift


def _check_shift(shift, t):
    """An integer shift in [1, t-1]: at least one fed-back frame and at least one overlap frame."""
    if int(shift) != shift or not 1 <= int(shift) <= t - 1:
        raise ValueError(f"rollout shift must be an integer in [1, {t - 1}], got {shift!r}")
    return int(shift)


def rollout_input(x, p0, shift, measured=2):
    """INFERENCE-TIME ROLLOUT (RINF-inference, 2026-09-24): the second pass's input window, numpy twin of
    scripts/train.py rollout_input restricted to the first `shift` predicted frames.

        x2 = concat(x[:, shift:], fb) along time,   fb = p0[:, :shift] made to look like a real input frame

    so x2 is a T_in-frame window that ends `shift` frames into the forecast. The fed-back frames follow
    train.py's convention channel by channel:
      * channels [0, measured) (u, v) come from p0, but every element that is a PIV HOLE in x's LAST input
        frame (u == 0 exactly) is set to +0.0 -- holes are a property of the sensor, not of time, and the
        model has never seen a hole-free input;
      * channels [measured, C) are CARRIED OVER from x's last input frame, broadcast over the fed-back
        frames. On real PIV that plane is the identically-zero pressure the grader hands over, so the
        fed-back frames get zeros exactly where x has zeros; the model's own channel-2 output (never
        trained on real data) is never used. A --mask-channel zip overwrites channel 2 with (u != 0)
        inside predict() anyway, per frame, which the hole-zeroing above keeps consistent.
    x (N, T_in, H, W, C), p0 (N, T_out, H, W, C'>=measured), numpy. Returns float32 (N, T_in, H, W, C), a
    fresh array (never a view of x or p0)."""
    x = np.asarray(x, dtype=np.float32)
    p = np.asarray(p0, dtype=np.float32)
    if x.ndim != 5 or p.ndim != 5:
        raise ValueError(f"rollout_input: x and p0 must be 5-D (N,T,H,W,C), got {x.shape} and {p.shape}")
    if x.shape[0] != p.shape[0] or x.shape[2:4] != p.shape[2:4]:
        raise ValueError(f"rollout_input: x {x.shape} and p0 {p.shape} disagree on N or (H, W)")
    if x.shape[-1] < measured or p.shape[-1] < measured:
        raise ValueError(f"rollout_input: measured={measured} exceeds x {x.shape[-1]} or p0 {p.shape[-1]} channels")
    s = _check_shift(shift, min(x.shape[1], p.shape[1]))
    last = x[:, -1:]                                                         # (N, 1, H, W, C)
    meas = last[..., :1] != 0                                                # (N, 1, H, W, 1)
    fb = np.empty((x.shape[0], s, *x.shape[2:]), np.float32)
    fb[..., :measured] = np.where(meas, p[:, :s, ..., :measured], np.float32(0.0))
    if x.shape[-1] > measured:
        fb[..., measured:] = last[..., measured:]                            # broadcast over the s frames
    return np.concatenate((x[:, s:], fb), axis=1)


def ramp_weights(shift, w19, t=20):
    """LEAD-DEPENDENT rollout blend weight (--rollout-ramp, 2026-09-24) for the overlap frames j in [shift, t):

        w(j) = w19 * (j - shift) / (t - 1 - shift)          0 at lead `shift`, w19 at the last lead t-1

    The second pass is WORSE than the first at the early overlap leads and BETTER at the last ones, so a
    constant w spends the late gain on the early cost; the ramp puts the weight where the 2nd pass wins.
    shift must leave at least TWO overlap frames (shift in [1, t-2]): at shift = t-1 the ramp has a single
    frame that is both its 0 end and its w19 end, so it is refused. Returns float64 (t - shift,)."""
    s = _check_shift(shift, t)
    if s > t - 2:
        raise ValueError(f"rollout ramp shift must be in [1, {t - 2}] (a ramp needs two overlap frames), got {shift!r}")
    return float(w19) * (np.arange(s, t, dtype=np.float64) - s) / (t - 1 - s)


def rollout_blend(p0, p2, shift, w):
    """Blend the second pass into the first over their overlap, bounds untouched by the caller:

        p'[:, :shift] = p0[:, :shift]                                         (no second-pass estimate)
        p'[:, shift:] = (1 - w) * p0[:, shift:] + w * p2[:, :T - shift]

    p2 = predict(rollout_input(x, p0, shift)), so p2 frame j forecasts true frame 20 + shift + j and
    p2[:, :T-shift] covers the same true frames as p0[:, shift:]. All channels are blended (the scorer reads
    only the measured ones). w == 0 returns a copy (bitwise identity); w == 1 copies p2's frames verbatim
    (bitwise; 0*p0 + p2 would turn -0.0 into +0.0 and NaN-propagate an inf in p0).
    w is a scalar (one weight for every overlap frame) or a 1-D array of length T - shift, one weight per
    overlap frame p'[:, shift + k] (ramp_weights); the array path applies the scalar rule frame by frame, so
    a frame with weight 0 / 1 is p0's / p2's frame bitwise and a constant array equals the scalar bitwise.
    p0, p2 (N, T, H, W, C) numpy, same shape. Returns float32, a fresh array."""
    a = np.asarray(p0, dtype=np.float32)
    b = np.asarray(p2, dtype=np.float32)
    if a.ndim != 5 or a.shape != b.shape:
        raise ValueError(f"rollout_blend: p0 {a.shape} and p2 {b.shape} must be the same 5-D shape")
    T = a.shape[1]
    s = _check_shift(shift, T)
    out = a.copy()
    if np.ndim(w) != 0:
        wv = np.asarray(w, dtype=np.float64)
        if wv.shape != (T - s,):
            raise ValueError(f"rollout_blend: per-frame w must be 1-D of length T - shift = {T - s}, got {wv.shape}")
        for k in range(T - s):
            wk = float(wv[k])
            if wk == 0.0:
                continue
            if wk == 1.0:
                out[:, s + k] = b[:, k]
            else:
                out[:, s + k] = np.float32(1.0 - wk) * a[:, s + k] + np.float32(wk) * b[:, k]
        return out
    w = float(w)
    if w == 0.0:
        return out
    if w == 1.0:
        out[:, s:] = b[:, :T - s]
        return out
    out[:, s:] = np.float32(1.0 - w) * a[:, s:] + np.float32(w) * b[:, :T - s]
    return out


def main() -> None:
    # Imported here, not at module level: the data loader pulls in torch, and mean_anchor() above must stay
    # importable numpy-only (tests/test_mean_anchor.py). Nothing else about the script changes.
    from pde_competition.dataset import list_runs, split_by_condition, build_eval_arrays
    from pde_competition.composite import estimate
    import scoring as ks

    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=pathlib.Path)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.05, 1.10, 1.20, 1.30])
    ap.add_argument("--ramp", type=float, nargs="+", default=None,
                    help="per-lead alpha ramp: alpha(t) = 1 + (A-1)*(t/(T-1))**P, given as A P. "
                         "The scorer's kinetic_energy is a per-pixel TIME variance, so a CONSTANT alpha "
                         "over-inflates lead 0 (where the measured energy ratio is already 1.07) to fix "
                         "lead 19 (where it is 0.457). A ramp targets the deficit where it exists.")
    ap.add_argument("--spectral", type=float, nargs="+", default=None, metavar="V",
                    help="SPATIAL-frequency inflation, given as A KCUT [A KCUT ...]: scale only the "
                         "fluctuation's spatial modes above normalised radial wavenumber KCUT (0..1) by A. "
                         "VINFb/VINFc showed a LEAD-dependent alpha loses even when made mvpe-neutral, "
                         "because by lead 19 the correlation is 0.506 and inflating late leads amplifies "
                         "content already decorrelated from truth. The deficit is not really lead-shaped, "
                         "it is SCALE-shaped: a conditional mean damps small scales first. This targets "
                         "that axis instead. Automatically mvpe-free: the fluctuation has mean_t = 0 by "
                         "construction and any per-frame linear operator maps 0 to 0.")
    ap.add_argument("--width-gated", type=float, nargs="+", default=None, metavar="A",
                    help="inflate by A where the SHIPPED INTERVAL IS NARROW and not at all where it is "
                         "wide, using alpha(pixel) = 1 + (A-1)*(1 - w/max(w)) with w the time-averaged "
                         "half-width. Same motivation as --spectral from the other side: the head's own "
                         "width is our only per-pixel confidence signal at inference, it costs nothing "
                         "extra (it is already computed and already in the zip), and it puts the energy "
                         "where the model is confident rather than where it is guessing. Gating on the "
                         "TIME-AVERAGED width keeps alpha independent of t, so mean_t is preserved exactly "
                         "and mvpe stays bit-invariant without any re-centring.")
    ap.add_argument("--mean-anchor", type=float, nargs="+", default=None, metavar="A",
                    help="DIAGNOSTIC: shift the per-pixel time mean toward the INPUT window's masked mean by "
                         "A, p' = p + A*(m_in - mean_t(p)), channels u/v only; zeros in the input are PIV "
                         "holes and are excluded from m_in, a pixel with no measured frame is not shifted. "
                         "The bounds stay EXACTLY those predict() returned for the untransformed prediction "
                         "(pinned), like every other mode here. The rms of the applied shift is printed per "
                         "channel, so a delta model that already tracks the input mean shows ~0.")
    ap.add_argument("--mean-anchor-source", choices=["input", "persist"], default="input",
                    help="what --mean-anchor anchors to: the input window's masked time mean (input, default) "
                         "or the last input frame (persist).")
    ap.add_argument("--rollout-blend", type=float, nargs="+", default=None, metavar="V",
                    help="INFERENCE-TIME ROLLOUT, given as S W [S W ...]: a SECOND predict() on the window "
                         "x2 = concat(x[:, S:], p0[:, :S]) (fed-back frames hole-masked with x's last frame, "
                         "channel 2 carried from x's last frame = zeros on PIV), then "
                         "p'[:, S:] = (1-W) p0[:, S:] + W p2[:, :20-S]; frames [0, S) untouched. Bounds stay "
                         "p0's (pinned). Also prints per-lead rel_l2 of the 2nd pass vs the 1st at leads S, "
                         "S+5, 19 -- the mechanism question. Scores PREDICTION effects only: the 2nd call's "
                         "time cost is not measured here.")
    ap.add_argument("--rollout-ramp", type=float, nargs="+", default=None, metavar="V",
                    help="like --rollout-blend but with a LEAD-DEPENDENT weight, given as S W19 [S W19 ...]: "
                         "p'[:, j] = (1-w(j)) p0[:, j] + w(j) p2[:, j-S] for j in [S, 20) with "
                         "w(j) = W19 (j-S)/(19-S), 0 at lead S and W19 at lead 19 (S=19 refused). The 2nd "
                         "pass loses to the 1st at the early overlap leads and wins at the late ones, so a "
                         "constant W spends the late gain on the early cost. Shares the per-S 2nd pass with "
                         "--rollout-blend; bounds pinned to p0's.")
    ap.add_argument("--rollout-alpha", type=float, default=1.0, metavar="A",
                    help="apply VINF's constant fluctuation scale A AFTER every --rollout-blend/--rollout-ramp "
                         "blend: p = m + A (p - m), m = the blended field's per-pixel time mean (bounds still "
                         "pinned). 1.0 (default) leaves the rollout arms exactly as before. Needed because the "
                         "2nd pass adds fluctuation energy of its own, so its gain and alpha's overlap.")
    ap.add_argument("--rollout-zip", type=pathlib.Path, default=None, metavar="PATH",
                    help="compute the 2nd pass p2 of every --rollout-blend/--rollout-ramp arm with THIS zip "
                         "(e.g. fewer members, a cheaper 2nd call) instead of the main one; p0 and the pinned "
                         "bounds still come from the main zip. Run in a SUBPROCESS (fresh interpreter, "
                         "like the grader's per-call process): both zips ship top-level flower_standalone / "
                         "geometry / einops modules and module-level torch state, which cannot be imported "
                         "twice in one process without one zip silently using the other's code.")
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--holdout-re", type=int, nargs="+", default=[13950, 26700])
    a = ap.parse_args()

    _t, val = split_by_condition(list_runs(), holdout_re=tuple(a.holdout_re))
    x, y, keys = build_eval_arrays(val, stride=a.stride, return_keys=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="vinf_"))
    with zipfile.ZipFile(a.zip_path) as z:
        z.extractall(work / "sub")
    sys.path.insert(0, str(work / "sub"))
    import submission                                                                 # noqa: E402
    # COPY EVERYTHING predict() returns, and hand it a copy of x. (1) The fast path's first call returns
    # VIEWS of pinned host buffers (submission_flower._predict_fast); today a later call copies out instead
    # (handed_out), but --rollout-blend makes a second call in this process and p0/lower/upper must not
    # depend on that guard. np.asarray(float32 array) is NOT a copy. (2) _predict_fields/_predict_fast take
    # xin = np.ascontiguousarray(x[..., :c_in]), which IS the caller's array when c_in == C, and a
    # --mask-channel zip then writes the measured mask into its channel 2 in place -- x must stay pristine
    # for rollout_input's channel-2 carry. Both copies are free for every other mode here.
    out = submission.predict(x.copy(), {})
    p0 = np.array(out["prediction"], dtype=np.float32, copy=True)
    lower = None if out.get("lower") is None else np.array(out["lower"], dtype=np.float32, copy=True)
    upper = None if out.get("upper") is None else np.array(out["upper"], dtype=np.float32, copy=True)
    del out
    c = ks.measured_channels(y)
    mean_t = p0.mean(axis=1, keepdims=True)
    rows = sorted({k[0] for k in keys}) + ["all"]

    print(f"windows {p0.shape[0]}  holdout {a.holdout_re}  measured channels {c}")
    T = p0.shape[1]
    combos = [("const", al, None) for al in a.scales]
    if a.ramp:
        A, P = a.ramp[0], a.ramp[1]
        combos.append(("ramp", A, P))
    if a.spectral:
        v = a.spectral
        assert len(v) % 2 == 0, "--spectral takes pairs: A KCUT [A KCUT ...]"
        combos += [("spectral", v[i], v[i + 1]) for i in range(0, len(v), 2)]
    if a.width_gated:
        combos += [("wgate", A, None) for A in a.width_gated]
    if a.mean_anchor:
        combos += [("manchor", A, None) for A in a.mean_anchor]
    if a.rollout_blend:
        v = a.rollout_blend
        assert len(v) % 2 == 0, "--rollout-blend takes pairs: S W [S W ...]"
        combos += [("rblend", v[i + 1], _check_shift(v[i], T)) for i in range(0, len(v), 2)]
    if a.rollout_ramp:
        v = a.rollout_ramp
        assert len(v) % 2 == 0, "--rollout-ramp takes pairs: S W19 [S W19 ...]"
        for i in range(0, len(v), 2):
            ramp_weights(v[i], v[i + 1], T)                     # refuse a bad S before any 2nd predict()
            combos.append(("rramp", v[i + 1], _check_shift(v[i], T)))
    p2_cache = {}
    # --rollout-zip: the 2nd pass runs the OTHER zip in a fresh interpreter (see its --help). x2 goes out and
    # p2 comes back as .npy files in this run's temp dir; the child inherits the environment (RPDE_* knobs),
    # its cwd and sys.path[0] are the other zip's extracted dir, so `import submission` and everything that
    # module imports beside itself resolve there and nowhere else.
    src_tag = ""
    if a.rollout_zip is not None and (a.rollout_blend or a.rollout_ramp):
        work_b = work / "sub_b"
        with zipfile.ZipFile(a.rollout_zip) as z:
            z.extractall(work_b)
        src_tag = f" (2nd pass from {a.rollout_zip.name})"
    elif a.rollout_zip is not None:
        print("--rollout-zip given without --rollout-blend/--rollout-ramp: ignored.")
    child = ("import sys\n"
             "d, xin, pout = sys.argv[1:4]\n"
             "sys.path.insert(0, d)\n"
             "import numpy as np\n"
             "import submission\n"
             "out = submission.predict(np.load(xin), {})\n"
             "np.save(pout, np.array(out['prediction'], dtype=np.float32, copy=True))\n")

    def second_pass(x2, S):
        if not src_tag:
            out2 = submission.predict(x2, {})
            p2 = np.array(out2["prediction"], dtype=np.float32, copy=True)
            del out2
            return p2
        xin, pout = work / f"x2_S{S}.npy", work / f"p2_S{S}.npy"
        np.save(xin, x2)
        sys.stdout.flush()                                      # keep the child's own prints in order
        subprocess.run([sys.executable, "-c", child, str(work_b), str(xin), str(pout)],
                       cwd=str(work_b), check=True)
        p2 = np.load(pout)
        if p2.shape != p0.shape:
            raise SystemExit(f"--rollout-zip {a.rollout_zip.name}: 2nd pass shape {p2.shape} != p0 {p0.shape}")
        return p2
    for kind, al, pw in combos:
        if kind == "const":
            p = mean_t + np.float32(al) * (p0 - mean_t)
            label = f"alpha = {al:.2f}"
        elif kind == "spectral":
            # Scale only the spatial modes above a normalised radial cutoff. rfft2 over (H, W) of the
            # fluctuation, one frame at a time. mean_t(f) = 0, and the filter is applied identically to
            # every frame, so mean_t of the result is 0 too: the per-pixel time mean is preserved EXACTLY
            # and mvpe is invariant by construction, not by a correction term.
            f = p0 - mean_t
            H, W = f.shape[2], f.shape[3]
            ky = np.fft.fftfreq(H)[:, None]
            kx = np.fft.rfftfreq(W)[None, :]
            krad = np.sqrt((ky / 0.5) ** 2 + (kx / 0.5) ** 2) / np.sqrt(2.0)
            gain = np.where(krad >= pw, np.float32(al), np.float32(1.0))[None, None, :, :, None]
            F = np.fft.rfft2(f, axes=(2, 3))
            p = mean_t + np.fft.irfft2(F * gain, s=(H, W), axes=(2, 3)).astype(np.float32)
            frac = float((krad >= pw).mean())
            label = f"spectral x{al:.2f} above k={pw:.2f}  ({frac:.0%} of the spatial modes)"
        elif kind == "wgate":
            if lower is None or upper is None:
                print("\n--width-gated needs the zip's lower/upper; this zip ships none. Skipped.")
                continue
            # The head's half-width, averaged over time so the gate carries no t dependence -> mean_t
            # is preserved exactly and mvpe is bit-invariant, the same free lunch a constant alpha gets.
            w = (np.asarray(upper) - np.asarray(lower)).astype(np.float32) / 2.0
            wbar = w.mean(axis=1, keepdims=True)
            wmax = wbar.max(axis=(2, 3), keepdims=True)
            g = 1.0 - wbar / np.maximum(wmax, 1e-12)
            a_px = (1.0 + (al - 1.0) * g).astype(np.float32)
            p = mean_t + a_px * (p0 - mean_t)
            label = (f"width-gated to {al:.2f} where the interval is narrowest "
                     f"(mean alpha {float(a_px.mean()):.3f})")
        elif kind == "manchor":
            # Bounds pinned: lower/upper below are still the predict() output for p0, untouched.
            p, shift = mean_anchor(p0, x, al, source=a.mean_anchor_source)
            rms = np.sqrt((shift.astype(np.float64) ** 2).mean(axis=(0, 1, 2, 3)))
            label = (f"mean-anchor A={al:.2f} to {a.mean_anchor_source} (bounds pinned; "
                     f"mean shift rms u/v {rms[0]:.4g}/{rms[1]:.4g})")
        elif kind in ("rblend", "rramp"):
            S = pw
            if S not in p2_cache:
                x2 = rollout_input(x, p0, S, measured=c)
                p2_cache[S] = second_pass(x2, S)
                del x2
                p2 = p2_cache[S]
                # Mechanism: at each lead j >= S the 2nd pass forecasts the same true frame 20+j from a
                # window S frames closer (S of them self-generated). p2 index j-S <-> truth y[:, j].
                leads = sorted({S, min(S + 5, T - 1), T - 1})
                print(f"\nrollout S={S}: per-lead rel_l2 (raw, lower is better), 1st pass p0 vs 2nd pass p2{src_tag}")
                print(f"  {'row':>6} " + " ".join(f"{'L' + str(j) + ' p0':>9} {'L' + str(j) + ' p2':>9}" for j in leads))
                for re in rows:
                    idx = np.array([i for i, k in enumerate(keys) if re == "all" or k[0] == re])
                    cells = []
                    for j in leads:
                        r0 = float(np.mean(ks.rel_l2_per_sample(p0[idx, j:j + 1], y[idx, j:j + 1], c)))
                        r2 = float(np.mean(ks.rel_l2_per_sample(p2[idx, j - S:j - S + 1], y[idx, j:j + 1], c)))
                        cells.append(f"{r0:9.4f} {r2:9.4f}")
                    print(f"  {str(re):>6} " + " ".join(cells))
            # Bounds pinned: lower/upper below are still the FIRST call's, for p0; the 2nd call's are dropped.
            if kind == "rblend":
                p = rollout_blend(p0, p2_cache[S], S, al)
                label = f"rollout-blend S={S} W={al:g} (2nd pass, bounds pinned){src_tag}"
            else:
                p = rollout_blend(p0, p2_cache[S], S, ramp_weights(S, al, T))
                label = f"rollout-ramp S={S} W19={al:g} (2nd pass, bounds pinned){src_tag}"
            if a.rollout_alpha != 1.0:
                m_b = p.mean(axis=1, keepdims=True)
                p = m_b + np.float32(a.rollout_alpha) * (p - m_b)
                label += f" + alpha {a.rollout_alpha:g}"
        else:
            t = np.arange(T, dtype=np.float32) / max(T - 1, 1)
            a_t = (1.0 + (al - 1.0) * t ** pw).astype(np.float32).reshape(1, T, 1, 1, 1)
            q = a_t * (p0 - mean_t)
            # RE-CENTRE. A CONSTANT alpha preserves every per-pixel time mean exactly, which is why mvpe
            # (a time-averaged probe quantity) was bit-invariant under VINF. A LEAD-DEPENDENT alpha does
            # not: mean_t(alpha(t)*f(t)) != 0 when alpha varies, so the time mean shifts by ~5e-2 and mvpe
            # pays for it -- measured, and it is why VINFb's first ramp lost to the best constant despite
            # targeting the deficit better. Subtracting the induced shift restores the invariance and
            # keeps the lead-dependent shape.
            p = mean_t + q - q.mean(axis=1, keepdims=True)
            label = f"ramp to {al:.2f} with power {pw:g}  (alpha: {a_t.ravel()[0]:.2f} at lead 0 -> {a_t.ravel()[-1]:.2f} at lead {T-1})"
        print(f"\n{label}")
        print(f"  {'row':>6} {'rel_l2':>8} {'tke':>8} {'mvpe':>8} {'sps':>8} {'est.final':>10}")
        for re in rows:
            idx = np.array([i for i, k in enumerate(keys) if re == "all" or k[0] == re])
            pp, tt = p[idx], y[idx]
            rel = float(np.mean(ks.rel_l2_per_sample(pp, tt, c)))
            tke = float(np.mean(ks.tke_rel_l2_per_sample(pp, tt, c)))
            mv = ks.mvpe_rel_l2(pp, tt)
            sps, _cov = ks.aggregate_sps(pp, tt, c,
                                         lower=None if lower is None else np.asarray(lower)[idx],
                                         upper=None if upper is None else np.asarray(upper)[idx])
            s = (ks.score_error(rel), ks.score_error(tke), ks.score_error(mv), ks.score_sps(sps))
            print(f"  {str(re):>6} {s[0]:8.3f} {s[1]:8.3f} {s[2]:8.3f} {s[3]:8.3f} "
                  f"{estimate(s[0], s[1], s[2], REF_TIME, s[3]):10.3f}")


if __name__ == "__main__":
    main()
