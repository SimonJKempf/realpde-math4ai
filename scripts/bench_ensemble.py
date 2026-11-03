"""Is an ensemble nearly free on a GPU this model cannot fill?

time_score charges for WALL-CLOCK PER WINDOW at batch 1, which is how the
grader calls predict(). So the question is not "how much arithmetic does an
ensemble need" but "does the GPU notice". Four measurements:

  1. BATCH SCALING. Latency at batch 1,2,4,8,16 for one model. If batch 8 costs
     about what batch 1 costs, the card has ~8x of idle width -- and that idle
     width is exactly what an ensemble would occupy. This is the robust proxy:
     it needs no new machinery and cannot fail for API reasons.
  2. N MODELS, SEQUENTIAL LOOP. The naive ensemble. Expect ~Nx if we are
     launch-bound, because N loops means N times as many kernel launches.
  3. N MODELS, VMAP-BATCHED. torch.func.stack_module_state stacks N weight sets
     and vmap runs them as ONE set of kernels doing N times the arithmetic.
     Same number of launches. If we are launch-bound this should be far cheaper
     than the loop. May fail -- the vendored Flower does explicit reshapes and
     grid_sample, and not every op has a batching rule. Failure is reported,
     not fatal; measurement 1 still answers the question in principle.
  4. CUDA GRAPHS via torch.compile(mode="reduce-overhead"). Attacks the launch
     overhead directly by replaying a captured sequence as one submission.
     Orthogonal to ensembling and useful on its own.

Everything is converted to time_score and to a composite delta at submission
7's operating point, because "1.4x slower" means nothing until it is priced.
"""
import copy, sys, time, torch
sys.path.insert(0, 'kit')
from scoring import score_time, T_NUMERICAL_SEC
from pde_competition.model import FlowerS2S, FlowerConfig
from pde_competition.composite import estimate

S7 = dict(rel_l2=94.536536, tke=74.636151, mvpe=92.912553, time=91.397614, sps=36.382652)
BASE = estimate(**S7)
r_ref = (100.0 / S7['time'] - 1.0) ** 2
t_ref = r_ref * T_NUMERICAL_SEC          # grader-side seconds for the d64 model
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = FlowerConfig(lifting_dim=128, n_levels=2)      # the shipping config
print(f"device {dev}   config d128/nl2 ({sum(p.numel() for p in FlowerS2S(CFG).parameters())/1e6:.2f} M)")
print(f"grader reference: d64 model = {t_ref*1000:.3f} ms/window at time_score {S7['time']:.2f}\n")

def bench(fn, x, iters=30, warmup=8):
    with torch.no_grad():
        for _ in range(warmup): fn(x)
        if dev == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): fn(x)
        if dev == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def price(t_per_window, label):
    """Convert a per-window latency RATIO into time_score and composite delta."""
    ratio = t_per_window / t1            # relative to one model at batch 1
    ts = score_time(t_ref * ratio)
    s2 = dict(S7); s2['time'] = ts
    print(f"  {label:<34}{t_per_window*1000:8.3f} ms{ratio:8.2f}x{ts:11.2f}{estimate(**s2)-BASE:+10.3f}")

m = FlowerS2S(CFG).to(dev).eval()
x1 = torch.randn(1, 20, 32, 64, 3, device=dev)
t1 = bench(m, x1)

print("1. BATCH SCALING -- does the card notice more work?")
print(f"  {'':34}{'latency':>11}{'vs b=1':>8}{'per-window':>13}")
for b in (1, 2, 4, 8, 16):
    xb = torch.randn(b, 20, 32, 64, 3, device=dev)
    t = bench(m, xb)
    print(f"  batch {b:<28}{t*1000:8.3f} ms{t/t1:8.2f}x{t/b*1000:11.3f} ms")

print(f"\n{'':36}{'latency':>10}{'vs 1':>9}{'time_score':>11}{'d(final)':>10}")
price(t1, "1 model, batch 1 (reference)")

print("\n2. N MODELS, SEQUENTIAL LOOP")
for N in (2, 3, 5):
    ms = [FlowerS2S(CFG).to(dev).eval() for _ in range(N)]
    price(bench(lambda x: [mm(x) for mm in ms], x1), f"{N} models, looped")

print("\n3. N MODELS, VMAP-BATCHED")
try:
    from torch.func import stack_module_state, functional_call
    for N in (2, 3, 5):
        ms = [FlowerS2S(CFG).to(dev).eval() for _ in range(N)]
        params, buffers = stack_module_state(ms)
        base = copy.deepcopy(ms[0]).to('meta')
        def f(p, b, xx): return functional_call(base, (p, b), (xx,))
        vm = torch.vmap(f, in_dims=(0, 0, None))
        price(bench(lambda x: vm(params, buffers, x), x1), f"{N} models, vmap")
except Exception as e:
    print(f"  VMAP FAILED: {type(e).__name__}: {str(e)[:120]}")
    print("  (measurement 1 still bounds the answer: spare batch width is spare")
    print("   ensemble width, whatever the API used to reach it)")

print("\n4. CUDA GRAPHS -- torch.compile(mode='reduce-overhead'), 1 model")
try:
    mc = torch.compile(FlowerS2S(CFG).to(dev).eval(), mode="reduce-overhead")
    price(bench(mc, x1, iters=30, warmup=15), "1 model, reduce-overhead")
except Exception as e:
    print(f"  COMPILE FAILED: {type(e).__name__}: {str(e)[:120]}")


print("\n5. HAND-CAPTURED CUDA GRAPHS (torch.cuda.CUDAGraph, no compiler) -- what the")
print("   submission actually ships. 1 model, and the vmap ensemble under capture:")
print("   if the card is launch-bound, N models replayed as one graph should cost")
print("   little more than one model replayed as one graph.")
def captured(fn, x):
    """Capture fn(x) into a CUDA graph with static buffers; return a replay callable."""
    sx = x.clone()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn(sx)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sy = fn(sx)
    def replay(xx):
        sx.copy_(xx); g.replay(); return sy
    return replay
if dev == 'cuda':
    try:
        price(bench(captured(m, x1), x1), "1 model, captured graph, b=1")
        x16 = torch.randn(16, 20, 32, 64, 3, device=dev)
        t16 = bench(captured(m, x16), x16)
        price(t16 / 16, "1 model, captured graph, b=16 (per window)")
        from torch.func import stack_module_state, functional_call
        for N in (3, 5):
            ms = [FlowerS2S(CFG).to(dev).eval() for _ in range(N)]
            params, buffers = stack_module_state(ms)
            base = copy.deepcopy(ms[0]).to('meta')
            def f(p, b, xx): return functional_call(base, (p, b), (xx,))
            vm = torch.vmap(f, in_dims=(0, 0, None))
            price(bench(captured(lambda xx: vm(params, buffers, xx), x1), x1), f"{N} models, vmap + captured graph")
        for N in (3, 5):
            ms = [FlowerS2S(CFG).to(dev).eval() for _ in range(N)]
            price(bench(captured(lambda xx: torch.stack([mm(xx) for mm in ms]), x1), x1), f"{N} models, loop + captured graph")
    except Exception as e:
        print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:120]}")
else:
    print("  (no CUDA device)")
