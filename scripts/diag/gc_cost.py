"""Import a submission zip's module exactly as the grader does (fresh process, import runs the warm-up), then time
what a cyclic-GC pass would cost right after the import -- the window the grader's gap_import_to_call_s falls in.

    python scripts/diag/gc_cost.py ZIP

Prints gc.isenabled(), the object count tracked by gc, and the wall time of gc.collect(0/1/2) after import.
A GCF module (gc frozen + disabled at import end) should show isenabled False and a ~0 ms full collect.
"""
import gc, sys, tempfile, time, zipfile, pathlib

z = sys.argv[1]
d = tempfile.mkdtemp()
zipfile.ZipFile(z).extractall(d)
sys.path.insert(0, d)
t0 = time.perf_counter()
import submission  # noqa: E402,F401
t1 = time.perf_counter()
print(f"import {t1 - t0:.3f}s  gc.isenabled={gc.isenabled()}  tracked objects={len(gc.get_objects())}  "
      f"frozen={gc.get_freeze_count()}  counts={gc.get_count()}")
for gen in (0, 1, 2):
    t = time.perf_counter(); n = gc.collect(gen); dt = (time.perf_counter() - t) * 1e3
    print(f"gc.collect({gen}) after import: {dt:.2f} ms  (collected {n})")
