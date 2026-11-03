# Vendored third-party code

Code copied in from elsewhere because the Track 1 eval container has no network
and no scientific-Python stack beyond `torch` / `numpy`. Not ours, not covered
by this repo's license, and not to be edited casually.

Keep this file current. A vendored file with no recorded origin is unmaintainable:
you can never tell whether upstream has since fixed the bug you are staring at.

---

## `flower_standalone.py`

Flower, a U-Net-style neural operator with spatial-warping layers. Single-file
build with all dependencies inlined for drag-and-drop use.

| | |
|---|---|
| Upstream repo | <https://github.com/t-muser/flowers> (file at repo root) |
| Commit / tag | `f1d2cad239e7022a1cd9ced51e7d9c701c9a6317` (`master`, pushed 2026-06-03) |
| Paper | Muser, Spitzer, Lassas, de Hoop, Dokmanic, *Flowers: A Warp Drive for Neural PDE Solvers* — <https://arxiv.org/abs/2603.04430> |
| License | MIT, (c) 2026 Till Muser / SADA Laboratory (UniBas) — copied to `LICENSE.flower` |
| Vendored on | 2026-08-28 (file mtime) |
| Local edits | none — sha256 `d0008930c48a8066baecdfe36f965dbfe12ba19b6cba4cdb96bf14a9fbfadfae`, verified byte-identical to upstream `master` on 2026-09-01 |

Upstream's `LICENSE` is dual: MIT for the repo, BSD 3-Clause (c) 2024 Polymathic AI
for the parts adapted from *The Well*. Both texts are in `LICENSE.flower`. Only the
model file is vendored here, not the training code, but the whole notice is kept
because upstream ships it as one file.

To re-check provenance after an upstream sync:

```bash
curl -sS https://raw.githubusercontent.com/t-muser/flowers/master/flower_standalone.py \
  | shasum -a 256   # compare against the sha256 above
```


### Notes

- **Needs `einops`**, which is not installed. `_vendor/__init__.py` puts the
  kit's vendored copy on `sys.path`, so importing through this package works.
  Importing `flower_standalone` by any other route will fail on einops.
- **`spatial_resolution` is required at construction** and is baked into the
  coordinate grids and warping layers. Unlike a plain CNN, one instance does not
  accept arbitrary input sizes. Our eval grid is 32x64 (see `data.py`).
- **Constructor args are `dim_in` / `dim_out`**, not `in_channels` / `out_channels`.
- **Do not `torch.save(model)` — save `state_dict()`.** Pickling the whole object
  records the class path `pde_competition._vendor.flower_standalone.Flower`, which
  does not exist inside the flat submission zip and will fail to load there.
- Its docstrings reference `configs/models/flower.yaml`; that is an *upstream*
  path, not a file in this repo.

## Local patches (keep this list complete)

The vendored `flower_standalone.py` is upstream's file with two minimal,
documented changes. Both are marked in place with a `VENDORED FILE, MINIMALLY
PATCHED` comment giving the date and the reason.

1. **2026-09-14 — `derive_padding_strategies` recognises `BORDER` and
   `REFLECTION`.** Upstream mapped every string that is not `"PERIODIC"` onto
   `'zeros'`, so the `border` and `reflection` modes that
   `custom_grid_sample2d` already supports were unreachable from config.
   Instrumenting a trained checkpoint showed **8.58% of warp samples fall
   outside the grid** (cross-stream 5.95%, streamwise 2.87%, reaching 1.84), so
   one sample in twelve was answered by a wraparound or a zero. `border` is the
   zero-gradient outflow treatment. Unknown strings still fall back to `zeros`.

2. **2026-09-14 — `out_ch` is seeded before the decoder loop**, so `n_levels=1`
   builds. The loop that assigns it is empty at one level, and the final
   projection raised `UnboundLocalError: cannot access local variable 'out_ch'`.
   With one level there is no down/upsampling, so the tensor reaching `project`
   is the bottleneck's own output and `out_ch = bottleneck_channels`. Every
   `n_levels >= 2` path is byte-identical, because the loop overwrites the seed
   on its first iteration. Upstream's configs are all 4 levels; ours are not —
   the depth ladder has improved at every rung down to 2.
