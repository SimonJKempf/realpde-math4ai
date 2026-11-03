#!/usr/bin/env python
"""Does every built ensemble zip ship exactly the members its ensemble.json records?

    python scripts/audit_zip_members.py [--glob 'submission/dist/*.zip']

WHY. build_submission.py wrote model.pth from --ckpt while _payload_members staged members[1:] and recorded
members[0], so --ckpt != members[0] silently shipped the wrong member set. Found 2026-09-23 after it had
corrupted 162 of 611 built ensemble zips (no submitted zip). There are TWO modes and they need different
checks, which is why this compares against the recorded SOURCE rather than merely checking distinctness:
  duplicated  --ckpt was inside --ensemble  -> a member appears twice, one is dropped. Distinctness catches it.
  substituted --ckpt was outside --ensemble -> a model the record does not name ships, one is dropped, and
              every member stays DISTINCT. Distinctness does NOT catch it; 17 zips are this.
A `last.pth` source is tolerated: _stage_weights re-saves those, so the bytes legitimately differ.

Run it through Slurm, not on the login node: it CRCs thousands of checkpoint files.
"""
import argparse, glob, json, zipfile, zlib, collections, os


def crc(path, buf=1 << 22):
    h = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h
            h = zlib.crc32(b, h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="submission/dist/*.zip")
    a = ap.parse_args()
    cat = collections.Counter()
    bad = collections.defaultdict(list)
    cache = {}
    for z in sorted(glob.glob(a.glob)):
        try:
            with zipfile.ZipFile(z) as zf:
                if "ensemble.json" not in zf.namelist():
                    continue
                ej = json.loads(zf.read("ensemble.json"))
                shipped = {i.filename: i.CRC for i in zf.infolist()
                           if i.filename == "model.pth" or i.filename.startswith("model_")}
        except Exception:
            cat["unreadable"] += 1
            continue
        mism = []
        for name, src in zip(ej.get("members", []), ej.get("sources", [])):
            if not os.path.exists(src):
                continue
            if src not in cache:
                cache[src] = crc(src)
            if cache[src] != shipped.get(name):
                mism.append("last" if src.endswith("last.pth") else "real")
        if not mism:
            cat["CLEAN"] += 1
        elif all(m == "last" for m in mism):
            cat["benign last.pth re-save"] += 1
        elif len(set(shipped.values())) < len(shipped):
            cat["BUG duplicated"] += 1
            bad["duplicated"].append(os.path.basename(z))
        else:
            cat["BUG substituted"] += 1
            bad["substituted"].append(os.path.basename(z))
    for k, v in cat.most_common():
        print(f"  {k:28} {v:5d}")
    print(f"  {'TOTAL CORRUPTED':28} {cat['BUG duplicated'] + cat['BUG substituted']:5d}")
    subs = [n for lst in bad.values() for n in lst if n.startswith("sub")]
    print(f"\n  submitted-style (sub*.zip) corrupted: {subs or 'NONE'}")
    for mode, lst in bad.items():
        pref = collections.Counter(n.split("_")[1] if "_" in n else "?" for n in lst)
        print(f"  {mode}: {len(lst)}  by experiment: {dict(pref.most_common(8))}")


if __name__ == "__main__":
    main()
