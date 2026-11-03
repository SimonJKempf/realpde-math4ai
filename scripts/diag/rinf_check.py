"""RINF sanity: does the shipped module run a --rollout-frames 10 checkpoint the way training does?"""
import sys, zipfile, tempfile, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from pde_competition.model import FlowerS2S
ck = sys.argv[1]; zp = sys.argv[2]
m = FlowerS2S.from_checkpoint(ck).eval()
print("cfg rollout_frames", m.cfg.rollout_frames, "t_out", m.cfg.t_out, "out_frames", getattr(m.cfg, "out_frames", None))
raw = torch.load(ck, map_location="cpu", weights_only=False)
c = raw.get("config") or raw.get("cfg")
print("ckpt config keys with rollout/t_out:", {k: v for k, v in (c.items() if isinstance(c, dict) else vars(c).items()) if "roll" in k or "t_out" in k or "t_in" in k})
rng = np.random.default_rng(0)
x = (rng.standard_normal((2, 20, 32, 64, 3)) * 0.05).astype(np.float32); x[..., 2] = 0
with torch.no_grad():
    ref = m(torch.from_numpy(x)).numpy()
work = pathlib.Path(tempfile.mkdtemp()); zipfile.ZipFile(zp).extractall(work)
sys.path.insert(0, str(work)); import submission
sub = submission._get_model() if hasattr(submission, "_get_model") else None
print("zip model cfg:", {k: v for k, v in (sub.cfg.items() if hasattr(sub, "cfg") and isinstance(sub.cfg, dict) else {}).items() if "roll" in k or "t_out" in k})
