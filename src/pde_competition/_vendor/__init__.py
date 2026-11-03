"""Third-party model code copied into this repo. Treat as read-only.

Nothing here is ours -- see README.md for where each file came from. Edits go
in a clearly marked block at the top of the file, never silently inline, or the
next upstream sync will quietly clobber them.

Importing this package puts the kit's vendored einops on sys.path, because
``flower_standalone`` does ``from einops import rearrange`` and einops is
deliberately NOT installed in the conda env -- the Track 1 eval container has
no einops either, so a local ImportError is an honest preview of a container
failure. We borrow kit/_vendor/einops rather than keeping a second copy: that
is the exact tree scripts/build_submission.py ships in the zip, so what we
train against and what gets graded cannot drift apart.
"""
import sys

from pde_competition.paths import KIT

_EINOPS_SRC = KIT / "_vendor"
if str(_EINOPS_SRC) not in sys.path:
    sys.path.insert(0, str(_EINOPS_SRC))
