"""Test package marker -- and one environment setting that must precede `import torch`.

flower_standalone decorates its grid-sample helpers with @torch.compile. On CPU
that drives the inductor backend, which shells out to clang++ and depends on a
precompiled-header cache under $TMPDIR that goes stale between torch upgrades
("has been modified since the precompiled header was built"). A unit test must
not depend on a compiler cache, so dynamo is disabled here: eager execution is
numerically identical and is exactly what the submission runs
(submission_flower.py sets the same variable, for the same reason).

unittest discovery imports this package before any test module, so the setting
is in place before torch is.
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
