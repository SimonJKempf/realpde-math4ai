"""The shape of the Track 1 problem: window length, grid, channels.

Deliberately imports nothing -- not torch, not numpy, not ``.paths``.
``dataset.py`` and ``model.py`` both read from here so there is one definition,
and a submission zip (flat, no package, no data layer) can carry this file
beside the model without dragging anything else in. Adding an import here
breaks that; don't.

Values are the scored geometry, taken from the organizers' starting kit: the raw PIV is
64x128 and Track 1 subsamples it by 2 (strided, NOT averaged), so scoring runs
at 32x64. The subsampling factor itself lives in scripts/subsample_data.py --
it describes how data/processed was produced, not what the model sees.
"""
T_IN, T_OUT = 20, 20
H_S, W_S = 32, 64
C = 3                         # [u, v, p]; p is identically 0 for real data
