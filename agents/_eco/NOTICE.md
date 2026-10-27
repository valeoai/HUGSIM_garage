# agents/_eco: ECO

`eco.py` is **our reimplementation** of ECO (Zhang et al., arXiv 2609.31383): no code was released with
the paper. Its module docstring and `agents/_vavam/NOTICE.md` (section *ECO*) list what follows the paper
and what is our choice. Used by `vavam_eco` (VaVAM-B's plans, headings from the segments) and
`drivor_eco` (DrivoR's plans, with DrivoR's own headings). `test_eco.py` checks the objective against the
paper's sums: `python -m agents._eco.test_eco`.
