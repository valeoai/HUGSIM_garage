"""Code shared by the NAVSIM-family baselines (LTF, DrivoR, TOAD, GTRS, ZTRS, GTRS-Aug), inference only.

Each agent is a folder of its own next to this one (``agents/ltf``, ``agents/drivor``, ...) with a
``spec.py`` that :mod:`.policy` loads by name; what two or more of them use is here. See
``NOTICE.md`` for where each file comes from. ``python -m agents._navsim.run --help``.
"""

# The NAVSIM-family agent names: the single list ``policy.py``, ``run.py`` and ``registry.py`` use.
AGENTS = ('ltf', 'drivor', 'toad', 'toad_release', 'gtrs', 'ztrs', 'gtrs_aug')
