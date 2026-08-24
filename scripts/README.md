# Scripts

Source-tree wrappers for the maintained pipeline steps:

- `run_tas.py`: sample the base `head_color` timeline and create rough proposals.
- `prepare_multiview_codex_pack.py`: build synchronized tri-view evidence packs.
- `apply_codex_vlm_decision.py`: convert an audited `codex_vlm_decision_v2` JSON into final outputs.

The implementations live in `robot_tas/cli/`.
