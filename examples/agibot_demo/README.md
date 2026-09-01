# AgiBot Demo Cases

This directory contains three compact cases:

- `327/685046`: produce pickup and bag placement.
- `388/685716`: packaged grocery manipulation.
- `446/687380`: table movement assistance.

Inputs are under `observations/` and `proprio_stats_extracted/`. Expected
VLM-reviewed outputs are under `expected_outputs/`.

The expected outputs are demo artifacts with repo-relative paths. Reuse them
with `--reuse-output-root` or as reviewed decision examples; deterministic
bootstrap decisions are for baseline/debug inspection only and are not final
semantic results.

Install dependencies from the repo root before running the demo:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the demo reuse path from the repo root:

```bash
python tools/run_no_gt_vision_traj_subset.py \
  --observations-root examples/agibot_demo/observations \
  --proprio-root examples/agibot_demo/proprio_stats_extracted \
  --output-root outputs/demo_vlm_reviewed \
  --reuse-output-root examples/agibot_demo/expected_outputs \
  --episodes-per-task 1 \
  --task-ids 327,388,446 \
  --views head_color,hand_left_color,hand_right_color \
  --sample-fps 4 \
  --candidate-count 16 \
  --trajectory-candidate-count 12
```

See `cases.json` for segment counts and audit artifact references.
