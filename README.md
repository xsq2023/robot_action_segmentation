# Robot TAS

No-GT multi-view temporal action segmentation for robot manipulation videos.

This repo is organized around the final pipeline described in
`prompts/final_prompt.txt`: synchronized head/hand evidence is used for visual
semantic state segmentation, while CV and trajectory signals are candidate
recall only. Final semantic claims must come from an audited
`codex_vlm_decision_v2` decision JSON.

## Layout

- `robot_tas/`: core sampling, segmentation, visualization, trajectory, and VLM-pack utilities.
- `robot_tas/cli/`: installed CLI implementations.
- `scripts/`: source-tree wrappers for the main single-episode steps.
- `tools/`: batch no-GT runner plus final audit/review utilities.
- `prompts/`: maintained base prompts and `codex_vlm_decision_v2.txt`.
- `examples/agibot_demo/`: three small AgiBotWorld demo cases with inputs and expected VLM-reviewed outputs.
- `tests/`: deterministic unit tests.

## Setup

```bash
cd "/mnt/f/ai/video grounding/api"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

For runtime-only use, `pip install -r requirements.txt` is enough.

## Demo Cases

The bundled cases use this layout:

```text
examples/agibot_demo/
  observations/<task>/<episode>/videos/
    head_color.mp4
    hand_left_color.mp4
    hand_right_color.mp4
  proprio_stats_extracted/<task>/<episode>/timeseries.csv
  expected_outputs/task_<task>/episodes/<episode>/
    decision/codex_vlm_decision.json
    final/final_segments.json
    final/timeline.html
    final/annotated_actions.mp4
```

Open `examples/agibot_demo/cases.json` for the selected task/episode IDs and
segment counts.

## Final No-GT Batch Runner

This command processes the demo observations and reuses the bundled final
VLM-reviewed outputs as the locked decisions/results:

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

For a fresh run without `--reuse-output-root`, place each reviewed
`decision/codex_vlm_decision.json` under the target episode output directory
before the apply step. Do not use `--allow-bootstrap-decisions` for final
semantic results.

## Single Episode Steps

```bash
python scripts/run_tas.py \
  --video examples/agibot_demo/observations/327/685046/videos/head_color.mp4 \
  --output-dir outputs/demo_single/task_327/episodes/685046/base_head_pipeline \
  --model codex-local \
  --sample-fps 4 \
  --window-size 16 \
  --window-stride 8 \
  --force

python scripts/prepare_multiview_codex_pack.py \
  --episode-dir examples/agibot_demo/observations/327/685046 \
  --output-dir outputs/demo_single/task_327/episodes/685046/multiview_pack \
  --sample-fps 4 \
  --views head_color,hand_left_color,hand_right_color \
  --trajectory-csv examples/agibot_demo/proprio_stats_extracted/327/685046/timeseries.csv \
  --candidate-count 16 \
  --trajectory-candidate-count 12 \
  --force

python scripts/apply_codex_vlm_decision.py \
  --pipeline-output-dir outputs/demo_single/task_327/episodes/685046/base_head_pipeline \
  --decision examples/agibot_demo/expected_outputs/task_327/episodes/685046/decision/codex_vlm_decision.json \
  --output-dir outputs/demo_single/task_327/episodes/685046/final
```

The final output is `final/final_segments.json`, `final/timeline.html`, and
`final/annotated_actions.mp4`.
