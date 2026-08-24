# Final Pipeline

This is the maintained runbook for the no-GT AgiBotWorld multi-view TAS
pipeline.

## Source Of Truth

Final semantic segmentation must come from:

```text
decision/codex_vlm_decision.json
prompt_version = codex_vlm_decision_v2
```

The decision must include `inspection_audit` with reviewed visual sources,
semantic change observations, trajectory change observations, and
under-segmentation checks.

Deterministic bootstrap decisions may be generated for pack inspection only.
They are not final semantic results.

## Core Flow

For each task and episode:

1. Discover episode directories with `robot_tas.cli.evaluate_agibot_task.find_episode_dirs`.
2. Run `scripts/run_tas.py` on `videos/head_color.mp4` to create the base timeline and sampled frames.
3. Run `scripts/prepare_multiview_codex_pack.py` to build synchronized head/left-hand/right-hand evidence and optional trajectory references.
4. Review the evidence with `prompts/codex_vlm_decision_v2.txt` and write `decision/codex_vlm_decision.json`.
5. Run `scripts/apply_codex_vlm_decision.py` to create `final/final_segments.json`, `final/timeline.html`, and `final/annotated_actions.mp4`.
6. Run `tools/audit_vlm_undersegmentation_risk.py` and `tools/build_segment_review_sheets.py` for final review artifacts.

`head_color.mp4` is the base timeline and final visualization view. It is not
the only semantic evidence source.

## Evidence Rules

- Use `head_color`, `hand_left_color`, and `hand_right_color` as the fixed tri-view review set.
- Treat CV scores as proposal/recall signals only.
- Treat trajectory events as proposal/timing/cross-check signals only.
- Do not infer grasp from gripper closure alone.
- Do not infer release from gripper opening alone.
- Confirm grasp/release/carry with visual evidence from hand views and global placement evidence from the head view.
- Every final boundary must be a supplied sampled frame.
- Every segment must state left-hand and right-hand status.

## Batch Command

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

For a fresh run, omit `--reuse-output-root` only after the reviewed
`decision/codex_vlm_decision.json` files are present in the target output tree.

## Per-Episode Layout

```text
task_<TASK>/episodes/<EPISODE>/
  base_head_pipeline/
    metadata.json
    sampled_frames/
  multiview_pack/
    manifest.json
    fused_cv_reference.json
    candidate_reference.json
    trajectory_reference.json
    contact_sheets/
  decision/
    codex_vlm_decision.json
    inspection_audit.json
  final/
    codex_vlm_decision.json
    final_segments.json
    timeline.html
    annotated_actions.mp4
```

Flat copies are written to `outputs/.../annotated_videos/`.

## Demo Cases

The repo includes three compact cases under `examples/agibot_demo`:

- task 327 episode 685046
- task 388 episode 685716
- task 446 episode 687380

Each case includes the three required input videos, extracted proprio
`timeseries.csv`, the audited decision JSON, final output files, and only the
review sheets referenced by the audit.

## Verification

```bash
pytest -q

python tools/audit_vlm_undersegmentation_risk.py \
  --output-root outputs/demo_vlm_reviewed
```

The bundled demo reuse path already includes the segment review sheets cited by
the audited decisions. Run `tools/build_segment_review_sheets.py` only on fresh
outputs that include full `multiview_pack/views/*/sampled_frames` artifacts.

Check that every final `final_segments.json` has `prompt_versions.codex_vlm_decision`
set to `codex_vlm_decision_v2`, no unsupported object identity, no invented
frame IDs, and no broad placeholder captions such as `visible item` or
`target object`.
