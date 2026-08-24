# Tools

Supporting tools for the final no-GT multi-view pipeline:

- `run_no_gt_vision_traj_subset.py`: batch runner for local AgiBotWorld episodes.
- `audit_vlm_undersegmentation_risk.py`: report segments that need extra visual review.
- `build_segment_review_sheets.py`: build final segment review sheets.
- `build_multiview_observations.py`: build multi-view evidence packs from an observations root.

The core library and CLI implementations live in `robot_tas/` and `robot_tas/cli/`.
