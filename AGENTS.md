# Robot TAS Agent Notes

This project implements the current AgiBotWorld vision + trajectory TAS pipeline.

- Treat `robot_tas/` as the core library.
- Treat `robot_tas/cli/` as the source of truth for command implementations.
- Use `scripts/` for direct source-tree execution of the main pipeline.
- Use `tools/` for probes, summaries, dataset helpers, and legacy conversion wrappers.
- Keep configs in `configs/`, prompts in `prompts/`, runbooks in `docs/`, and tests in `tests/`.
- The final decision is visual-first: overhead video has weight `0.50`, tri-view video `0.25`, and pixel CV prior `0.25`; trajectory is used for candidate recall, timing refinement, and separate evidence, not as a standalone boundary source.
- Do not use `task_info` or GT before prediction is locked. They are evaluation-only inputs.
- Preserve `selected_views`, `view_evidence`, and `trajectory_evidence` in final outputs.

Run verification from this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```
