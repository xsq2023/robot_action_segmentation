from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_tas.undersegmentation_risk import find_undersegmentation_risks, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report VLM final segments that still need explicit under-segmentation review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", required=True, help="VLM output root containing task_*/episodes/*.")
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional JSON report path. Defaults to <output-root>/codex_vlm_undersegmentation_risk_report.json.",
    )
    return parser.parse_args()


def _decision_segments(episode_dir: Path) -> list[dict[str, Any]]:
    decision_path = episode_dir / "decision" / "codex_vlm_decision.json"
    if decision_path.is_file():
        return list(read_json(decision_path).get("segments", []))
    final_path = episode_dir / "final" / "final_segments.json"
    if final_path.is_file():
        return list(read_json(final_path).get("segments", []))
    return []


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    report_path = Path(args.write_report).resolve() if args.write_report else output_root / "codex_vlm_undersegmentation_risk_report.json"
    episodes: list[dict[str, Any]] = []

    for episode_dir in sorted(output_root.glob("task_*/episodes/*")):
        if not episode_dir.is_dir():
            continue
        segments = _decision_segments(episode_dir)
        risks = find_undersegmentation_risks(episode_dir=episode_dir, segments=segments)
        episodes.append(
            {
                "task_id": episode_dir.parents[1].name.removeprefix("task_"),
                "episode_id": episode_dir.name,
                "segment_count": len(segments),
                "risk_count": len(risks),
                "risks": risks,
            }
        )

    report = {
        "description": (
            "Segments listed here are not automatically wrong. They contain long non-terminal actions, "
            "internal gripper events, or dense trajectory/visual candidates and therefore require explicit "
            "VLM under-segmentation review."
        ),
        "output_root": str(output_root),
        "episode_count": len(episodes),
        "risk_segment_count": sum(item["risk_count"] for item in episodes),
        "episodes": episodes,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {report_path}")
    print(f"episodes={report['episode_count']} risk_segments={report['risk_segment_count']}")
    for episode in episodes:
        if not episode["risk_count"]:
            continue
        print(f"task_{episode['task_id']} ep_{episode['episode_id']} risk_segments={episode['risk_count']}")
        for risk in episode["risks"]:
            print(
                "  "
                f"seg={risk['segment_index']} {risk['action_label']} "
                f"{risk['frame_range'][0]}-{risk['frame_range'][1]} "
                f"events={risk['required_internal_event_frames']} "
                f"reasons={','.join(risk['risk_reasons'])}"
            )


if __name__ == "__main__":
    main()
