from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VIEWS = ("head_color", "hand_left_color", "hand_right_color")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build enlarged frame-row review sheets for current final segments using head/left/right view frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default="outputs/codex_vlm_fresh_multiview")
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--view-thumb-width", type=int, default=360, help="Width of each enlarged view column cell.")
    parser.add_argument("--view-thumb-height", type=int, default=270, help="Height of each enlarged view column cell.")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_view_frames(view_dir: Path) -> dict[int, Path]:
    payload = _read_json(view_dir / "metadata.json")
    frames: dict[int, Path] = {}
    for item in payload["sampled_frames"]:
        frame_id = int(item["original_frame_id"])
        frames[frame_id] = view_dir / item["image_path"]
    return frames


def _nearest_frame(frame_id: int, available: list[int]) -> int:
    return min(available, key=lambda value: (abs(value - frame_id), value))


def _fit_text(text: str, max_chars: int = 150) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (20, 20, 20))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _frame_review_rows(final: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for segment in final["segments"]:
        start_frame = int(segment["start_frame_id"])
        end_frame = int(segment["end_frame_id"])
        points = [
            ("start", start_frame),
            ("mid", (start_frame + end_frame) // 2),
            ("end", end_frame),
        ]
        for point_name, frame_id in points:
            key = (int(segment["segment_id"]), point_name, frame_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "segment_id": int(segment["segment_id"]),
                    "point": point_name,
                    "frame_id": frame_id,
                    "action_label": str(segment["action_label"]),
                    "description": str(segment.get("description", "")),
                }
            )
    return rows


def _build_sheet(final_path: Path, output_path: Path, view_thumb_size: tuple[int, int]) -> None:
    episode_root = final_path.parents[1]
    pack_dir = episode_root / "multiview_pack"
    final = _read_json(final_path)
    view_frames = {view: _load_view_frames(pack_dir / "views" / view) for view in VIEWS if (pack_dir / "views" / view).exists()}
    if not view_frames:
        return
    available_by_view = {view: sorted(frames) for view, frames in view_frames.items()}

    rows = _frame_review_rows(final)
    thumb_width, thumb_height = view_thumb_size
    header_height = 44
    row_label_height = 58
    margin = 8
    sheet_width = margin * 2 + thumb_width * len(view_frames)
    sheet_height = header_height + (row_label_height + thumb_height) * len(rows) + margin
    image = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = f"{final_path.parents[3].name} episode {final_path.parents[1].name} enlarged tri-view frame review"
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for col, view in enumerate(view_frames):
        draw.text((margin + col * thumb_width + 6, header_height - 24), view, fill=(0, 0, 0), font=font)

    y = header_height
    for row in rows:
        draw.rectangle((0, y, sheet_width, y + row_label_height - 1), fill=(230, 230, 230))
        label = f"segment {row['segment_id']} {row['point']} frame {row['frame_id']:04d}: {row['action_label']}"
        draw.text((margin + 6, y + 7), label, fill=(0, 0, 0), font=font)
        for line_index, line in enumerate(_fit_text(row["description"])):
            draw.text((margin + 6, y + 24 + line_index * 13), line, fill=(50, 50, 50), font=font)
        y += row_label_height
        for col, (view, frames) in enumerate(view_frames.items()):
            nearest = _nearest_frame(int(row["frame_id"]), available_by_view[view])
            x = margin + col * thumb_width
            tile = _thumb(frames[nearest], view_thumb_size)
            image.paste(tile, (x, y))
            draw.rectangle((x, y, x + thumb_width - 1, y + thumb_height - 1), outline=(35, 35, 35))
            draw.text((x + 3, y + 3), f"{view.replace('_color', '')} f={nearest}", fill=(255, 255, 255), font=font)
        y += thumb_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    review_dir = Path(args.review_dir).resolve() if args.review_dir else output_root / "segment_review_sheets"
    view_thumb_size = (args.view_thumb_width, args.view_thumb_height)
    final_paths = sorted(output_root.glob("task_*/episodes/*/final/final_segments.json"))
    for final_path in final_paths:
        task_id = final_path.parents[3].name.removeprefix("task_")
        episode_id = final_path.parents[1].name
        output_path = review_dir / f"task_{task_id}_episode_{episode_id}_segments.jpg"
        _build_sheet(final_path, output_path, view_thumb_size=view_thumb_size)
        print(output_path)


if __name__ == "__main__":
    main()
