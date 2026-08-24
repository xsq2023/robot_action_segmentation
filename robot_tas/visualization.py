from __future__ import annotations

import html
import logging
from pathlib import Path
import subprocess
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_tas.cache import ensure_dir
from robot_tas.schemas import LabeledSegment, MergedBoundary, SampledFrame, VideoMetadata


LOGGER = logging.getLogger(__name__)


def _segment_rows(segments: list[LabeledSegment]) -> str:
    return "\n".join(
        (
            "<tr>"
            f"<td>{segment.segment_id}</td>"
            f"<td>{html.escape(segment.action_label)}</td>"
            f"<td>{segment.start_time:.3f}</td>"
            f"<td>{segment.end_time:.3f}</td>"
            f"<td>{segment.start_frame_id}</td>"
            f"<td>{segment.end_frame_id}</td>"
            f"<td>{segment.confidence:.2f}</td>"
            f"<td>{html.escape(segment.description)}</td>"
            "</tr>"
        )
        for segment in segments
    )


def _boundary_rows(boundaries: list[MergedBoundary]) -> str:
    return "\n".join(
        (
            "<tr>"
            f"<td>{boundary.boundary_sample_index}</td>"
            f"<td>{boundary.boundary_frame_id}</td>"
            f"<td>{boundary.boundary_time:.3f}</td>"
            f"<td>{html.escape(boundary.transition_type)}</td>"
            f"<td>{boundary.confidence:.2f}</td>"
            f"<td>{html.escape('; '.join(boundary.visual_evidence))}</td>"
            "</tr>"
        )
        for boundary in boundaries
    )


def _thumbnail_strip(sampled_frames: list[SampledFrame], boundaries: set[int]) -> str:
    cards: list[str] = []
    for frame in sampled_frames:
        badge = '<div class="badge">boundary</div>' if frame.sample_index in boundaries else ""
        cards.append(
            "<div class='thumb'>"
            f"{badge}"
            f"<img src='{html.escape(frame.image_path)}' alt='sample {frame.sample_index}'>"
            f"<div class='meta'>sample {frame.sample_index}<br>frame {frame.original_frame_id}<br>{frame.timestamp_seconds:.3f}s</div>"
            "</div>"
        )
    return "\n".join(cards)


def write_timeline_html(
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    boundaries: list[MergedBoundary],
    segments: list[LabeledSegment],
    output_path: Path,
) -> None:
    """Write a lightweight HTML timeline visualization."""

    boundary_set = {boundary.boundary_sample_index for boundary in boundaries}
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Robot TAS Timeline</title>
  <style>
    body {{
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      margin: 24px;
      color: #1d1d1d;
      background: linear-gradient(180deg, #f4efe6 0%, #ffffff 100%);
    }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }}
    .thumb {{
      position: relative;
      border: 1px solid #cfc6b9;
      border-radius: 12px;
      background: #fffdf9;
      padding: 8px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    }}
    .thumb img {{
      width: 100%;
      border-radius: 8px;
      display: block;
    }}
    .meta {{
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.4;
    }}
    .badge {{
      position: absolute;
      top: 10px;
      left: 10px;
      background: #b33a3a;
      color: white;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 24px;
      background: white;
    }}
    th, td {{
      border: 1px solid #ddd3c5;
      padding: 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #efe6d8;
    }}
    .meta-box {{
      background: white;
      border: 1px solid #ddd3c5;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 24px;
    }}
  </style>
</head>
<body>
  <h1>Robot TAS Timeline</h1>
  <div class="meta-box">
    <div><strong>Video:</strong> {html.escape(metadata.path)}</div>
    <div><strong>FPS:</strong> {metadata.fps:.3f}</div>
    <div><strong>Total frames:</strong> {metadata.total_frames}</div>
    <div><strong>Duration:</strong> {metadata.duration_seconds:.3f}s</div>
    <div><strong>Sample FPS:</strong> {metadata.sample_fps:.3f}</div>
    <div><strong>Resolution:</strong> {metadata.width} x {metadata.height}</div>
  </div>

  <h2>Segments</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Label</th><th>Start Time</th><th>End Time</th><th>Start Frame</th><th>End Frame</th><th>Conf.</th><th>Description</th>
      </tr>
    </thead>
    <tbody>
      {_segment_rows(segments)}
    </tbody>
  </table>

  <h2>Boundaries</h2>
  <table>
    <thead>
      <tr>
        <th>Sample</th><th>Frame</th><th>Time</th><th>Transition</th><th>Conf.</th><th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {_boundary_rows(boundaries)}
    </tbody>
  </table>

  <h2>Sampled Frames</h2>
  <div class="grid">
    {_thumbnail_strip(sampled_frames, boundary_set)}
  </div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    return right - left


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or _text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    if not lines:
        return textwrap.wrap(text, width=72) or [text]
    return lines


def _segment_color(index: int) -> tuple[int, int, int]:
    palette = [
        (44, 123, 182),
        (215, 95, 72),
        (46, 160, 120),
        (150, 110, 188),
        (220, 165, 64),
        (88, 145, 200),
        (205, 105, 145),
        (94, 155, 92),
    ]
    return palette[index % len(palette)]


def _segment_for_frame(segments: list[LabeledSegment], frame_id: int) -> LabeledSegment:
    if not segments:
        raise ValueError("segments must not be empty")
    current = segments[0]
    for segment in segments:
        if frame_id >= segment.start_frame_id:
            current = segment
        else:
            break
    return current


def _subtitle_lines(
    segment: LabeledSegment,
    frame_id: int,
    fps: float,
    total_segments: int,
) -> list[str]:
    return [segment.description]


def _draw_subtitle(
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    segment: LabeledSegment,
    frame_id: int,
    fps: float,
    total_segments: int,
) -> None:
    title_font = _load_font(22, bold=True)
    margin = 18
    box_width = frame_width - 2 * margin
    raw_lines = _subtitle_lines(segment, frame_id, fps, total_segments)
    wrapped: list[tuple[str, ImageFont.ImageFont]] = []
    for line_index, line in enumerate(raw_lines):
        font = title_font
        max_lines = 2
        for wrapped_line in _wrap_text(draw, line, font, box_width - 24)[:max_lines]:
            wrapped.append((wrapped_line, font))

    line_heights = [
        draw.textbbox((0, 0), text, font=font)[3] - draw.textbbox((0, 0), text, font=font)[1]
        for text, font in wrapped
    ]
    box_height = sum(line_heights) + 12 * (len(wrapped) - 1) + 22
    top = margin
    bottom = top + box_height
    draw.rounded_rectangle(
        [margin, top, frame_width - margin, bottom],
        radius=8,
        fill=(0, 0, 0, 178),
    )

    y = top + 11
    accent = _segment_color(segment.segment_id)
    draw.rounded_rectangle([margin + 10, top + 10, margin + 16, bottom - 10], radius=3, fill=accent)
    for (text, font), line_height in zip(wrapped, line_heights):
        draw.text((margin + 26, y), text, fill=(255, 255, 255, 255), font=font)
        y += line_height + 12


def _draw_action_timeline(
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    video_height: int,
    timeline_height: int,
    total_frames: int,
    segments: list[LabeledSegment],
    current_frame_id: int,
) -> None:
    label_font = _load_font(13, bold=True)
    small_font = _load_font(11)
    top = video_height
    draw.rectangle([0, top, frame_width, top + timeline_height], fill=(20, 22, 24, 255))

    margin = 18
    track_top = top + 24
    track_height = 34
    track_width = frame_width - 2 * margin
    safe_total = max(1, total_frames - 1)
    draw.text((margin, top + 6), "Action timeline", fill=(235, 235, 235, 255), font=label_font)

    for segment in segments:
        start_x = margin + int(track_width * max(0, segment.start_frame_id) / safe_total)
        end_x = margin + int(track_width * min(safe_total, segment.end_frame_id) / safe_total)
        end_x = max(end_x, start_x + 2)
        color = _segment_color(segment.segment_id)
        active = segment.start_frame_id <= current_frame_id <= segment.end_frame_id
        fill = (*color, 255)
        outline = (255, 255, 255, 255) if active else (70, 70, 70, 255)
        draw.rectangle([start_x, track_top, end_x, track_top + track_height], fill=fill, outline=outline, width=3 if active else 1)
        label = segment.action_label.replace("_", " ")
        label_width = _text_width(draw, label, small_font)
        if label_width + 8 < end_x - start_x:
            draw.text((start_x + 4, track_top + 10), label, fill=(255, 255, 255, 255), font=small_font)
        elif end_x - start_x >= 18:
            draw.text((start_x + 4, track_top + 10), str(segment.segment_id + 1), fill=(255, 255, 255, 255), font=small_font)

    cursor_x = margin + int(track_width * min(safe_total, max(0, current_frame_id)) / safe_total)
    draw.line([cursor_x, track_top - 8, cursor_x, track_top + track_height + 20], fill=(255, 255, 255, 255), width=2)
    draw.polygon(
        [(cursor_x, track_top - 10), (cursor_x - 6, track_top - 18), (cursor_x + 6, track_top - 18)],
        fill=(255, 255, 255, 255),
    )

    current = _segment_for_frame(segments, current_frame_id)
    footer = (
        f"frame {current_frame_id}/{total_frames - 1}  |  "
        f"current: {current.action_label}  |  "
        f"{current.start_time:.2f}s-{current.end_time:.2f}s"
    )
    draw.text((margin, top + timeline_height - 24), footer, fill=(230, 230, 230, 255), font=small_font)


def write_annotated_action_video(
    metadata: VideoMetadata,
    segments: list[LabeledSegment],
    output_path: Path,
    timeline_height: int = 112,
) -> Path:
    """Render a video with changing semantic subtitles and a bottom action timeline."""

    if not segments:
        raise ValueError("Cannot render annotated action video without segments.")

    video_path = Path(metadata.path)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing source video for action subtitle render: {video_path}")

    ensure_dir(output_path.parent)
    fps = metadata.fps
    width = metadata.width
    height = metadata.height
    total_frames = metadata.total_frames
    output_size = (width, height + timeline_height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open annotated video writer: {output_path}")

    frame_size = width * height * 3
    decode = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if decode.stdout is None:
        writer.release()
        raise RuntimeError("Failed to open ffmpeg stdout for annotated video render.")

    frame_id = 0
    sorted_segments = sorted(segments, key=lambda item: item.start_frame_id)
    while True:
        raw = decode.stdout.read(frame_size)
        if not raw:
            break
        if len(raw) != frame_size:
            LOGGER.warning("Ignoring partial decoded frame while rendering annotated video: %s", video_path)
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        canvas = np.zeros((height + timeline_height, width, 3), dtype=np.uint8)
        canvas[:height, :, :] = frame

        rgba = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA))
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        current = _segment_for_frame(sorted_segments, frame_id)
        _draw_subtitle(
            draw=draw,
            frame_width=width,
            segment=current,
            frame_id=frame_id,
            fps=fps,
            total_segments=len(sorted_segments),
        )
        _draw_action_timeline(
            draw=draw,
            frame_width=width,
            video_height=height,
            timeline_height=timeline_height,
            total_frames=total_frames,
            segments=sorted_segments,
            current_frame_id=frame_id,
        )
        composed = Image.alpha_composite(rgba, overlay).convert("RGB")
        writer.write(cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR))
        frame_id += 1

    stderr = decode.stderr.read().decode("utf-8", errors="replace") if decode.stderr is not None else ""
    return_code = decode.wait()
    writer.release()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg decode failed while rendering annotated video: {stderr.strip() or return_code}")
    if frame_id == 0:
        raise RuntimeError(f"No frames decoded while rendering annotated video: {video_path}")
    LOGGER.info("Wrote annotated action video: %s", output_path)
    return output_path
