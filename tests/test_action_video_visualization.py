from pathlib import Path

import cv2
import numpy as np

from robot_tas.schemas import LabeledSegment, VideoMetadata
from robot_tas.visualization import write_annotated_action_video


def test_write_annotated_action_video_adds_timeline_band(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    width, height = 64, 48
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(6):
        frame = np.full((height, width, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    metadata = VideoMetadata(
        path=str(video_path),
        fps=5.0,
        total_frames=6,
        duration_seconds=1.2,
        width=width,
        height=height,
        sample_fps=5.0,
    )
    segments = [
        LabeledSegment(
            segment_id=0,
            start_sample_index=0,
            end_sample_index=2,
            start_frame_id=0,
            end_frame_id=2,
            start_time=0.0,
            end_time=0.4,
            action_label="pick",
            description="The gripper picks the object.",
            primary_object="object",
            secondary_objects=[],
            actor_motion="grasp",
            contact_state="grasped",
            object_motion="moving_with_gripper",
            goal="pick the object",
            confidence=0.9,
        ),
        LabeledSegment(
            segment_id=1,
            start_sample_index=3,
            end_sample_index=5,
            start_frame_id=3,
            end_frame_id=5,
            start_time=0.6,
            end_time=1.0,
            action_label="place",
            description="The gripper places the object.",
            primary_object="object",
            secondary_objects=[],
            actor_motion="place",
            contact_state="released",
            object_motion="placed",
            goal="place the object",
            confidence=0.9,
        ),
    ]

    output_path = write_annotated_action_video(
        metadata=metadata,
        segments=segments,
        output_path=tmp_path / "annotated.mp4",
        timeline_height=40,
    )

    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height + 40
    cap.release()
