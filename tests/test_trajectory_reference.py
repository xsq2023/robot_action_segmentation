from pathlib import Path

from robot_tas.trajectory_reference import build_trajectory_reference, trajectory_reference_lines


def test_build_trajectory_reference_finds_active_effector_and_candidates(tmp_path: Path) -> None:
    csv_path = tmp_path / "timeseries.csv"
    csv_path.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "action.effector.position[0]",
                        "action.effector.position[1]",
                        "state.effector.position[0]",
                        "state.effector.position[1]",
                        "state.end.position[0]",
                        "state.end.position[1]",
                        "state.end.position[2]",
                        "state.end.position[3]",
                        "state.end.position[4]",
                        "state.end.position[5]",
                        "state.end.orientation[0]",
                        "state.end.orientation[1]",
                        "state.end.orientation[2]",
                        "state.end.orientation[3]",
                        "state.end.orientation[4]",
                        "state.end.orientation[5]",
                        "state.end.orientation[6]",
                        "state.end.orientation[7]",
                        "state.joint.position[0]",
                        "state.joint.position[1]",
                        "state.joint.position[2]",
                        "state.joint.position[3]",
                        "state.joint.position[4]",
                        "state.joint.position[5]",
                        "state.joint.position[6]",
                        "state.joint.position[7]",
                        "state.joint.position[8]",
                        "state.joint.position[9]",
                        "state.joint.position[10]",
                        "state.joint.position[11]",
                        "state.joint.position[12]",
                        "state.joint.position[13]",
                        "timestamp",
                    ]
                ),
                "0,0,35,35,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,100",
                "0,1,35,35,0,0,0,1.1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1.1,1,1,1,1,1,1,101",
                "0,1,35,75,0,0,0,1.2,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1.2,1,1,1,1,1,1,102",
                "0,0,35,70,0,0,0,1.3,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1.3,1,1,1,1,1,1,103",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reference = build_trajectory_reference(
        csv_path,
        frame_ids=[0, 1, 2, 3],
        fps=30.0,
        candidate_count=4,
        min_gap_frames=1,
    )

    assert reference["active_signals"]["effector_index"] == 1
    assert reference["selected_candidates"]
    assert any(candidate["event_type"] == "gripper_command_close" for candidate in reference["selected_candidates"])

    lines = trajectory_reference_lines(reference)
    assert any("trajectory_candidate" in line for line in lines)
