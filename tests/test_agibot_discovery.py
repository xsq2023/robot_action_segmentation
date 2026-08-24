from robot_tas.cli.evaluate_agibot_task import find_episode_dirs


def test_find_episode_dirs_supports_flat_observation_layout(tmp_path) -> None:
    (tmp_path / "101" / "videos").mkdir(parents=True)
    (tmp_path / "102" / "depth").mkdir(parents=True)

    assert [path.name for path in find_episode_dirs(tmp_path)] == ["101"]


def test_find_episode_dirs_supports_range_observation_layout(tmp_path) -> None:
    (tmp_path / "100-200" / "101" / "videos").mkdir(parents=True)
    (tmp_path / "100-200" / "102" / "depth").mkdir(parents=True)

    assert [path.name for path in find_episode_dirs(tmp_path)] == ["101"]
