from robot_tas.action_set import DEFAULT_ACTION_SET, format_action_set_contract, parse_action_set


def test_default_action_set_covers_downloaded_task_skills() -> None:
    expected = {"pick", "place", "close", "pull", "grasp", "lift", "lower", "move", "release"}
    assert expected.issubset(set(DEFAULT_ACTION_SET))


def test_format_action_set_contract_lists_allowed_labels() -> None:
    contract = format_action_set_contract(["pick", "place", "pull"])
    assert "Allowed action labels: pick, place, pull." in contract
    assert "transition_type must be before_action_to_after_action" in contract


def test_parse_action_set_normalizes_and_deduplicates() -> None:
    assert parse_action_set("Pick,Place,pick,Approach") == ["pick", "place", "reach_for"]
