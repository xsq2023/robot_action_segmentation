from __future__ import annotations

from collections.abc import Iterable

from robot_tas.normalization import normalize_label


DEFAULT_ACTION_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("reach_for", "move the gripper or end-effector toward a target object or interaction site"),
    ("grasp", "close or position the gripper to establish control of an object"),
    ("pick", "remove or acquire an object from a source area"),
    ("lift", "raise a contacted or grasped object upward"),
    ("carry", "transport a grasped object while maintaining control of it"),
    ("transport", "transport a grasped object while maintaining control of it"),
    ("move", "transport or reposition the gripper, object, tool, handle, or articulated part"),
    ("place", "put a held object at a target location or into a container"),
    ("lower", "move a held object or end-effector downward toward a surface or target"),
    ("release", "open or disengage the gripper so the object is no longer controlled"),
    ("pull", "draw a handle, drawer, door, object, or part toward the robot"),
    ("push", "drive a button, door, drawer, object, or part away from the robot"),
    ("open", "increase the open state of a door, drawer, lid, container, or gripper"),
    ("close", "decrease the open state of a door, drawer, lid, container, or gripper"),
    ("rotate", "turn a knob, cap, tool, object, or wrist orientation"),
    ("press", "push a button, switch, surface, or object with a short forceful contact"),
    ("insert", "put an object into a slot, hole, container, fixture, or receptacle"),
    ("remove", "take an object out of a slot, hole, container, fixture, or receptacle"),
    ("retract", "move the gripper or arm away after an interaction"),
    ("hold", "maintain control or contact without a clear state change"),
    ("wait", "remain mostly still or observe without a manipulation action"),
)

DEFAULT_ACTION_SET: tuple[str, ...] = tuple(label for label, _definition in DEFAULT_ACTION_DEFINITIONS)


def normalize_action_set(actions: Iterable[str]) -> list[str]:
    """Normalize action labels while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for action in actions:
        label = normalize_label(action)
        if not label or label in seen:
            continue
        normalized.append(label)
        seen.add(label)
    return normalized


def parse_action_set(raw_actions: str | None) -> list[str]:
    if not raw_actions:
        return list(DEFAULT_ACTION_SET)
    actions = normalize_action_set(action for action in raw_actions.split(",") if action.strip())
    if not actions:
        raise ValueError("Action set must contain at least one label.")
    return actions


def merge_action_sets(primary: Iterable[str], extra: Iterable[str]) -> list[str]:
    return normalize_action_set([*primary, *extra])


def format_action_set_contract(action_set: Iterable[str]) -> str:
    actions = normalize_action_set(action_set)
    definitions = {label: definition for label, definition in DEFAULT_ACTION_DEFINITIONS}
    lines = [
        "Action-set contract:",
        f"- Allowed action labels: {', '.join(actions)}.",
        "- Labels not in this set are invalid; choose the closest visible action from this set.",
        "- Use exactly these snake_case labels in action_label, before_state.action, and after_state.action.",
        "- transition_type must be before_action_to_after_action using labels from this set.",
        "- Default label meanings:",
    ]
    for action in actions:
        definition = definitions.get(action, f"perform the visible {action.replace('_', ' ')} action")
        lines.append(f"  - {action}: {definition}.")
    return "\n".join(lines)
