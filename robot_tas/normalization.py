from __future__ import annotations

import re


LABEL_RULES = {
    "move_toward": "reach_for",
    "approach": "reach_for",
    "approach_workspace": "reach_for",
    "reach_for_object": "reach_for",
    "grab": "grasp",
    "grip": "grasp",
    "move_while_holding": "carry",
    "put_down": "place",
    "let_go": "release",
}


def to_snake_case(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_label(label: str) -> str:
    snake = to_snake_case(label)
    return LABEL_RULES.get(snake, snake)


def normalize_transition(before_action: str, after_action: str, explicit: str | None = None) -> str:
    if explicit:
        return to_snake_case(explicit)
    return f"{normalize_label(before_action)}_to_{normalize_label(after_action)}"


def semantics_compatible(
    before_a: str,
    after_a: str,
    before_b: str,
    after_b: str,
    transition_a: str,
    transition_b: str,
) -> bool:
    normalized_pair_a = (normalize_label(before_a), normalize_label(after_a))
    normalized_pair_b = (normalize_label(before_b), normalize_label(after_b))
    return normalized_pair_a == normalized_pair_b or to_snake_case(transition_a) == to_snake_case(transition_b)
