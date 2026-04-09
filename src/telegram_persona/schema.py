"""Formal vocabulary definitions and validation for persona output."""

from __future__ import annotations

import sys
from typing import Any

# ============================================================
# Allowed enum values
# ============================================================

CHAT_TYPES = {"private", "group", "channel"}

CONTEXT_STYLES = {
    "private_casual", "private_serious",
    "group_casual", "group_technical",
    "channel",
}

RELATIONSHIP_TYPES = {
    "close_friend", "friend", "acquaintance", "colleague",
    "family", "mentor", "mentee", "community_peer",
}

ENGAGEMENT_BIAS = {"avoid", "reactive", "normal", "proactive"}

DECISION = {"reply", "skip", "uncertain", "defer"}

PRIORITY = {"high", "medium", "low"}

STAKES = {"low", "medium", "high"}

EFFECT = {
    "short_reply", "multi_burst", "diagnosis_first", "minimal_punctuation",
    "heavy_emoji", "no_emoji", "code_snippet", "link_share",
    "sticker_only", "language_switch", "formality_increase", "formality_decrease",
}

TRIGGER_TAGS = {
    "any_message", "question", "technical_question", "help_request",
    "greeting", "humor", "disagreement", "success", "failure",
    "media_share", "link_share",
}

SOCIAL_FUNCTION = {
    "humor", "deflection", "bonding", "complaint", "celebration",
    "sarcasm", "self_deprecation", "surprise", "agreement",
}

GROUP_ROLE = {
    "core_member", "active_contributor", "helper", "entertainer",
    "lurker", "leader", "moderator",
}

ACTIVITY_LEVEL = {"high", "medium", "low"}

RESPONSE_SHAPE = {
    "short_diagnostic", "detailed_explanation", "multi_message",
    "link_share", "sticker_only", "acknowledgment", "question",
    "code_snippet", "opinion", "redirect",
}

REPLY_LIKELIHOOD = {"high", "medium", "low", "none"}


# ============================================================
# Validation
# ============================================================

_FIELD_VOCAB: dict[str, tuple[str, set[str]]] = {
    # (json_path_hint, allowed_values)
    "chat_types": ("style_rules/response_policies", CHAT_TYPES),
    "applies_in": ("style_rules", CONTEXT_STYLES),
    "not_applies_in": ("style_rules", CONTEXT_STYLES),
    "applies_to_chat_types": ("response_policies", CHAT_TYPES),
    "priority": ("style_rules", PRIORITY),
    "effect": ("style_rules", EFFECT),
    "trigger_tags": ("style_rules", TRIGGER_TAGS),
    "decision": ("response_policies", DECISION),
    "engagement_bias": ("group_profiles", ENGAGEMENT_BIAS),
    "activity_level": ("group_profiles", ACTIVITY_LEVEL),
    "social_function": ("memes", SOCIAL_FUNCTION),
    "stakes": ("memes/style_rules", STAKES),
    "reply_likelihood": ("topic_graph", REPLY_LIKELIHOOD),
    "response_shape": ("response_policies", RESPONSE_SHAPE),
    "relationship_type": ("contact_profiles", RELATIONSHIP_TYPES),
}

# Fields that should only be validated in specific files.
# Key = field name, value = set of file names where validation applies.
_FILE_SCOPED_FIELDS: dict[str, set[str]] = {
    "activity_level": {"group_profiles.json"},
}


def validate_enums(data: Any, file_name: str, path: str = "") -> list[str]:
    """Walk a data structure and warn on unknown enum values.

    Returns a list of warning strings. Does not raise.
    """
    warnings: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else key
            if key in _FIELD_VOCAB:
                # Skip file-scoped fields when not in their target file
                scoped_files = _FILE_SCOPED_FIELDS.get(key)
                if scoped_files is not None and file_name not in scoped_files:
                    pass
                else:
                    _, allowed = _FIELD_VOCAB[key]
                    _check_value(value, allowed, child_path, file_name, warnings)
            warnings.extend(validate_enums(value, file_name, child_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            warnings.extend(validate_enums(item, file_name, f"{path}[{i}]"))

    return warnings


def _check_value(value: Any, allowed: set[str], path: str, file_name: str, warnings: list[str]):
    """Check a single value or list of values against allowed set."""
    if isinstance(value, str):
        if value not in allowed and value != "all":
            warnings.append(f"{file_name}: {path} = {value!r} not in {sorted(allowed)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item not in allowed and item != "all":
                warnings.append(f"{file_name}: {path} contains {item!r} not in {sorted(allowed)}")


def validate_output(data: dict | list, file_name: str) -> None:
    """Validate a single output file's enum values. Prints warnings to stderr."""
    warnings = validate_enums(data, file_name)
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)
