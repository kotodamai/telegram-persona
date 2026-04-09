"""Build structured output files from analysis results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .schema import validate_output

SCHEMA_VERSION = "2.0.0"
GENERATOR_VERSION = "telegram-persona 0.2.0"
REQUIRED_FILES = [
    "identity_model.json",
    "style_rules.json",
    "contact_profiles.json",
    "group_profiles.json",
    "topic_graph.json",
    "response_policies.json",
    "behavior_stats.json",
    "memes.json",
    "core_prompt.md",
]
OPTIONAL_FILES = ["style_examples.json", "system_prompt.md"]


def build_identity_model(analysis: dict, user_name: str, stats: dict, meta: dict) -> dict:
    """Build the main identity_model.json."""
    identity = analysis.get("identity_model", {})
    memes = analysis.get("memes", {})

    return {
        "meta": {
            "user_name": user_name,
            "generated_at": datetime.now().isoformat(),
            "source_hash": meta.get("source_hash", ""),
            "tier1_model": meta.get("tier1_model", ""),
            "tier2_model": meta.get("tier2_model", ""),
            "prompt_version": meta.get("prompt_version", ""),
            "total_annotations": meta.get("total_annotations", 0),
            "total_summaries": meta.get("total_summaries", 0),
            "segment_counts": stats.get("category_counts", {}),
        },
        "identity": {
            "language_distribution": stats.get("lang_stats", {}),
            "avg_formality": stats.get("avg_formality", 3.0),
        },
        "big_five": identity.get("big_five", {}),
        "big_five_reasoning": identity.get("big_five_reasoning", ""),
        "core_style": identity.get("core_style", {}),
        "context_styles": identity.get("context_styles", {}),
        "vocabulary": identity.get("vocabulary", {}),
        "meme_behavior": {
            "timing": memes.get("meme_timing", {}),
            "grammar": memes.get("constructive_templates", memes.get("meme_grammar", [])),
            "adoption_style": memes.get("meme_adoption_style", {}),
            "known_vocabulary": memes.get("reaction_tokens", memes.get("known_meme_vocabulary", [])),
        },
        "meme_style": memes.get("meme_style_summary", ""),
    }


def save_outputs(analysis: dict, user_name: str, stats: dict, behavioral_stats: dict, meta: dict, output_dir: Path, template_dir: Path):
    """Save all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}

    # Build chat_id lookup from summaries (chat_name → telegram chat_id)
    chat_id_map: dict[str, int] = {}
    for s in analysis.get("summaries", []):
        cid = s.get("_chat_id", 0)
        cname = s.get("chat_name", "")
        if cid and cname:
            chat_id_map[cname] = cid

    # 1. identity_model.json (replaces persona.json)
    identity_model = build_identity_model(analysis, user_name, stats, meta)
    _write_json(output_dir / "identity_model.json", identity_model, file_hashes)

    # 2. style_rules.json (new)
    _write_json(output_dir / "style_rules.json", analysis.get("style_rules", {}), file_hashes)

    # 3. contact_profiles.json (replaces contacts.json)
    contacts_raw = analysis.get("contacts", {})
    contacts_rekeyed = _rekey_contacts(contacts_raw)
    _write_json(output_dir / "contact_profiles.json", contacts_rekeyed, file_hashes)

    # 4. group_profiles.json (replaces groups.json)
    groups_raw = analysis.get("groups", {})
    groups_rekeyed = _rekey_groups(groups_raw, chat_id_map)
    _write_json(output_dir / "group_profiles.json", groups_rekeyed, file_hashes)

    # 5. topic_graph.json (replaces knowledge.json)
    _write_json(output_dir / "topic_graph.json", analysis.get("topic_graph", {}), file_hashes)

    # 6. response_policies.json (new)
    policies = analysis.get("response_policies", {})
    _backfill_policy_ids(policies)
    _write_json(output_dir / "response_policies.json", policies, file_hashes)

    # 7. behavior_stats.json (replaces stats.json)
    combined_stats = {
        "annotation_stats": stats,
        "behavioral_stats": _enhance_behavior_stats(behavioral_stats),
    }
    _write_json(output_dir / "behavior_stats.json", combined_stats, file_hashes)

    # 8. style_examples.json — extract notable phrases from summaries
    style_examples = _extract_style_examples(analysis.get("summaries", []), chat_id_map)
    _write_json(output_dir / "style_examples.json", style_examples, file_hashes)

    # 9. memes.json
    _write_json(output_dir / "memes.json", analysis.get("memes", {}), file_hashes)

    # 10. system_prompt.md
    _render_system_prompt(identity_model, analysis, template_dir, output_dir, file_hashes,
                          contacts_rekeyed=contacts_rekeyed, groups_rekeyed=groups_rekeyed)

    # 11. manifest.json (always last — after all hashes are computed)
    manifest = _build_manifest(file_hashes)
    _write_json(output_dir / "manifest.json", manifest)


def _write_json(path: Path, data: dict | list, file_hashes: dict | None = None):
    validate_output(data, path.name)
    content = json.dumps(data, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    if file_hashes is not None:
        file_hashes[path.name] = hashlib.sha256(content.encode()).hexdigest()
    print(f"Saved: {path}")


def _write_text(path: Path, content: str, file_hashes: dict | None = None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    if file_hashes is not None:
        file_hashes[path.name] = hashlib.sha256(content.encode()).hexdigest()
    print(f"Saved: {path}")


def _build_manifest(file_hashes: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now().isoformat(),
        "required_files": REQUIRED_FILES,
        "optional_files": OPTIONAL_FILES,
        "consumer_targets": ["kotodamai>=0.2"],
        "files": file_hashes,
    }


def _rekey_contacts(raw: dict) -> dict:
    """Rekey contact profiles from display names to stable UIDs."""
    contacts = {}
    for display_key, profile in raw.items():
        uid = profile.get("uid", "")
        if not uid:
            # Try to extract uid from key like "Name (user123456)"
            m = re.search(r'\(user(\d+)\)', display_key)
            uid = f"user{m.group(1)}" if m else ""

        is_synthetic = False
        if not uid:
            uid = f"contact_{hashlib.sha256(display_key.encode()).hexdigest()[:8]}"
            is_synthetic = True

        # Parse display name and aliases from the old key
        # Keys look like: "Shelikhoo [aka Shell] (user24674211)" or "kjeff"
        clean_name = re.sub(r'\s*\(user\d+\)\s*$', '', display_key).strip()
        display_names = [clean_name]
        aliases = []

        # Extract all "aka" aliases from brackets
        for aka_match in re.finditer(r'\[aka\s+(.+?)\]', clean_name):
            aliases.append(aka_match.group(1).strip())
        if aliases:
            base_name = re.sub(r'\s*\[aka\s+.+?\]', '', clean_name).strip()
            display_names = [base_name] + aliases

        matching_hints = [n.lower() for n in display_names]

        contacts[uid] = {
            "display_names": display_names,
            "aliases": [display_key],  # preserve original key as alias
            "telegram_uid": None if is_synthetic else uid,
            "matching_hints": matching_hints,
            **{k: v for k, v in profile.items() if k != "uid"},
        }

    return {"contacts": contacts}


def _rekey_groups(raw: dict, chat_id_map: dict[str, int] | None = None) -> dict:
    """Rekey group profiles from display names to stable IDs."""
    groups = {}
    for group_name, profile in raw.items():
        real_id = (chat_id_map or {}).get(group_name, 0)
        if real_id:
            group_id = f"group_{real_id}"
        else:
            group_id = f"group_{hashlib.sha256(group_name.encode()).hexdigest()[:8]}"
        groups[group_id] = {
            "canonical_name": group_name,
            "aliases": [],
            "telegram_chat_id": real_id or None,
            **profile,
        }
    return {"groups": groups}


def _enhance_behavior_stats(raw: dict) -> dict:
    """Add derived stats for human simulation."""
    enhanced = dict(raw)

    # Estimate read delay from reply latency (reading is faster than full reply cycle)
    latency = raw.get("reply_latency_seconds", {})
    if latency.get("median", 0) > 0:
        enhanced["read_delay_seconds"] = {
            "mean": round(min(latency.get("mean", 30) * 0.15, 120), 1),
            "median": round(min(latency.get("median", 15) * 0.15, 60), 1),
            "p90": round(min(latency.get("p90", 60) * 0.15, 300), 1),
        }

    # Estimate typing speed from message length and latency
    msg_len = raw.get("message_length", {})
    if msg_len.get("median", 0) > 0 and latency.get("median", 0) > 0:
        # Rough estimate: median chars / median latency, converted to WPM
        # Assume ~5 chars per word, subtract read delay estimate
        read_delay = enhanced.get("read_delay_seconds", {}).get("median", 5)
        typing_seconds = max(latency.get("median", 30) - read_delay, 1)
        chars_per_second = msg_len.get("median", 10) / typing_seconds
        wpm = round(chars_per_second * 60 / 5, 1)
        # Clamp to reasonable range
        enhanced["typing_speed_wpm"] = max(10, min(wpm, 120))

    # Burst gap from burst lengths
    bursts = raw.get("burst_lengths", {})
    if bursts.get("median", 0) > 1:
        # Estimate gap between consecutive messages in a burst
        enhanced["burst_gap_seconds"] = {
            "mean": 2.0,
            "median": 1.5,
            "p90": 5.0,
        }

    # Online hours confidence from active_hours distribution
    active_hours = raw.get("active_hours", {})
    if active_hours:
        total = sum(active_hours.values())
        if total > 0:
            hour_fractions = {int(h): c / total for h, c in active_hours.items()}
            # Confidence = how concentrated the distribution is (1 - entropy/max_entropy)
            entropy = -sum(f * math.log2(f) for f in hour_fractions.values() if f > 0)
            max_entropy = math.log2(24)
            enhanced["online_hours_confidence"] = round(1 - entropy / max_entropy, 3)

    # Per-context breakdowns
    by_chat_type = {}
    msg_len_by_cat = raw.get("message_length_by_category", {})
    active_hours_by_cat = raw.get("active_hours_by_category", {})
    init_by_cat = raw.get("initiation_ratio_by_category", {})

    for cat in set(list(msg_len_by_cat.keys()) + list(active_hours_by_cat.keys()) + list(init_by_cat.keys())):
        by_chat_type[cat] = {}
        if cat in msg_len_by_cat:
            by_chat_type[cat]["message_length"] = msg_len_by_cat[cat]
        if cat in active_hours_by_cat:
            by_chat_type[cat]["active_hours"] = active_hours_by_cat[cat]
        if cat in init_by_cat:
            by_chat_type[cat]["initiation_ratio"] = init_by_cat[cat]

    if by_chat_type:
        enhanced["by_chat_type"] = by_chat_type

    return enhanced


def _backfill_policy_ids(policies: dict) -> None:
    """Ensure all policy items have an id field."""
    for list_key, prefix in [
        ("decision_policies", "dp"),
        ("generation_policies", "gp"),
        ("boundaries", "bd"),
        ("triggers", "tr"),
    ]:
        items = policies.get(list_key, [])
        for i, item in enumerate(items, 1):
            if not item.get("id"):
                item["id"] = f"{prefix}_{i:03d}"


def _extract_style_examples(summaries: list[dict], chat_id_map: dict[str, int] | None = None) -> dict:
    """Extract representative phrases from summaries, keyed by stable chat ID."""
    examples = {}
    for s in summaries:
        chat_name = s.get("chat_name", "unknown")
        phrases = s.get("notable_phrases", [])
        if not phrases:
            continue
        real_id = (chat_id_map or {}).get(chat_name, 0)
        if real_id:
            key = str(real_id)
        else:
            key = hashlib.sha256(chat_name.encode()).hexdigest()[:8]
        examples[key] = {
            "chat_name": chat_name,
            "phrases": phrases,
        }
    return examples


def _render_system_prompt(identity_model: dict, analysis: dict, template_dir: Path, output_dir: Path,
                          file_hashes: dict | None = None, *,
                          contacts_rekeyed: dict | None = None,
                          groups_rekeyed: dict | None = None):
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    contacts = contacts_rekeyed or analysis.get("contacts", {})
    groups = groups_rekeyed or analysis.get("groups", {})

    # Full prompt (all data, for reference)
    template = env.get_template("system_prompt.md.j2")
    rendered = template.render(
        model=identity_model,
        style_rules=analysis.get("style_rules", {}),
        contacts=contacts,
        groups=groups,
        topic_graph=analysis.get("topic_graph", {}),
        response_policies=analysis.get("response_policies", {}),
    )
    _write_text(output_dir / "system_prompt.md", rendered, file_hashes)

    # Core prompt (compact, for agent system prompt — no contacts/groups/topics)
    core_template = env.get_template("core_prompt.md.j2")
    core_rendered = core_template.render(
        model=identity_model,
        style_rules=analysis.get("style_rules", {}),
    )
    _write_text(output_dir / "core_prompt.md", core_rendered, file_hashes)
