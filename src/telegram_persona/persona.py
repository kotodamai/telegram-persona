"""Build structured output files from analysis results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


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
            "grammar": memes.get("meme_grammar", []),
            "adoption_style": memes.get("meme_adoption_style", {}),
            "known_vocabulary": memes.get("known_meme_vocabulary", []),
        },
        "meme_style": memes.get("meme_style_summary", ""),
    }


def save_outputs(analysis: dict, user_name: str, stats: dict, behavioral_stats: dict, meta: dict, output_dir: Path, template_dir: Path):
    """Save all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. identity_model.json (replaces persona.json)
    identity_model = build_identity_model(analysis, user_name, stats, meta)
    _write_json(output_dir / "identity_model.json", identity_model)

    # 2. style_rules.json (new)
    _write_json(output_dir / "style_rules.json", analysis.get("style_rules", {}))

    # 3. contact_profiles.json (replaces contacts.json)
    _write_json(output_dir / "contact_profiles.json", analysis.get("contacts", {}))

    # 4. group_profiles.json (replaces groups.json)
    _write_json(output_dir / "group_profiles.json", analysis.get("groups", {}))

    # 5. topic_graph.json (replaces knowledge.json)
    _write_json(output_dir / "topic_graph.json", analysis.get("topic_graph", {}))

    # 6. response_policies.json (new)
    _write_json(output_dir / "response_policies.json", analysis.get("response_policies", {}))

    # 7. behavior_stats.json (replaces stats.json)
    combined_stats = {
        "annotation_stats": stats,
        "behavioral_stats": behavioral_stats,
    }
    _write_json(output_dir / "behavior_stats.json", combined_stats)

    # 8. style_examples.json — extract notable phrases from summaries
    style_examples = _extract_style_examples(analysis.get("summaries", []))
    _write_json(output_dir / "style_examples.json", style_examples)

    # 9. memes.json
    _write_json(output_dir / "memes.json", analysis.get("memes", {}))

    # 10. system_prompt.md
    _render_system_prompt(identity_model, analysis, template_dir, output_dir)


def _write_json(path: Path, data: dict | list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def _extract_style_examples(summaries: list[dict]) -> dict:
    """Extract representative phrases from summaries, grouped by chat."""
    examples = {}
    for s in summaries:
        chat_name = s.get("chat_name", "unknown")
        phrases = s.get("notable_phrases", [])
        if phrases:
            examples[chat_name] = phrases
    return examples


def _render_system_prompt(identity_model: dict, analysis: dict, template_dir: Path, output_dir: Path):
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Full prompt (all data, for reference)
    template = env.get_template("system_prompt.md.j2")
    rendered = template.render(
        model=identity_model,
        style_rules=analysis.get("style_rules", {}),
        contacts=analysis.get("contacts", {}),
        groups=analysis.get("groups", {}),
        topic_graph=analysis.get("topic_graph", {}),
        response_policies=analysis.get("response_policies", {}),
    )
    path = output_dir / "system_prompt.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(rendered)
    print(f"Saved: {path}")

    # Core prompt (compact, for agent system prompt — no contacts/groups/topics)
    core_template = env.get_template("core_prompt.md.j2")
    core_rendered = core_template.render(
        model=identity_model,
        style_rules=analysis.get("style_rules", {}),
    )
    core_path = output_dir / "core_prompt.md"
    with open(core_path, "w", encoding="utf-8") as f:
        f.write(core_rendered)
    print(f"Saved: {core_path}")
