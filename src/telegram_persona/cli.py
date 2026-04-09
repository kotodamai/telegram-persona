"""CLI entry point for tg-persona."""

from __future__ import annotations

import argparse
import asyncio
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .analyzer import Analyzer
from .chunker import chunk_export, classify_chat_type
from .config import Config
from .parser import compute_behavioral_stats, parse_export
from .persona import save_outputs, build_identity_model, _render_system_prompt
from .prompts import prompt_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tg-persona",
        description="Extract personality and relationship profiles from Telegram export data",
    )
    parser.add_argument("export_path", type=Path, nargs="?", help="Path to Telegram result.json")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="Output directory")

    # Model configuration
    parser.add_argument("--tier1-model", help="Model for Tier-1 annotation")
    parser.add_argument("--tier2-model", help="Model for Tier-2 distillation")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--concurrency", type=int, help="Concurrent API calls for Tier-1")
    parser.add_argument("--api-timeout", type=float, help="Per-request API timeout in seconds (default: 180)")
    parser.add_argument("--api-max-retries", type=int, help="Max retries per API call (default: 3)")

    # Data filtering
    parser.add_argument("--since", type=str, help="Only include messages after this date (YYYY-MM-DD)")
    parser.add_argument("--until", type=str, help="Only include messages before this date (YYYY-MM-DD)")
    parser.add_argument("--include-chat", action="append", dest="include_chats", help="Only process these chats (repeatable)")
    parser.add_argument("--exclude-chat", action="append", dest="exclude_chats", help="Skip these chats (repeatable)")
    parser.add_argument("--min-messages", type=int, default=0, help="Skip chats with fewer user messages")

    # Run control
    parser.add_argument("--skip-bots", action="store_true", default=True)
    parser.add_argument("--no-skip-bots", action="store_false", dest="skip_bots")
    parser.add_argument("--only-tier1", action="store_true", help="Only run Tier-1 annotation")
    parser.add_argument("--only-tier2", action="store_true", help="Only run Tier-1.5 + Tier-2 (reuse cached Tier-1)")
    parser.add_argument("--render-only", action="store_true", help="Re-render prompts from existing JSON outputs (no LLM calls)")
    parser.add_argument("--estimate-cost", action="store_true", help="Dry run: estimate token usage and cost")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached results before running")
    parser.add_argument("--sample", type=int, help="Only process N random chunks (for testing)")

    parser.add_argument("--env-file", type=Path, help="Path to .env file")
    return parser.parse_args()


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


def _estimate_cost(chunks, config, user_id):
    """Estimate token usage and cost for a run."""
    total_chars = sum(len(c.render(user_id)) for c in chunks)
    est_tokens = total_chars / 3  # rough estimate

    print(f"\n--- Cost Estimate ---")
    print(f"Chunks: {len(chunks)}")
    print(f"Total input chars: {total_chars:,.0f}")
    print(f"Estimated input tokens: {est_tokens:,.0f}")
    print(f"Tier-1 calls ({config.tier1_model}): {len(chunks)}")

    # Tier-1.5: one call per unique chat
    chats = set(c.chat_name for c in chunks)
    print(f"Tier-1.5 calls ({config.tier1_model}): ~{len(chats)}")

    # Tier-2: identity + style_rules + contacts + groups + topic_graph + response_policies + memes
    print(f"Tier-2 calls ({config.tier2_model}): ~7-20 (depends on contacts/groups)")
    print(f"\nTotal estimated API calls: ~{len(chunks) + len(chats) + 15}")
    print("---")


async def async_main():
    args = parse_args()

    # Load config
    config = Config.from_env(args.env_file)
    config.output_dir = args.output

    # CLI overrides
    if args.tier1_model:
        config.tier1_model = args.tier1_model
    if args.tier2_model:
        config.tier2_model = args.tier2_model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.concurrency:
        config.concurrency = args.concurrency
    if args.api_timeout:
        config.api_timeout = args.api_timeout
    if args.api_max_retries is not None:
        config.api_max_retries = args.api_max_retries

    # Clear cache
    if args.clear_cache:
        cache_dir = config.output_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")

    if not args.estimate_cost and not args.render_only and not config.api_key:
        print("Error: API key required. Set OPENAI_API_KEY in .env or pass --api-key", file=sys.stderr)
        sys.exit(1)

    # Render-only mode: re-render prompts from existing JSON, no parsing/LLM needed
    if args.render_only:
        import json as _json
        output_dir = config.output_dir
        template_dir = Path(__file__).parent.parent.parent / "templates"

        identity_path = output_dir / "identity_model.json"
        if not identity_path.exists():
            print(f"Error: {identity_path} not found. Run full analysis first.", file=sys.stderr)
            sys.exit(1)

        with open(identity_path) as f:
            identity_model = _json.load(f)

        analysis = {}
        file_to_key = {
            "style_rules.json": "style_rules",
            "contact_profiles.json": "contacts",
            "group_profiles.json": "groups",
            "topic_graph.json": "topic_graph",
            "response_policies.json": "response_policies",
        }
        for filename, key in file_to_key.items():
            p = output_dir / filename
            if p.exists():
                with open(p) as f:
                    analysis[key] = _json.load(f)

        _render_system_prompt(identity_model, analysis, template_dir, output_dir)
        print("\nDone!")
        return

    # export_path is required for all other modes
    if not args.export_path:
        print("Error: export_path is required (unless using --render-only)", file=sys.stderr)
        sys.exit(1)

    # Parse
    print(f"Parsing {args.export_path}...")
    export = parse_export(
        args.export_path,
        skip_bots=args.skip_bots,
        since=_parse_date(args.since),
        until=_parse_date(args.until),
        include_chats=set(args.include_chats) if args.include_chats else None,
        exclude_chats=set(args.exclude_chats) if args.exclude_chats else None,
        min_messages=args.min_messages,
    )
    print(f"  User: {export.user_name} ({export.user_id})")
    print(f"  Chats: {len(export.chats)}")
    print(f"  My messages: {len(export.user_messages())}")
    print(f"  Identity map: {len(export.identity_map.all_ids())} unique IDs")
    print(f"  Source hash: {export.source_hash}")

    # Behavioral stats
    behavioral_stats = compute_behavioral_stats(export)

    # Chunk
    print("Chunking conversations...")
    chunks = chunk_export(export)
    print(f"  Chunks: {len(chunks)}")

    # Sample
    if args.sample and args.sample < len(chunks):
        chunks = random.sample(chunks, args.sample)
        print(f"  Sampled: {len(chunks)} chunks")

    # Estimate cost
    if args.estimate_cost:
        _estimate_cost(chunks, config, export.user_id)
        return

    # Analyze
    analyzer = Analyzer(
        config,
        export.user_id,
        export.user_name,
        export.identity_map,
        export.source_hash,
    )

    # Build private chat → other person UID mapping from message data
    private_chat_uids = {}
    for chat in export.chats:
        if classify_chat_type(chat.chat_type) == "personal":
            other_ids = {m.from_id for m in chat.messages if m.from_id and m.from_id != export.user_id and not m.is_service}
            if len(other_ids) == 1:
                private_chat_uids[chat.name] = other_ids.pop()
    analyzer.set_private_chat_uids(private_chat_uids)

    if args.only_tier1:
        annotations = await analyzer.tier1_annotate(chunks)
        print(f"\nTier-1 complete: {len(annotations)} annotations cached.")
        return

    if args.only_tier2:
        # Reuse cached Tier-1, run Tier-1.5 + Tier-2
        print("--only-tier2: loading Tier-1 from cache...")
        annotations = await analyzer.tier1_annotate(chunks)
        if not annotations:
            print("Error: no cached Tier-1 annotations found. Run without --only-tier2 first.", file=sys.stderr)
            sys.exit(1)
        summaries = await analyzer.tier1_5_summarize(annotations)
        if not summaries:
            print("Error: no summaries produced from Tier-1.5.", file=sys.stderr)
            sys.exit(1)
        analysis = await analyzer.run_tier2(summaries, annotations, behavioral_stats)
    else:
        analysis = await analyzer.run(chunks, behavioral_stats)

    # Build output
    meta = {
        "source_hash": export.source_hash,
        "tier1_model": config.tier1_model,
        "tier2_model": config.tier2_model,
        "prompt_version": prompt_version(),
        "total_annotations": len(analysis.get("annotations", [])),
        "total_summaries": len(analysis.get("summaries", [])),
    }

    template_dir = Path(__file__).parent.parent.parent / "templates"
    save_outputs(
        analysis,
        export.user_name,
        analysis["stats"],
        behavioral_stats,
        meta,
        config.output_dir,
        template_dir,
    )

    print("\nDone!")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
