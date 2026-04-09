"""LLM analysis engine with hierarchical summarization, stable identity, and caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections import Counter, defaultdict
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

from .chunker import ConversationChunk
from .config import Config
from .parser import IdentityMap
from .prompts import (
    TIER1_SYSTEM,
    TIER1_PERSONAL_CHAT,
    TIER1_GROUP_CHAT,
    TIER1_CHANNEL,
    TIER1_5_SYSTEM,
    TIER1_5_SUMMARY,
    TIER2_IDENTITY_SYSTEM,
    TIER2_IDENTITY_USER,
    TIER2_STYLE_RULES_SYSTEM,
    TIER2_STYLE_RULES_USER,
    TIER2_RELATIONSHIP_SYSTEM,
    TIER2_PRIVATE_RELATIONSHIP_USER,
    TIER2_GROUP_RELATIONSHIP_USER,
    TIER2_GROUP_PROFILE_SYSTEM,
    TIER2_GROUP_PROFILE_USER,
    TIER2_TOPIC_GRAPH_SYSTEM,
    TIER2_TOPIC_GRAPH_USER,
    TIER2_RESPONSE_POLICIES_SYSTEM,
    TIER2_RESPONSE_POLICIES_USER,
    TIER2_MEME_SYSTEM,
    TIER2_MEME_USER,
    prompt_version,
)

# Minimum interaction records to analyze a group-based relationship
MIN_GROUP_INTERACTIONS = 3


def _cache_key(*parts: str) -> str:
    """Generate a cache key from multiple parts (content, model, prompt version, etc.)."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _select_tier1_prompt(chunk: ConversationChunk) -> str:
    if chunk.chat_category == "personal":
        return TIER1_PERSONAL_CHAT
    elif chunk.chat_category == "group":
        return TIER1_GROUP_CHAT
    elif chunk.chat_category == "channel":
        return TIER1_CHANNEL
    return TIER1_GROUP_CHAT


class Analyzer:
    def __init__(
        self,
        config: Config,
        user_id: str,
        user_name: str,
        identity_map: IdentityMap,
        source_hash: str = "",
    ):
        self.config = config
        self.user_id = user_id
        self.user_name = user_name
        self.identity_map = identity_map
        self.source_hash = source_hash
        self.prompt_ver = prompt_version()
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.api_timeout,
            max_retries=config.api_max_retries,
        )
        self.cache_dir = config.output_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Mapping of private chat name → other person's uid
        # Built externally via set_private_chat_uids()
        self._private_chat_uids: dict[str, str] = {}

    def set_private_chat_uids(self, mapping: dict[str, str]):
        """Set chat_name → other_person_uid mapping for private chats."""
        self._private_chat_uids = mapping

    def _load_cache(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: dict):
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def _call_llm(self, model: str, system: str, user: str) -> dict:
        resp = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = resp.choices[0].message.content
        return json.loads(content)

    # ============================================================
    # Tier-1: Per-chunk annotation
    # ============================================================

    async def _annotate_chunk(
        self, chunk: ConversationChunk, semaphore: asyncio.Semaphore
    ) -> tuple[ConversationChunk, dict | None]:
        chunk_text = chunk.render(self.user_id)
        key = _cache_key(chunk_text, self.config.tier1_model, self.prompt_ver, self.source_hash, "t1")
        cached = self._load_cache(key)
        if cached:
            return chunk, cached

        prompt_template = _select_tier1_prompt(chunk)
        prompt = prompt_template.format(chunk_text=chunk_text)

        async with semaphore:
            try:
                result = await self._call_llm(
                    self.config.tier1_model, TIER1_SYSTEM, prompt
                )
                result["_chat_name"] = chunk.chat_name
                result["_chat_type"] = chunk.chat_type
                result["_chat_category"] = chunk.chat_category
                self._save_cache(key, result)
                return chunk, result
            except Exception as e:
                print(f"\nError annotating chunk from {chunk.chat_name}: {e}", file=sys.stderr)
                return chunk, None

    async def tier1_annotate(self, chunks: list[ConversationChunk]) -> list[dict]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = [self._annotate_chunk(c, semaphore) for c in chunks]

        annotations = []
        with tqdm(total=len(tasks), desc="Tier-1 annotation") as pbar:
            for coro in asyncio.as_completed(tasks):
                _, result = await coro
                if result:
                    annotations.append(result)
                pbar.update(1)

        return annotations

    # ============================================================
    # Tier-1.5: Per-chat summarization
    # ============================================================

    async def _summarize_chat(
        self, chat_name: str, chat_category: str, annotations: list[dict],
        semaphore: asyncio.Semaphore,
    ) -> dict | None:
        annotations_text = "\n".join(json.dumps(a, ensure_ascii=False) for a in annotations)
        input_hash = hashlib.sha256(annotations_text.encode()).hexdigest()[:12]
        key = _cache_key(input_hash, self.config.tier1_model, self.prompt_ver, self.source_hash, "t1.5")
        cached = self._load_cache(key)
        if cached:
            return cached

        prompt = TIER1_5_SUMMARY.format(
            chat_name=chat_name,
            chat_category=chat_category,
            annotation_count=len(annotations),
            annotations_text=annotations_text,
        )

        async with semaphore:
            try:
                result = await self._call_llm(
                    self.config.tier1_model, TIER1_5_SYSTEM, prompt
                )
                self._save_cache(key, result)
                return result
            except Exception as e:
                print(f"Error summarizing {chat_name}: {e}", file=sys.stderr)
                return None

    async def tier1_5_summarize(self, annotations: list[dict]) -> list[dict]:
        """Summarize annotations per chat for Tier-2 consumption."""
        by_chat: dict[str, list[dict]] = {}
        for a in annotations:
            chat_name = a.get("_chat_name", "unknown")
            by_chat.setdefault(chat_name, []).append(a)

        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = []
        for chat_name, chat_anns in by_chat.items():
            category = chat_anns[0].get("_chat_category", "unknown")
            tasks.append(self._summarize_chat(chat_name, category, chat_anns, semaphore))

        summaries = []
        with tqdm(total=len(tasks), desc="Tier-1.5 summarization") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    summaries.append(result)
                pbar.update(1)

        return summaries

    # ============================================================
    # Tier-2: Identity model (replaces tier2_personality)
    # ============================================================

    async def tier2_identity_model(self, summaries: list[dict], behavioral_stats: dict) -> dict:
        summaries_text = "\n\n".join(json.dumps(s, ensure_ascii=False) for s in summaries)
        stats_text = json.dumps(behavioral_stats, ensure_ascii=False, indent=2)

        input_hash = hashlib.sha256((summaries_text + stats_text).encode()).hexdigest()[:12]
        key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_identity")
        cached = self._load_cache(key)
        if cached:
            return cached

        cat_counts = Counter(s.get("chat_category", "unknown") for s in summaries)

        prompt = TIER2_IDENTITY_USER.format(
            user_name=self.user_name,
            chat_count=len(summaries),
            private_count=cat_counts.get("personal", 0),
            group_count=cat_counts.get("group", 0),
            channel_count=cat_counts.get("channel", 0),
            behavioral_stats=stats_text,
            summaries_text=summaries_text,
        )

        print("Running Tier-2 identity model...")
        try:
            result = await self._call_llm(
                self.config.tier2_model, TIER2_IDENTITY_SYSTEM, prompt
            )
            self._save_cache(key, result)
            return result
        except Exception as e:
            print(f"Error in identity model: {e}", file=sys.stderr)
            return {}

    # ============================================================
    # Tier-2: Style rules (depends on identity_model)
    # ============================================================

    async def tier2_style_rules(self, summaries: list[dict], behavioral_stats: dict, identity_model: dict) -> dict:
        summaries_text = "\n\n".join(json.dumps(s, ensure_ascii=False) for s in summaries)
        stats_text = json.dumps(behavioral_stats, ensure_ascii=False, indent=2)
        identity_summary = json.dumps(identity_model, ensure_ascii=False, indent=2)

        input_hash = hashlib.sha256(
            (summaries_text + stats_text + identity_summary).encode()
        ).hexdigest()[:12]
        key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_style_rules")
        cached = self._load_cache(key)
        if cached:
            return cached

        prompt = TIER2_STYLE_RULES_USER.format(
            user_name=self.user_name,
            identity_summary=identity_summary,
            behavioral_stats=stats_text,
            summaries_text=summaries_text,
        )

        print("Running Tier-2 style rules extraction...")
        try:
            result = await self._call_llm(
                self.config.tier2_model, TIER2_STYLE_RULES_SYSTEM, prompt
            )
            self._save_cache(key, result)
            return result
        except Exception as e:
            print(f"Error in style rules extraction: {e}", file=sys.stderr)
            return {"rules": [], "boundaries": [], "fallback_rules": []}

    # ============================================================
    # Tier-2: Relationships (split: contacts + groups)
    # ============================================================

    async def tier2_contacts(self, annotations: list[dict], summaries: list[dict]) -> dict[str, dict]:
        """Analyze individual relationships from both private chats and group interactions."""
        results = {}

        # --- Private chat contacts ---
        private_by_chat: dict[str, list[dict]] = {}
        for a in annotations:
            if a.get("_chat_category") == "personal":
                private_by_chat.setdefault(a["_chat_name"], []).append(a)

        for chat_name, chat_anns in private_by_chat.items():
            if len(chat_anns) < 2:
                continue

            # Resolve UID from pre-built mapping (extracted from message from_ids)
            # No fallback to name lookup — it's unreliable when chat title != display name
            contact_uid = self._private_chat_uids.get(chat_name, "")

            # Skip self (saved messages or self-chat)
            if contact_uid == self.user_id:
                continue

            # Skip deleted/unknown accounts
            if contact_uid in ("user_unknown", "unknown", "user0"):
                continue

            anns_text = "\n".join(json.dumps(a, ensure_ascii=False) for a in chat_anns)
            input_hash = hashlib.sha256(anns_text.encode()).hexdigest()[:12]
            key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_priv_rel")

            cached = self._load_cache(key)
            if cached:
                display_key = f"{chat_name} ({contact_uid})" if contact_uid else chat_name
                results[display_key] = cached
                continue

            prompt = TIER2_PRIVATE_RELATIONSHIP_USER.format(
                user_name=self.user_name,
                contact_name=chat_name,
                contact_uid=contact_uid or "unknown",
                segment_count=len(chat_anns),
                annotations_text=anns_text,
            )

            print(f"Analyzing private contact: {chat_name}...")
            try:
                result = await self._call_llm(
                    self.config.tier2_model, TIER2_RELATIONSHIP_SYSTEM, prompt
                )
                result["uid"] = contact_uid
                self._save_cache(key, result)
                display_key = f"{chat_name} ({contact_uid})" if contact_uid else chat_name
                results[display_key] = result
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

        # --- Group-extracted individual contacts ---
        person_interactions: dict[str, list[dict]] = defaultdict(list)
        person_groups: dict[str, set[str]] = defaultdict(set)

        for a in annotations:
            if a.get("_chat_category") != "group":
                continue
            chat_name = a.get("_chat_name", "")
            for interaction in a.get("person_interactions", []):
                # Use uid as the key (stable identity)
                uid = (interaction.get("person_uid") or "").strip()
                name = (interaction.get("person_name") or interaction.get("person") or "").strip()
                if not uid and not name:
                    continue
                # Skip self (by UID or by name match)
                if uid == self.user_id:
                    continue
                if not uid and name == self.user_name:
                    continue
                # Skip invalid/placeholder UIDs and deleted accounts
                if uid in ("user_unknown", "unknown", "user0"):
                    continue
                # If LLM didn't provide uid, try to resolve from identity map
                if not uid and name:
                    uid = self.identity_map.resolve_name_to_id(name) or ""
                # Re-check after resolution in case name resolved to self
                if uid == self.user_id:
                    continue
                record = {**interaction, "_group": chat_name}
                person_key = uid or name  # prefer uid
                person_interactions[person_key].append(record)
                person_groups[person_key].add(chat_name)

        for person_key, interactions in person_interactions.items():
            if len(interactions) < MIN_GROUP_INTERACTIONS:
                continue

            # Resolve display name
            if person_key.startswith("user"):
                display_name = self.identity_map.get_latest_name(person_key)
                uid = person_key
            else:
                display_name = person_key
                uid = ""

            # Skip deleted/unknown accounts
            if uid in ("user_unknown", "unknown", "user0"):
                continue
            if display_name.lower() in ("deleted account", "unknown", "user_unknown"):
                continue

            display_key = f"{display_name} ({uid})" if uid else display_name

            # Skip if already covered by private chat (compare by UID, not name substring)
            existing_uids = {r.get("uid", "") for r in results.values() if r.get("uid")}
            if uid and uid in existing_uids:
                continue

            # Time-balanced sampling: keep first 10, last 10, and evenly spaced from middle
            sampled = interactions
            max_interactions = 50
            if len(interactions) > max_interactions:
                first = interactions[:10]
                last = interactions[-10:]
                middle_pool = interactions[10:-10]
                middle_count = max_interactions - 20
                step = max(1, len(middle_pool) // middle_count)
                middle = middle_pool[::step][:middle_count]
                sampled = first + middle + last

            interactions_text = "\n".join(
                json.dumps(i, ensure_ascii=False) for i in sampled
            )
            input_hash = hashlib.sha256(interactions_text.encode()).hexdigest()[:12]
            key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_grp_rel")

            cached = self._load_cache(key)
            if cached:
                results[display_key] = cached
                continue

            groups = sorted(person_groups[person_key])
            prompt = TIER2_GROUP_RELATIONSHIP_USER.format(
                user_name=self.user_name,
                contact_name=display_name,
                contact_uid=uid or "unknown",
                groups=", ".join(groups),
                interaction_count=len(interactions),
                interactions_text=interactions_text,
                groups_json=json.dumps(groups, ensure_ascii=False),
            )

            print(f"Analyzing group contact: {display_name} ({len(interactions)} interactions)...")
            try:
                result = await self._call_llm(
                    self.config.tier2_model, TIER2_RELATIONSHIP_SYSTEM, prompt
                )
                result["uid"] = uid
                self._save_cache(key, result)
                results[display_key] = result
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

        return results

    async def tier2_groups(self, summaries: list[dict]) -> dict[str, dict]:
        """Analyze user's behavior profile within each group."""
        results = {}

        group_summaries = [s for s in summaries if s.get("chat_category") == "group"]

        for summary in group_summaries:
            group_name = summary.get("chat_name", "unknown")

            summary_text = json.dumps(summary, ensure_ascii=False)
            input_hash = hashlib.sha256(summary_text.encode()).hexdigest()[:12]
            key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_grp_prof")

            cached = self._load_cache(key)
            if cached:
                results[group_name] = cached
                continue

            prompt = TIER2_GROUP_PROFILE_USER.format(
                user_name=self.user_name,
                group_name=group_name,
                chat_type="group",
                segment_count=summary.get("annotation_count", 0),
                summary_text=summary_text,
            )

            print(f"Analyzing group profile: {group_name}...")
            try:
                result = await self._call_llm(
                    self.config.tier2_model, TIER2_GROUP_PROFILE_SYSTEM, prompt
                )
                self._save_cache(key, result)
                results[group_name] = result
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

        return results

    # ============================================================
    # Tier-2: Topic graph (replaces tier2_knowledge)
    # ============================================================

    async def tier2_topic_graph(self, summaries: list[dict]) -> dict:
        all_knowledge = []
        for s in summaries:
            for k in s.get("knowledge_summary", []):
                if k.get("key_points"):
                    k["_chat"] = s.get("chat_name", "")
                    all_knowledge.append(k)

        # Collect all topics from summaries for topic distribution
        all_topics: Counter[str] = Counter()
        for s in summaries:
            for t in s.get("topics", []):
                all_topics[t] += 1

        if not all_knowledge and not all_topics:
            return {"domains": [], "overall_profile": "No significant knowledge or topic data."}

        knowledge_text = "\n".join(json.dumps(k, ensure_ascii=False) for k in all_knowledge)
        topics_text = json.dumps(dict(all_topics.most_common(50)), ensure_ascii=False)

        input_hash = hashlib.sha256(
            (knowledge_text + topics_text).encode()
        ).hexdigest()[:12]
        key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_topic_graph")

        cached = self._load_cache(key)
        if cached:
            return cached

        prompt = TIER2_TOPIC_GRAPH_USER.format(
            user_name=self.user_name,
            chunk_count=len(all_knowledge),
            knowledge_text=knowledge_text or "(no explicit knowledge records)",
            topics_text=topics_text,
        )

        print(f"Running Tier-2 topic graph ({len(all_knowledge)} knowledge records, {len(all_topics)} topics)...")
        try:
            result = await self._call_llm(
                self.config.tier2_model, TIER2_TOPIC_GRAPH_SYSTEM, prompt
            )
            self._save_cache(key, result)
            return result
        except Exception as e:
            print(f"Error in topic graph: {e}", file=sys.stderr)
            return {"domains": [], "overall_profile": "Analysis failed."}

    # ============================================================
    # Tier-2: Response policies
    # ============================================================

    async def tier2_response_policies(self, summaries: list[dict], behavioral_stats: dict) -> dict:
        summaries_text = "\n\n".join(json.dumps(s, ensure_ascii=False) for s in summaries)
        stats_text = json.dumps(behavioral_stats, ensure_ascii=False, indent=2)

        input_hash = hashlib.sha256(
            (summaries_text + stats_text).encode()
        ).hexdigest()[:12]
        key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_response_policies")

        cached = self._load_cache(key)
        if cached:
            return cached

        prompt = TIER2_RESPONSE_POLICIES_USER.format(
            user_name=self.user_name,
            summaries_text=summaries_text,
            behavioral_stats=stats_text,
        )

        print("Running Tier-2 response policies...")
        try:
            result = await self._call_llm(
                self.config.tier2_model, TIER2_RESPONSE_POLICIES_SYSTEM, prompt
            )
            self._save_cache(key, result)
            return result
        except Exception as e:
            print(f"Error in response policies: {e}", file=sys.stderr)
            return {"triggers": [], "boundaries": []}

    # ============================================================
    # Tier-2: Memes
    # ============================================================

    async def tier2_memes(self, summaries: list[dict]) -> dict:
        all_memes = []
        for s in summaries:
            for m in s.get("memes_summary", []):
                if m.get("expression"):
                    m["_chat"] = s.get("chat_name", "")
                    all_memes.append(m)

        if not all_memes:
            return {"meme_categories": [], "meme_style_summary": "No significant meme usage detected."}

        memes_text = "\n".join(json.dumps(m, ensure_ascii=False) for m in all_memes)
        input_hash = hashlib.sha256(memes_text.encode()).hexdigest()[:12]
        key = _cache_key(input_hash, self.config.tier2_model, self.prompt_ver, self.source_hash, "t2_memes")

        cached = self._load_cache(key)
        if cached:
            return cached

        prompt = TIER2_MEME_USER.format(
            user_name=self.user_name,
            record_count=len(all_memes),
            memes_text=memes_text,
        )

        print(f"Running Tier-2 meme distillation ({len(all_memes)} records)...")
        try:
            result = await self._call_llm(
                self.config.tier2_model, TIER2_MEME_SYSTEM, prompt
            )
            self._save_cache(key, result)
            return result
        except Exception as e:
            print(f"Error in meme distillation: {e}", file=sys.stderr)
            return {"meme_categories": [], "meme_style_summary": "Analysis failed."}

    # ============================================================
    # Aggregate stats (from annotations, not summaries)
    # ============================================================

    def _aggregate_stats(self, annotations: list[dict]) -> dict[str, Any]:
        lang_counter = Counter(a.get("language", "unknown") for a in annotations)
        tone_counter = Counter(a.get("tone", "unknown") for a in annotations)
        formalities = [a.get("formality", 3) for a in annotations]
        all_topics = []
        for a in annotations:
            all_topics.extend(a.get("topics", []))
        topic_counter = Counter(all_topics)
        cat_counter = Counter(a.get("_chat_category", "unknown") for a in annotations)

        return {
            "lang_stats": dict(lang_counter.most_common()),
            "tone_stats": dict(tone_counter.most_common()),
            "avg_formality": sum(formalities) / len(formalities) if formalities else 3.0,
            "top_topics": [t for t, _ in topic_counter.most_common(20)],
            "category_counts": dict(cat_counter),
        }

    # ============================================================
    # Full pipeline
    # ============================================================

    async def run_tier2(self, summaries: list[dict], annotations: list[dict], behavioral_stats: dict) -> dict:
        """Run only Tier-2 (Phase A + B), reusing existing annotations and summaries."""
        # Tier-2 Phase A (independent analyses, run concurrently)
        print("Running Tier-2 Phase A...")
        phase_a = await asyncio.gather(
            self.tier2_identity_model(summaries, behavioral_stats),
            self.tier2_contacts(annotations, summaries),
            self.tier2_groups(summaries),
            self.tier2_topic_graph(summaries),
            self.tier2_memes(summaries),
            return_exceptions=True,
        )

        phase_a_names = ["identity_model", "contacts", "groups", "topic_graph", "memes"]
        phase_a_fallbacks = [{}, {}, {}, {}, {}]
        tier2 = {}
        for name, result, fallback in zip(phase_a_names, phase_a, phase_a_fallbacks):
            if isinstance(result, BaseException):
                print(f"Warning: Tier-2 {name} failed: {result}", file=sys.stderr)
                tier2[name] = fallback
            else:
                tier2[name] = result

        # Tier-2 Phase B (depends on identity_model from Phase A)
        print("Running Tier-2 Phase B...")
        phase_b = await asyncio.gather(
            self.tier2_style_rules(summaries, behavioral_stats, tier2["identity_model"]),
            self.tier2_response_policies(summaries, behavioral_stats),
            return_exceptions=True,
        )

        phase_b_names = ["style_rules", "response_policies"]
        phase_b_fallbacks = [
            {"rules": [], "boundaries": [], "fallback_rules": []},
            {"triggers": [], "boundaries": []},
        ]
        for name, result, fallback in zip(phase_b_names, phase_b, phase_b_fallbacks):
            if isinstance(result, BaseException):
                print(f"Warning: Tier-2 {name} failed: {result}", file=sys.stderr)
                tier2[name] = fallback
            else:
                tier2[name] = result

        return {
            "annotations": annotations,
            "summaries": summaries,
            "identity_model": tier2["identity_model"],
            "style_rules": tier2["style_rules"],
            "contacts": tier2["contacts"],
            "groups": tier2["groups"],
            "topic_graph": tier2["topic_graph"],
            "response_policies": tier2["response_policies"],
            "memes": tier2["memes"],
            "stats": self._aggregate_stats(annotations),
        }

    async def run(self, chunks: list[ConversationChunk], behavioral_stats: dict) -> dict:
        """Run full pipeline: Tier-1 → Tier-1.5 → Tier-2 (Phase A + B)."""
        # Tier-1
        annotations = await self.tier1_annotate(chunks)
        if not annotations:
            raise RuntimeError("No annotations produced from Tier-1")

        # Tier-1.5
        summaries = await self.tier1_5_summarize(annotations)
        if not summaries:
            raise RuntimeError("No summaries produced from Tier-1.5")

        # Tier-2
        return await self.run_tier2(summaries, annotations, behavioral_stats)
