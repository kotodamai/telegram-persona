"""Parse Telegram Desktop JSON export into structured messages."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Reaction:
    emoji: str
    count: int
    from_ids: list[str]
    from_names: list[str]


@dataclass
class Message:
    id: int
    text: str
    date: datetime
    from_id: str
    from_name: str
    chat_name: str
    chat_type: str
    chat_id: int
    reply_to_id: int | None = None
    sticker_emoji: str | None = None
    is_edited: bool = False
    reactions: list[Reaction] = field(default_factory=list)
    is_service: bool = False
    media_type: str | None = None
    forwarded_from: str | None = None

    @property
    def is_text(self) -> bool:
        return bool(self.text) and not self.is_service

    @property
    def is_sticker(self) -> bool:
        return self.media_type == "sticker"

    @property
    def has_link(self) -> bool:
        return "http://" in self.text or "https://" in self.text


@dataclass
class Chat:
    name: str
    chat_type: str
    chat_id: int
    messages: list[Message]
    msg_index: dict[int, Message] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.msg_index = {m.id: m for m in self.messages}

    def get_message(self, msg_id: int) -> Message | None:
        return self.msg_index.get(msg_id)


class IdentityMap:
    """Maps from_id to display names. Handles name changes by tracking all seen names."""

    def __init__(self):
        self._id_to_names: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    def record(self, from_id: str, from_name: str, date: datetime):
        if from_id and from_name:
            self._id_to_names[from_id].append((date, from_name))

    def get_latest_name(self, from_id: str) -> str:
        entries = self._id_to_names.get(from_id, [])
        if not entries:
            return from_id
        # Return name from most recent message
        return max(entries, key=lambda x: x[0])[1]

    def get_all_names(self, from_id: str) -> set[str]:
        return {name for _, name in self._id_to_names.get(from_id, [])}

    def all_ids(self) -> list[str]:
        return list(self._id_to_names.keys())

    def resolve_name_to_id(self, name: str) -> str | None:
        """Best-effort reverse lookup: name → id. Returns None if ambiguous."""
        candidates = []
        for uid, entries in self._id_to_names.items():
            for _, n in entries:
                if n == name:
                    candidates.append(uid)
                    break
        if len(candidates) == 1:
            return candidates[0]
        return None


@dataclass
class ExportData:
    user_id: str
    user_name: str
    chats: list[Chat]
    identity_map: IdentityMap = field(default_factory=IdentityMap)
    source_hash: str = ""

    @property
    def all_messages(self) -> list[Message]:
        return [m for c in self.chats for m in c.messages]

    def user_messages(self) -> list[Message]:
        return [m for m in self.all_messages if m.from_id == self.user_id]


def flatten_text(text_field: str | list[Any]) -> str:
    """Convert Telegram text field (string or entity array) to plain text with format markers."""
    if isinstance(text_field, str):
        return text_field

    parts = []
    for entity in text_field:
        if isinstance(entity, str):
            parts.append(entity)
        elif isinstance(entity, dict):
            t = entity.get("text", "")
            etype = entity.get("type", "plain")
            if etype == "strikethrough":
                parts.append(f"~~{t}~~")
            elif etype == "bold":
                parts.append(f"**{t}**")
            elif etype == "italic":
                parts.append(f"_{t}_")
            elif etype == "code":
                parts.append(f"`{t}`")
            elif etype == "pre":
                parts.append(f"```{t}```")
            elif etype in ("text_link", "link"):
                parts.append(t)
            elif etype == "mention":
                parts.append(t)
            else:
                parts.append(t)
    return "".join(parts)


def parse_reactions(raw_reactions: list[dict[str, Any]] | None) -> list[Reaction]:
    if not raw_reactions:
        return []
    result = []
    for r in raw_reactions:
        emoji = r.get("emoji", "")
        count = r.get("count", 0)
        from_ids = [p.get("from_id", "") for p in r.get("recent", [])]
        from_names = [p.get("from", "unknown") for p in r.get("recent", [])]
        result.append(Reaction(emoji=emoji, count=count, from_ids=from_ids, from_names=from_names))
    return result


def parse_message(raw: dict[str, Any], chat_name: str, chat_type: str, chat_id: int) -> Message:
    text = flatten_text(raw.get("text", ""))
    sticker_emoji = raw.get("sticker_emoji")
    media_type = raw.get("media_type")

    # For stickers, represent as text marker
    if media_type == "sticker" and sticker_emoji:
        text = f"[sticker:{sticker_emoji}]"

    return Message(
        id=raw["id"],
        text=text,
        date=datetime.fromisoformat(raw["date"]),
        from_id=raw.get("from_id", ""),
        from_name=raw.get("from", ""),
        chat_name=chat_name,
        chat_type=chat_type,
        chat_id=chat_id,
        reply_to_id=raw.get("reply_to_message_id"),
        sticker_emoji=sticker_emoji,
        is_edited="edited" in raw,
        reactions=parse_reactions(raw.get("reactions")),
        is_service=raw.get("type") == "service",
        media_type=media_type,
        forwarded_from=raw.get("forwarded_from"),
    )


BOT_CHAT_TYPE = "bot_chat"


def _file_hash(path: Path) -> str:
    """Compute SHA256 of a file (first 1MB for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def parse_export(
    path: str | Path,
    skip_bots: bool = True,
    since: datetime | None = None,
    until: datetime | None = None,
    include_chats: set[str] | None = None,
    exclude_chats: set[str] | None = None,
    min_messages: int = 0,
) -> ExportData:
    """Parse Telegram export JSON file.

    Args:
        path: Path to result.json
        skip_bots: Whether to skip bot chats
        since: Only include messages after this date
        until: Only include messages before this date
        include_chats: Whitelist of chat names (None = all)
        exclude_chats: Blacklist of chat names (None = none)
        min_messages: Skip chats with fewer user messages than this
    """
    path = Path(path)
    source_hash = _file_hash(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Extract user identity
    personal = raw.get("personal_information", {})
    user_id_num = personal.get("user_id", 0)
    user_id = f"user{user_id_num}"
    first = personal.get("first_name", "")
    last = personal.get("last_name", "")
    user_name = f"{first} {last}".strip() or "Unknown"

    identity_map = IdentityMap()
    chats = []

    for raw_chat in raw.get("chats", {}).get("list", []):
        chat_type = raw_chat.get("type", "unknown")
        if skip_bots and chat_type == BOT_CHAT_TYPE:
            continue

        chat_name = raw_chat.get("name", f"unnamed_{raw_chat.get('id', '?')}")
        chat_id = raw_chat.get("id", 0)

        # Chat filtering
        if include_chats and chat_name not in include_chats:
            continue
        if exclude_chats and chat_name in exclude_chats:
            continue

        messages = []
        for raw_msg in raw_chat.get("messages", []):
            msg = parse_message(raw_msg, chat_name, chat_type, chat_id)

            # Date filtering
            if since and msg.date < since:
                continue
            if until and msg.date > until:
                continue

            messages.append(msg)

            # Build identity map
            identity_map.record(msg.from_id, msg.from_name, msg.date)
            # Also record reaction authors
            for reaction in msg.reactions:
                for rid, rname in zip(reaction.from_ids, reaction.from_names):
                    if rid and rname:
                        identity_map.record(rid, rname, msg.date)

        # Min messages filter
        user_msg_count = sum(1 for m in messages if m.from_id == user_id and not m.is_service)
        if user_msg_count < min_messages:
            continue

        chats.append(Chat(
            name=chat_name,
            chat_type=chat_type,
            chat_id=chat_id,
            messages=messages,
        ))

    return ExportData(
        user_id=user_id,
        user_name=user_name,
        chats=chats,
        identity_map=identity_map,
        source_hash=source_hash,
    )


def _detect_language_simple(text: str) -> str:
    """Simple heuristic language detection: CJK vs Latin."""
    if not text:
        return "unknown"
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    total = len(text.strip())
    if total == 0:
        return "unknown"
    if cjk_count / total > 0.3:
        return "zh"
    return "en"


def _compute_burst_lengths(chat_msgs: list[Message], user_id: str) -> list[int]:
    """Compute consecutive message burst lengths for user within a chat."""
    bursts = []
    current = 0
    for m in chat_msgs:
        if m.is_service:
            continue
        if m.from_id == user_id:
            current += 1
        else:
            if current > 0:
                bursts.append(current)
            current = 0
    if current > 0:
        bursts.append(current)
    return bursts


def _length_stats(values: list[int | float]) -> dict[str, float]:
    """Compute mean/median/p90 for a list of numbers."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "p90": round(sorted(values)[int(len(values) * 0.9)], 1),
    }


def compute_behavioral_stats(export: ExportData) -> dict[str, Any]:
    """Compute behavioral statistics from non-text signals.

    These stats are more reliable than LLM inference for quantitative patterns.
    """
    from .chunker import classify_chat_type

    my_msgs = export.user_messages()
    all_msgs = export.all_messages
    if not my_msgs:
        return {}

    user_id = export.user_id
    id_map = export.identity_map
    total = len(my_msgs)

    # Classify chats for per-category stats
    chat_categories = {c.chat_id: classify_chat_type(c.chat_type) for c in export.chats}

    # Reply patterns
    reply_count = sum(1 for m in my_msgs if m.reply_to_id)
    initiation_count = sum(1 for m in my_msgs if not m.reply_to_id)

    reply_targets: Counter[str] = Counter()
    reply_latencies: list[float] = []
    for m in my_msgs:
        if m.reply_to_id:
            for chat in export.chats:
                if chat.chat_id == m.chat_id:
                    ref = chat.get_message(m.reply_to_id)
                    if ref and ref.from_id != user_id:
                        reply_targets[ref.from_id] += 1
                        # Reply latency (only private chats)
                        if chat_categories.get(chat.chat_id) == "personal":
                            delta = (m.date - ref.date).total_seconds()
                            if 0 < delta < 86400:  # within 24h
                                reply_latencies.append(delta)
                    break

    # Reactions received on my messages
    reaction_received: Counter[str] = Counter()
    for m in my_msgs:
        for r in m.reactions:
            reaction_received[r.emoji] += r.count

    # Rates
    sticker_count = sum(1 for m in my_msgs if m.is_sticker)
    edit_count = sum(1 for m in my_msgs if m.is_edited)
    forward_count = sum(1 for m in my_msgs if m.forwarded_from)
    link_count = sum(1 for m in my_msgs if m.has_link)
    code_count = sum(1 for m in my_msgs if "```" in m.text)
    photo_count = sum(1 for m in my_msgs if m.media_type == "photo")
    video_count = sum(1 for m in my_msgs if m.media_type in ("video_file", "video_message"))

    # Message length stats — global and per category
    text_msgs = [m for m in my_msgs if m.is_text and not m.is_sticker]
    lengths = [len(m.text) for m in text_msgs] if text_msgs else [0]

    lengths_by_cat: dict[str, list[int]] = defaultdict(list)
    for m in text_msgs:
        cat = chat_categories.get(m.chat_id, "other")
        lengths_by_cat[cat].append(len(m.text))

    # Active hours — global and per category
    active_hours: Counter[int] = Counter()
    active_hours_by_cat: dict[str, Counter[int]] = defaultdict(Counter)
    for m in my_msgs:
        active_hours[m.date.hour] += 1
        cat = chat_categories.get(m.chat_id, "other")
        active_hours_by_cat[cat][m.date.hour] += 1

    # Media mix
    media_mix: Counter[str] = Counter()
    for m in my_msgs:
        if m.media_type:
            media_mix[m.media_type] += 1

    # Messages per day
    daily: Counter[str] = Counter()
    for m in my_msgs:
        daily[m.date.strftime("%Y-%m-%d")] += 1

    # Burst lengths (consecutive messages from user before someone else speaks)
    all_bursts: list[int] = []
    for chat in export.chats:
        all_bursts.extend(_compute_burst_lengths(chat.messages, user_id))

    burst_dist: Counter[int] = Counter(all_bursts)

    # Initiation ratio per category
    init_by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"init": 0, "total": 0})
    for m in my_msgs:
        cat = chat_categories.get(m.chat_id, "other")
        init_by_cat[cat]["total"] += 1
        if not m.reply_to_id:
            init_by_cat[cat]["init"] += 1

    # Language switch rate
    prev_lang = None
    switch_count = 0
    lang_total = 0
    for m in sorted(my_msgs, key=lambda x: (x.chat_id, x.date)):
        if not m.is_text or m.is_sticker:
            continue
        lang = _detect_language_simple(m.text)
        if lang == "unknown":
            continue
        if prev_lang is not None and lang != prev_lang:
            switch_count += 1
        prev_lang = lang
        lang_total += 1

    # Reply target names (resolved)
    reply_target_names = {
        id_map.get_latest_name(uid): count
        for uid, count in reply_targets.most_common(20)
    }

    return {
        "total_messages": total,
        "reply_ratio": round(reply_count / total, 3) if total else 0,
        "initiation_ratio": round(initiation_count / total, 3) if total else 0,
        "reply_targets": reply_target_names,
        "reaction_received": dict(reaction_received.most_common(20)),
        "sticker_rate": round(sticker_count / total, 3) if total else 0,
        "edit_rate": round(edit_count / total, 3) if total else 0,
        "forward_rate": round(forward_count / total, 3) if total else 0,
        "link_rate": round(link_count / total, 3) if total else 0,
        "code_rate": round(code_count / total, 3) if total else 0,
        "photo_rate": round(photo_count / total, 3) if total else 0,
        "video_rate": round(video_count / total, 3) if total else 0,
        "media_mix": dict(media_mix.most_common()),
        "message_length": _length_stats(lengths),
        "message_length_by_category": {
            cat: _length_stats(lens) for cat, lens in lengths_by_cat.items()
        },
        "burst_lengths": {
            **_length_stats(all_bursts),
            "distribution": dict(sorted(burst_dist.items())[:10]),
        },
        "reply_latency_seconds": _length_stats(reply_latencies),
        "active_hours": dict(sorted(active_hours.items())),
        "active_hours_by_category": {
            cat: dict(sorted(hours.items()))
            for cat, hours in active_hours_by_cat.items()
        },
        "initiation_ratio_by_category": {
            cat: round(v["init"] / v["total"], 3) if v["total"] else 0
            for cat, v in init_by_cat.items()
        },
        "language_switch_rate": round(switch_count / lang_total, 3) if lang_total > 1 else 0,
        "avg_messages_per_active_day": (
            round(statistics.mean(daily.values()), 1) if daily else 0
        ),
        "active_days": len(daily),
    }
