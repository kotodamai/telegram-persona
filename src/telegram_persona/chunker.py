"""Split parsed messages into conversation chunks with context for LLM analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from .parser import Chat, ExportData, IdentityMap, Message

# Gap threshold for splitting conversations
CONVERSATION_GAP = timedelta(minutes=30)
# Shorter gap for burst detection in group chats
BURST_GAP = timedelta(minutes=5)

# Target ~3000 tokens per chunk
MAX_CHUNK_CHARS = 6000

# Chat type classification
GROUP_TYPES = {"private_supergroup", "public_supergroup"}
CHANNEL_TYPES = {"private_channel", "public_channel"}
PERSONAL_TYPES = {"personal_chat"}
SKIP_TYPES = {"saved_messages"}


def classify_chat_type(chat_type: str) -> str:
    """Classify raw Telegram chat type into semantic category."""
    if chat_type in PERSONAL_TYPES:
        return "personal"
    elif chat_type in GROUP_TYPES:
        return "group"
    elif chat_type in CHANNEL_TYPES:
        return "channel"
    elif chat_type in SKIP_TYPES:
        return "skip"
    else:
        return "other"


@dataclass
class ConversationChunk:
    chat_name: str
    chat_type: str  # raw Telegram type
    chat_category: str  # "personal" | "group" | "channel"
    messages: list[Message]
    reply_context: dict[int, Message] = field(default_factory=dict)

    def render(self, user_id: str) -> str:
        """Render chunk as text for LLM consumption, with type-aware context and uid annotations."""
        lines = []

        # Type-aware header
        if self.chat_category == "personal":
            other_name = self._get_other_name(user_id)
            other_id = self._get_other_id(user_id)
            id_note = f" [uid:{other_id}]" if other_id else ""
            lines.append(f"[Private Chat with: {other_name}{id_note}]")
            lines.append(f"This is a 1-on-1 private conversation between [Self] and {other_name}.")
        elif self.chat_category == "group":
            participants = self._get_participants(user_id)
            lines.append(f"[Group Chat: {self.chat_name}]")
            lines.append(f"This is a group chat with multiple participants.")
            if participants:
                p_list = ", ".join(f"{name} [uid:{uid}]" for name, uid in participants)
                lines.append(f"Participants in this segment: {p_list}")
        elif self.chat_category == "channel":
            lines.append(f"[Channel: {self.chat_name}]")
            lines.append(f"This is a channel where [Self] publishes content.")

        lines.append("")

        for msg in self.messages:
            ts = msg.date.strftime("%Y-%m-%d %H:%M")

            if msg.from_id == user_id:
                role = "[Self]"
            else:
                role = f"[{msg.from_name}|{msg.from_id}]"

            # Show reply context inline
            reply_note = ""
            if msg.reply_to_id and msg.reply_to_id in self.reply_context:
                ref = self.reply_context[msg.reply_to_id]
                ref_preview = ref.text[:60] + ("..." if len(ref.text) > 60 else "")
                reply_note = f" (replying to {ref.from_name}: \"{ref_preview}\")"

            # Reactions on this message
            reaction_note = ""
            if msg.reactions:
                parts = [f"{r.emoji}×{r.count}" for r in msg.reactions]
                reaction_note = f"  [{', '.join(parts)}]"

            text = msg.text if msg.text else "[media]"
            line = f"[{ts}] {role}{reply_note} {text}{reaction_note}"
            lines.append(line)

        return "\n".join(lines)

    def _get_other_name(self, user_id: str) -> str:
        for m in self.messages:
            if m.from_id != user_id and m.from_name:
                return m.from_name
        return self.chat_name

    def _get_other_id(self, user_id: str) -> str:
        for m in self.messages:
            if m.from_id != user_id and m.from_id:
                return m.from_id
        return ""

    def _get_participants(self, user_id: str) -> list[tuple[str, str]]:
        """Get unique (name, uid) pairs for non-self participants."""
        result = []
        seen = set()
        for m in self.messages:
            if m.from_id != user_id and m.from_id not in seen:
                result.append((m.from_name, m.from_id))
                seen.add(m.from_id)
        return result


def _estimate_chars(messages: list[Message]) -> int:
    return sum(len(m.text) + 50 for m in messages)  # 50 chars overhead per line with uid


def _split_into_segments(messages: list[Message], chat_category: str) -> list[list[Message]]:
    """Split messages into conversation segments.

    For groups: use reply-chain awareness and burst detection.
    For personal: use standard time gap.
    """
    if not messages:
        return []

    segments = []
    current = [messages[0]]

    for msg in messages[1:]:
        gap = msg.date - current[-1].date

        # Should we start a new segment?
        should_split = False

        if gap > CONVERSATION_GAP:
            # Long gap — always split
            should_split = True
        elif chat_category == "group" and gap > BURST_GAP:
            # In groups, also consider splitting at burst boundaries
            # BUT don't split if this message replies to something in the current segment
            current_ids = {m.id for m in current}
            if msg.reply_to_id and msg.reply_to_id in current_ids:
                # This message is part of an ongoing thread — don't split
                should_split = False
            else:
                should_split = True

        if should_split:
            segments.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        segments.append(current)

    return segments


def _segment_has_user_message(segment: list[Message], user_id: str) -> bool:
    return any(m.from_id == user_id for m in segment)


def _find_split_point(segment: list[Message], start: int, target: int, user_id: str) -> int:
    """Find a good split point near target index.

    Prefers splitting at:
    1. Speaker changes with no reply link
    2. Speaker changes
    3. Target index (fallback)
    """
    # Search window: target ± 20%
    window = max(3, int(target * 0.2))
    lo = max(start + 1, target - window)
    hi = min(len(segment) - 1, target + window)

    best = target
    best_score = 0

    for i in range(lo, hi + 1):
        if i >= len(segment):
            break
        score = 0
        prev = segment[i - 1]
        curr = segment[i]

        # Speaker change
        if prev.from_id != curr.from_id:
            score += 2

        # No reply link to previous segment
        current_ids = {m.id for m in segment[start:i]}
        if curr.reply_to_id and curr.reply_to_id in current_ids:
            score -= 3  # strong penalty: don't break reply chains

        # Don't split in the middle of user's consecutive messages
        if prev.from_id == user_id and curr.from_id == user_id:
            score -= 2

        if score > best_score:
            best_score = score
            best = i

    return best


def _split_oversized_segment(segment: list[Message], user_id: str) -> list[list[Message]]:
    """Split an oversized segment with smart split point selection."""
    if _estimate_chars(segment) <= MAX_CHUNK_CHARS:
        return [segment]

    chunks = []
    start = 0

    while start < len(segment):
        # Find how many messages fit in the budget
        running = 0
        end = start
        for i in range(start, len(segment)):
            msg_size = len(segment[i].text) + 50
            if running + msg_size > MAX_CHUNK_CHARS and i > start:
                break
            running += msg_size
            end = i + 1

        if end >= len(segment):
            chunks.append(segment[start:])
            break

        # Find a smart split point near 'end'
        split_at = _find_split_point(segment, start, end, user_id)
        if split_at <= start:
            split_at = end  # fallback

        chunks.append(segment[start:split_at])
        start = split_at

    return chunks


def _resolve_reply_context(segment: list[Message], chat: Chat) -> dict[int, Message]:
    """Find the original messages for all reply_to references in this segment."""
    segment_ids = {m.id for m in segment}
    context = {}
    for msg in segment:
        if msg.reply_to_id and msg.reply_to_id not in segment_ids:
            # Only add external references (in-segment replies are visible in context)
            ref = chat.get_message(msg.reply_to_id)
            if ref:
                context[msg.reply_to_id] = ref
    return context


def chunk_chat(chat: Chat, user_id: str) -> list[ConversationChunk]:
    """Chunk a single chat into conversation segments."""
    category = classify_chat_type(chat.chat_type)

    if category == "skip":
        return []

    # Filter out service messages
    messages = [m for m in chat.messages if not m.is_service]
    if not messages:
        return []

    segments = _split_into_segments(messages, category)

    chunks = []
    for segment in segments:
        if not _segment_has_user_message(segment, user_id):
            continue

        sub_segments = _split_oversized_segment(segment, user_id)
        for sub in sub_segments:
            if not _segment_has_user_message(sub, user_id):
                continue

            reply_context = _resolve_reply_context(sub, chat)
            chunks.append(ConversationChunk(
                chat_name=chat.name,
                chat_type=chat.chat_type,
                chat_category=category,
                messages=sub,
                reply_context=reply_context,
            ))

    return chunks


def chunk_export(export: ExportData) -> list[ConversationChunk]:
    """Chunk all chats from an export into conversation segments."""
    all_chunks = []
    for chat in export.chats:
        all_chunks.extend(chunk_chat(chat, export.user_id))
    return all_chunks
