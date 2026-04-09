"""Prompt templates for tiered LLM analysis, differentiated by chat type."""

# ============================================================
# Tier-1: Per-chunk annotation
# ============================================================

TIER1_SYSTEM = """You are a conversation analyst. You analyze chat message excerpts and extract structured observations about a specific user marked as [Self].

Each participant is annotated with a uid in brackets like [uid:user123]. Use these uids when referencing specific people — they are stable identifiers even if display names change.

You MUST respond with valid JSON only. No markdown, no explanations outside the JSON."""


TIER1_PERSONAL_CHAT = """Analyze the following PRIVATE CHAT excerpt. Focus ONLY on the person marked [Self].

This is a 1-on-1 conversation. Pay close attention to:
- How [Self] relates to this specific person (tone, formality, emotional openness)
- Whether [Self] shares expertise or knowledge with this person
- The nature of their relationship (friend, colleague, mentor/mentee, acquaintance, etc.)
- What [Self] chooses NOT to do (avoids certain topics, refuses to engage, ignores questions)
- How [Self] reacts to different types of input (questions, jokes, complaints, requests)

Extract the following as JSON:
{{
  "language": "zh|en|mixed",
  "tone": "neutral|playful|serious|frustrated|enthusiastic|sarcastic|warm|cold",
  "formality": 1-5,
  "topics": ["brief topic keywords"],
  "style_markers": ["notable language patterns: sentence length, punctuation, internet slang, strikethrough humor, etc."],
  "relationship_signals": {{
    "closeness": "intimate|close|familiar|neutral|distant",
    "dynamic": "equal|self_leads|other_leads|self_helps|other_helps",
    "emotional_openness": "high|medium|low"
  }},
  "notable_phrases": ["exact quotes of distinctive expressions from [Self], max 5"],
  "sticker_context": "description of when/why [Self] uses stickers, or null",
  "knowledge_shared": [
    {{
      "domain": "topic area (e.g. networking, programming, finance)",
      "content": "what [Self] explained or advised",
      "depth": "expert|intermediate|casual",
      "original_text": "exact quote from [Self], max 100 chars"
    }}
  ],
  "memes_used": [
    {{
      "expression": "the exact meme phrase, slang, or reference used by [Self]",
      "context": "what was happening in the conversation when it was used",
      "meaning": "what [Self] intended to convey by using it",
      "effect": "how it landed (e.g. got laughs, defused tension, expressed frustration playfully)"
    }}
  ],
  "negative_markers": ["things [Self] explicitly avoided, refused, or declined to engage with in this excerpt — e.g. ignored a question, changed topic, refused to give opinion"],
  "conditional_reactions": [
    {{
      "trigger": "what happened or what the other person did/said",
      "reaction": "how [Self] responded (action, not just tone)",
      "tone_shift": "how [Self]'s tone changed from baseline in this reaction"
    }}
  ],
  "message_structure": {{
    "avg_length": "short(<30chars)|medium(30-100)|long(>100)",
    "multi_message": true or false,
    "uses_code_blocks": true or false,
    "uses_links": true or false
  }}
}}

If [Self] did not share knowledge or use memes/slang, set those arrays to [].
If [Self] did not avoid anything or show conditional reactions, set those arrays to [].
"memes_used" should capture: internet slang, in-group references, ironic phrases, exaggerated expressions, copypasta fragments, reaction catchphrases, or any culturally-loaded shorthand that would be hard for a language model to reproduce without examples.
"negative_markers" should capture: refusals to answer, topic avoidance, deliberate silence on issues, explicit statements of "I won't" or "I don't do that".
"conditional_reactions" should capture: how [Self] changes behavior in response to specific inputs — e.g. becomes more detailed when asked tech questions, gets playful when teased, goes silent when topic gets political.

---

{chunk_text}"""


TIER1_GROUP_CHAT = """Analyze the following GROUP CHAT excerpt. Focus ONLY on the person marked [Self].

This is a multi-person group chat. Each participant has a uid like [uid:user123]. Pay close attention to:
- [Self]'s role in this group (active contributor, lurker, leader, helper, entertainer, etc.)
- How [Self] interacts with EACH specific person — use their uid for identification
- Whether [Self] demonstrates expertise or shares knowledge
- Who [Self] replies to, agrees with, disagrees with, or jokes with
- What [Self] chooses NOT to engage with (ignores certain people, avoids certain topics)
- How [Self]'s behavior changes in response to different triggers

Extract the following as JSON:
{{
  "language": "zh|en|mixed",
  "tone": "neutral|playful|serious|frustrated|enthusiastic|sarcastic|warm|cold",
  "formality": 1-5,
  "topics": ["brief topic keywords"],
  "style_markers": ["notable language patterns"],
  "group_role": "active_contributor|helper|entertainer|lurker|leader|debater",
  "notable_phrases": ["exact quotes of distinctive expressions from [Self], max 5"],
  "sticker_context": "description of when/why [Self] uses stickers, or null",
  "person_interactions": [
    {{
      "person_name": "display name of the other participant",
      "person_uid": "their uid (e.g. user123456)",
      "interaction_type": "helping|asking|debating|agreeing|joking|teasing|coordinating|replying",
      "tone": "warm|neutral|playful|sarcastic|respectful|confrontational",
      "brief": "one-sentence description of what happened between them"
    }}
  ],
  "knowledge_shared": [
    {{
      "domain": "topic area",
      "content": "what [Self] explained or advised",
      "depth": "expert|intermediate|casual",
      "original_text": "exact quote from [Self], max 100 chars"
    }}
  ],
  "memes_used": [
    {{
      "expression": "the exact meme phrase, slang, or reference used by [Self]",
      "context": "what was happening in the conversation when it was used",
      "meaning": "what [Self] intended to convey",
      "effect": "how it landed or what social function it served"
    }}
  ],
  "negative_markers": ["things [Self] explicitly avoided or refused to engage with — e.g. stayed silent during a debate, ignored a direct question, changed topic"],
  "conditional_reactions": [
    {{
      "trigger": "what happened in the group (e.g. someone asked for help, debate started, meme was shared)",
      "reaction": "how [Self] responded",
      "tone_shift": "how [Self]'s tone changed from baseline"
    }}
  ],
  "message_structure": {{
    "avg_length": "short(<30chars)|medium(30-100)|long(>100)",
    "multi_message": true or false,
    "uses_code_blocks": true or false,
    "uses_links": true or false
  }}
}}

If no notable interactions, knowledge, or memes occurred, set those arrays to [].
If [Self] did not avoid anything or show conditional reactions, set those arrays to [].
"memes_used" should capture: internet slang, in-group references, ironic phrases, exaggerated expressions, copypasta fragments, reaction catchphrases, or any culturally-loaded shorthand that would be hard for a language model to reproduce without examples.

IMPORTANT: Always include person_uid from the message annotations (e.g. user7658940875). Do NOT guess or fabricate uids.

---

{chunk_text}"""


TIER1_CHANNEL = """Analyze the following CHANNEL post(s) by [Self]. This is content published to a channel/audience.

Extract the following as JSON:
{{
  "language": "zh|en|mixed",
  "tone": "neutral|playful|serious|frustrated|enthusiastic|sarcastic|warm|cold",
  "formality": 1-5,
  "topics": ["brief topic keywords"],
  "style_markers": ["notable writing patterns"],
  "notable_phrases": ["exact quotes, max 5"],
  "knowledge_shared": [
    {{
      "domain": "topic area",
      "content": "what was shared",
      "depth": "expert|intermediate|casual",
      "original_text": "exact quote, max 100 chars"
    }}
  ],
  "memes_used": [
    {{
      "expression": "meme phrase used",
      "context": "context of usage",
      "meaning": "intended meaning",
      "effect": "social function"
    }}
  ],
  "negative_markers": [],
  "conditional_reactions": [],
  "message_structure": {{
    "avg_length": "short(<30chars)|medium(30-100)|long(>100)",
    "multi_message": true or false,
    "uses_code_blocks": true or false,
    "uses_links": true or false
  }}
}}

---

{chunk_text}"""


# ============================================================
# Tier-1.5: Per-chat summarization (hierarchical)
# ============================================================

TIER1_5_SYSTEM = """You are a conversation analyst synthesizing multiple annotation records from the same chat into a single coherent summary. Preserve all important details while reducing redundancy.

You MUST respond with valid JSON only."""

TIER1_5_SUMMARY = """Summarize these {annotation_count} Tier-1 annotations from chat "{chat_name}" ({chat_category}) into a single concise summary.

Preserve:
- All unique topics, style markers, notable phrases
- All person_interactions with their uids (merge duplicates, keep the richest description)
- All knowledge_shared entries (deduplicate)
- All memes_used entries (deduplicate)
- Aggregated tone/formality patterns (note if they vary)
- All negative_markers (merge into recurring patterns)
- All conditional_reactions (merge similar triggers, note frequency)

## Annotations
{annotations_text}

---

Respond with this JSON:
{{
  "chat_name": "{chat_name}",
  "chat_category": "{chat_category}",
  "annotation_count": {annotation_count},
  "language_distribution": {{"zh": N, "en": N, "mixed": N}},
  "dominant_tone": "most common tone",
  "tone_range": ["all tones observed"],
  "avg_formality": 1-5,
  "topics": ["all unique topics"],
  "style_markers": ["all unique style markers"],
  "group_role": "overall role if group chat, or null",
  "notable_phrases": ["top 10 most distinctive phrases"],
  "person_interactions_summary": [
    {{
      "person_name": "name",
      "person_uid": "uid",
      "interaction_count": N,
      "dominant_type": "most common interaction type",
      "dominant_tone": "most common tone",
      "summary": "1-2 sentence summary of the relationship pattern"
    }}
  ],
  "knowledge_summary": [
    {{
      "domain": "topic",
      "depth": "expert|intermediate|casual",
      "key_points": ["main knowledge items shared"],
      "best_quote": "most representative quote"
    }}
  ],
  "memes_summary": [
    {{
      "expression": "phrase",
      "usage_count": N,
      "typical_context": "when used"
    }}
  ],
  "negative_patterns": ["recurring things [Self] avoids or refuses to engage with in this chat"],
  "conditional_reactions_summary": [
    {{
      "trigger_type": "category of trigger (e.g. tech_question, humor, conflict, request_for_help)",
      "typical_reaction": "how they usually react to this trigger",
      "count": N,
      "tone": "typical tone in response"
    }}
  ]
}}"""


# ============================================================
# Tier-2: Identity model synthesis
# ============================================================

TIER2_IDENTITY_SYSTEM = """You are a personality psychologist building an executable identity model from chat behavior data. Your output must be structured for machine consumption — specific, categorical, and grounded in evidence. No vague narratives.

You MUST respond with valid JSON only."""

TIER2_IDENTITY_USER = """Based on the following data from user "{user_name}", build an IDENTITY MODEL.

Data sources: {chat_count} different chats ({private_count} private, {group_count} group, {channel_count} channel).

## Behavioral Statistics (computed from raw data — these are exact, not LLM-inferred)
{behavioral_stats}

## Per-Chat Summaries
{summaries_text}

---

IMPORTANT: The behavioral statistics are precise and should be weighted heavily. Note how behavior DIFFERS across contexts (private vs. group, with different people).

Respond with this JSON structure:
{{
  "big_five": {{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "neuroticism": 0.0-1.0
  }},
  "big_five_reasoning": "Brief justification for each score, referencing specific evidence",
  "core_style": {{
    "sentence_length": "short|medium|long|mixed",
    "punctuation_habits": ["specific patterns: e.g. rarely uses periods, heavy ellipsis, no exclamation marks"],
    "paragraph_style": "single_line_per_thought|multi_sentence_blocks|mixed",
    "rhythm": "fast_short_bursts|measured_paragraphs|variable",
    "emoji_usage": "frequent|moderate|rare|never",
    "formality_baseline": 1-5,
    "language_preference": "description of primary language and when they switch"
  }},
  "context_styles": {{
    "private_casual": {{
      "formality_shift": -2 to +2 from baseline,
      "humor_level": "high|medium|low|none",
      "sentence_length": "very_short|short|medium|long",
      "tone": "dominant tone in this context",
      "notable_differences": "what changes vs. baseline"
    }},
    "private_serious": {{
      "formality_shift": -2 to +2,
      "humor_level": "high|medium|low|none",
      "sentence_length": "very_short|short|medium|long",
      "tone": "dominant tone",
      "notable_differences": "what changes"
    }},
    "group_casual": {{
      "formality_shift": -2 to +2,
      "humor_level": "high|medium|low|none",
      "sentence_length": "very_short|short|medium|long",
      "tone": "dominant tone",
      "notable_differences": "what changes"
    }},
    "group_technical": {{
      "formality_shift": -2 to +2,
      "humor_level": "high|medium|low|none",
      "sentence_length": "very_short|short|medium|long",
      "tone": "dominant tone",
      "notable_differences": "what changes"
    }},
    "channel": {{
      "formality_shift": -2 to +2,
      "humor_level": "high|medium|low|none",
      "sentence_length": "very_short|short|medium|long",
      "tone": "dominant tone",
      "notable_differences": "what changes"
    }}
  }},
  "vocabulary": {{
    "high_freq_words": ["frequently used words/phrases"],
    "signature_phrases": ["distinctive expressions unique to this person"],
    "avoided_patterns": ["things this person never or rarely does/says"]
  }}
}}"""


# ============================================================
# Tier-2: Style rules (depends on identity_model)
# ============================================================

TIER2_STYLE_RULES_SYSTEM = """You are a behavioral rule extractor. You convert observed chat behavior into precise, executable rules that an AI agent can follow to mimic a person's communication style. Each rule must be specific, testable, and grounded in evidence.

You MUST respond with valid JSON only."""

TIER2_STYLE_RULES_USER = """Based on the following data from user "{user_name}", extract EXECUTABLE STYLE RULES.

## Core Style Baseline
{identity_summary}

## Behavioral Statistics
{behavioral_stats}

## Per-Chat Summaries (with negative patterns and conditional reactions)
{summaries_text}

---

Extract rules in three categories:
1. **rules**: Positive patterns — things this person DOES, with conditions
2. **boundaries**: Negative patterns — things this person does NOT do
3. **fallback_rules**: What to do when uncertain

Each rule must include confidence, evidence count, example quotes, and context applicability.

Respond with this JSON structure:
{{
  "rules": [
    {{
      "id": "rule_NNN",
      "rule": "specific, actionable instruction (e.g. 'Send messages under 30 characters in casual contexts')",
      "type": "sentence_structure|tone|language_choice|punctuation|emoji|topic_engagement|formality|multi_message|code_style|meme_usage",
      "confidence": 0.0-1.0,
      "evidence_count": N,
      "example_quotes": ["exact quotes from data, max 3"],
      "applies_in": ["private_casual", "group_casual", etc.],
      "not_applies_in": ["contexts where this rule does NOT apply"],
      "priority": "high|medium|low"
    }}
  ],
  "boundaries": [
    {{
      "id": "boundary_NNN",
      "description": "what this person does NOT do (e.g. 'Never uses formal greetings like 您好')",
      "type": "expression|tone|topic|behavior",
      "confidence": 0.0-1.0,
      "evidence_count": N,
      "applies_in": ["all" or specific contexts],
      "not_applies_in": ["exceptions where this boundary doesn't hold"]
    }}
  ],
  "fallback_rules": [
    {{
      "condition": "uncertain_about_topic|unfamiliar_person|ambiguous_tone|new_group",
      "action": "what to do (e.g. 'admit ignorance directly', 'mirror the other person's formality')",
      "style": "how to do it",
      "confidence": 0.0-1.0,
      "example_quotes": ["examples if available"]
    }}
  ]
}}

IMPORTANT:
- Generate at least 15 rules, 5 boundaries, and 3 fallback rules
- Rules must be SPECIFIC and TESTABLE, not vague (bad: "be friendly"; good: "use 哈哈 or emoji when agreeing with friends")
- Confidence should reflect how consistently this pattern appears in the data
- Every rule must reference at least one example quote from the actual data"""


# ============================================================
# Tier-2: Topic graph
# ============================================================

TIER2_TOPIC_GRAPH_SYSTEM = """You are a knowledge and topic analyst. You build structured topic profiles showing what someone knows, discusses, and avoids — with expression style per topic.

You MUST respond with valid JSON only."""

TIER2_TOPIC_GRAPH_USER = """Based on the following knowledge and topic data from "{user_name}" across {chunk_count} entries, build a TOPIC GRAPH.

## Knowledge Records
{knowledge_text}

## Topic Distribution from Summaries
{topics_text}

---

Classify each topic as: expertise (they teach/advise), discussion (they engage but don't teach), or avoided (they deflect or stay silent).

Respond with this JSON structure:
{{
  "domains": [
    {{
      "name": "domain name",
      "type": "expertise|discussion|avoided",
      "expertise_level": "expert|advanced|intermediate|beginner|null",
      "sub_topics": ["specific sub-areas"],
      "expression_style": "advising|explaining|complaining|sharing|forwarding|debating|joking|null",
      "confidence": 0.0-1.0,
      "evidence_count": N,
      "key_knowledge": ["concrete pieces of demonstrated knowledge"],
      "representative_quotes": ["1-3 exact quotes, max 150 chars each"],
      "note": "optional note for avoided topics or special patterns"
    }}
  ],
  "overall_profile": "1-2 paragraph summary of expertise landscape and topic engagement patterns"
}}"""


# ============================================================
# Tier-2: Response policies
# ============================================================

TIER2_RESPONSE_POLICIES_SYSTEM = """You are a behavioral response analyst. You model how a person reacts to different types of conversational triggers, and what boundaries they maintain. Your output must be specific enough for an AI agent to execute.

You MUST respond with valid JSON only."""

TIER2_RESPONSE_POLICIES_USER = """Based on the following data from "{user_name}", build a RESPONSE POLICY model.

## Per-Chat Summaries (focus on conditional_reactions_summary and negative_patterns)
{summaries_text}

## Behavioral Statistics
{behavioral_stats}

---

Model two things:
1. **triggers**: How this person reacts to different conversational situations
2. **boundaries**: What this person will NOT do in specific contexts

Respond with this JSON structure:
{{
  "triggers": [
    {{
      "condition": "specific situation (e.g. 'someone asks a technical question in their expertise area')",
      "response_style": "how they respond (e.g. 'detailed explanation with code examples, patient tone')",
      "tone": "specific tone used",
      "message_structure": "short_reply|detailed_paragraph|multi_message|link_share",
      "confidence": 0.0-1.0,
      "evidence_count": N,
      "example_quotes": ["actual examples from data"]
    }}
  ],
  "boundaries": [
    {{
      "situation": "context where boundary applies (e.g. 'unfamiliar people', 'formal channels')",
      "constraint": "what they won't do (e.g. 'won't use profanity', 'won't share personal opinions on politics')",
      "confidence": 0.0-1.0,
      "evidence_count": N,
      "exceptions": "any known exceptions to this boundary"
    }}
  ]
}}

IMPORTANT: Generate at least 8 triggers and 5 boundaries. Be specific — "responds helpfully" is too vague; "gives step-by-step instructions with terminal commands" is good."""


# ============================================================
# Tier-2: Private chat relationship
# ============================================================

TIER2_RELATIONSHIP_SYSTEM = """You are a social relationship analyst. You analyze chat patterns to understand how a person relates to specific contacts.

You MUST respond with valid JSON only."""

TIER2_PRIVATE_RELATIONSHIP_USER = """Analyze user "{user_name}"'s relationship with "{contact_name}" (uid: {contact_uid}) based on these PRIVATE CHAT annotations.

This is a 1-on-1 direct conversation. Number of conversation segments: {segment_count}

## Annotations
{annotations_text}

---

Respond with this JSON structure:
{{
  "relationship_type": "close_friend|friend|acquaintance|colleague|family|mentor|mentee",
  "intimacy_score": 0.0-1.0,
  "interaction_style": "Description of how {user_name} interacts with {contact_name}",
  "tone_adjustments": "How {user_name}'s tone differs in this relationship vs. baseline",
  "typical_topics": ["What they usually discuss"],
  "dynamic": "Who initiates more, power dynamics, emotional balance",
  "summary": "1-2 sentence summary",
  "source": "private_chat"
}}"""


# ============================================================
# Tier-2: Group-extracted individual relationship
# ============================================================

TIER2_GROUP_RELATIONSHIP_USER = """Analyze user "{user_name}"'s relationship with "{contact_name}" (uid: {contact_uid}) based on their interactions observed in GROUP CHATS.

These are public interactions within shared groups, NOT private conversations.

Groups where they interacted: {groups}
Total interaction records: {interaction_count}

## Interaction Records
{interactions_text}

---

Respond with this JSON structure:
{{
  "relationship_type": "close_friend|friend|acquaintance|colleague|community_peer|rival|mentor|mentee",
  "intimacy_score": 0.0-1.0,
  "interaction_style": "How {user_name} interacts with {contact_name} in groups",
  "tone_adjustments": "Tone differences when addressing {contact_name} vs. others",
  "typical_topics": ["Topics they engage on together"],
  "dynamic": "Power dynamics, who helps whom, who teases whom",
  "summary": "1-2 sentence summary as observed in groups",
  "source": "group_chat",
  "groups_observed": {groups_json}
}}"""


# ============================================================
# Tier-2: Group profile (user's behavior within a group)
# ============================================================

TIER2_GROUP_PROFILE_SYSTEM = """You are a community behavior analyst. You analyze how a person behaves within a specific group or community.

You MUST respond with valid JSON only."""

TIER2_GROUP_PROFILE_USER = """Analyze how "{user_name}" behaves in the group "{group_name}".

Chat type: {chat_type}
Total annotation segments: {segment_count}

## Per-chat summary
{summary_text}

---

Describe this person's role, style, and behavior within THIS specific group.

Respond with this JSON structure:
{{
  "role": "core_member|active_contributor|helper|entertainer|lurker|leader|moderator",
  "activity_level": "high|medium|low",
  "contribution_style": "what they typically contribute (technical help, banter, news sharing, etc.)",
  "tone_in_group": "description of their typical tone here",
  "typical_topics": ["what they usually talk about in this group"],
  "notable_relationships": ["names of people they interact with most, with brief description"],
  "summary": "2-3 sentence description of their presence in this group"
}}"""


# ============================================================
# Tier-2: Meme/slang behavior model
# ============================================================

TIER2_MEME_SYSTEM = """You are a cultural linguist specializing in internet slang, memes, and in-group language. You analyze HOW a person uses memes — their patterns, timing, and creative process — not just WHICH memes they use.

You MUST respond with valid JSON only."""

TIER2_MEME_USER = """Based on the following meme/slang usage records from "{user_name}" across {record_count} instances, build a MEME BEHAVIOR MODEL.

IMPORTANT: Memes are ephemeral — specific phrases go viral and die within days or weeks. Do NOT focus on cataloging which exact memes this person has used. Instead, focus on:
1. HOW they use memes (timing, placement, creative transformation)
2. WHEN they reach for memes vs. plain language
3. What STRUCTURAL PATTERNS they follow when adapting or creating meme expressions
4. How they LEARN and ADOPT new memes from conversations (do they echo others? remix? one-up?)

## Meme Usage Records
{memes_text}

---

Respond with this JSON structure:
{{
  "meme_timing": {{
    "triggers": ["situations/emotions that trigger meme usage"],
    "anti_triggers": ["situations where they stay serious"],
    "placement": "where memes appear in message flow"
  }},
  "meme_grammar": [
    {{
      "pattern_name": "descriptive name",
      "structure": "abstract template or grammatical pattern",
      "when_to_use": "conversational moment that fits",
      "examples_from_data": ["2-3 actual examples"]
    }}
  ],
  "meme_adoption_style": {{
    "how_they_learn": "how they pick up new expressions",
    "transformation_tendency": "verbatim vs. modified vs. original variations",
    "source_sensitivity": "who they adopt memes from"
  }},
  "known_meme_vocabulary": [
    {{
      "phrase": "frequently used expression",
      "meaning": "what it conveys",
      "social_function": "humor|deflection|bonding|complaint|celebration|sarcasm|self_deprecation",
      "frequency": "high|medium|low"
    }}
  ],
  "meme_style_summary": "2-3 paragraph description of meme personality and agent guidance"
}}"""


# ============================================================
# Prompt version hash (for cache invalidation)
# ============================================================

def prompt_version() -> str:
    """Return a hash of all prompt templates for cache keying."""
    import hashlib
    all_prompts = (
        TIER1_SYSTEM + TIER1_PERSONAL_CHAT + TIER1_GROUP_CHAT + TIER1_CHANNEL
        + TIER1_5_SYSTEM + TIER1_5_SUMMARY
        + TIER2_IDENTITY_SYSTEM + TIER2_IDENTITY_USER
        + TIER2_STYLE_RULES_SYSTEM + TIER2_STYLE_RULES_USER
        + TIER2_TOPIC_GRAPH_SYSTEM + TIER2_TOPIC_GRAPH_USER
        + TIER2_RESPONSE_POLICIES_SYSTEM + TIER2_RESPONSE_POLICIES_USER
        + TIER2_RELATIONSHIP_SYSTEM + TIER2_PRIVATE_RELATIONSHIP_USER
        + TIER2_GROUP_RELATIONSHIP_USER
        + TIER2_GROUP_PROFILE_SYSTEM + TIER2_GROUP_PROFILE_USER
        + TIER2_MEME_SYSTEM + TIER2_MEME_USER
    )
    return hashlib.sha256(all_prompts.encode()).hexdigest()[:8]
