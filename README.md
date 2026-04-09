# telegram-persona

Extract an executable persona model from Telegram chat exports. Turns your messaging history into structured, machine-readable rules that an AI agent can use to mimic your communication style.

## What it does

```
Telegram Desktop Export (result.json)
        |
        v
  Tier-1: Per-chunk annotation (concurrent LLM calls)
        |
        v
  Tier-1.5: Per-chat summarization
        |
        v
  Tier-2 Phase A: Identity model, contacts, groups, topics, memes (parallel)
        |
        v
  Tier-2 Phase B: Style rules, response policies (depends on Phase A)
        |
        v
  Structured JSON outputs + System prompts
```

## Output files

| File | Contents |
|------|----------|
| `identity_model.json` | Core style, Big Five, context-dependent styles, vocabulary |
| `style_rules.json` | Executable rules with confidence, evidence, and context applicability |
| `contact_profiles.json` | Per-contact relationship and tone adjustment profiles |
| `group_profiles.json` | User's role and behavior within each group |
| `topic_graph.json` | Topics by type (expertise / discussion / avoided) with expression style |
| `response_policies.json` | Conditional triggers and behavioral boundaries |
| `memes.json` | Meme timing, grammar patterns, adoption style |
| `behavior_stats.json` | Quantitative behavioral statistics (non-LLM) |
| `style_examples.json` | Notable phrases grouped by chat |
| `core_prompt.md` | Compact system prompt for agent use (~100 lines) |
| `system_prompt.md` | Full system prompt with all data (reference) |

## Install

```bash
git clone https://github.com/kotodamai/telegram-persona.git
cd telegram-persona
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

```bash
cp .env.example .env
# Edit .env with your API key and model preferences
```

Works with any OpenAI-compatible API endpoint.

## Usage

### Export your Telegram data

1. Open Telegram Desktop
2. Settings > Advanced > Export Telegram data
3. Select JSON format, include all chats you want analyzed
4. Wait for export, note the path to `result.json`

### Run analysis

```bash
# Full run
telegram-persona path/to/result.json -o output/

# Only run Tier-1 annotation (useful for large datasets)
telegram-persona path/to/result.json --only-tier1

# Reuse cached Tier-1, only re-run Tier-2 modeling
telegram-persona path/to/result.json --only-tier2

# Re-render prompts from existing JSON (no API calls)
telegram-persona --render-only -o output/

# Estimate cost before running
telegram-persona path/to/result.json --estimate-cost

# Test with a small sample
telegram-persona path/to/result.json --sample 10
```

### Filtering

```bash
# Date range
telegram-persona result.json --since 2024-01-01 --until 2024-12-31

# Specific chats only
telegram-persona result.json --include-chat "Alice" --include-chat "Work Group"

# Exclude chats
telegram-persona result.json --exclude-chat "Spam Group"

# Skip low-activity chats
telegram-persona result.json --min-messages 10
```

### CLI options

```
positional arguments:
  export_path           Path to Telegram result.json

options:
  -o, --output          Output directory (default: output/)
  --tier1-model         Model for Tier-1 annotation
  --tier2-model         Model for Tier-2 distillation
  --base-url            OpenAI-compatible API base URL
  --api-key             API key
  --concurrency         Concurrent API calls for Tier-1 (default: 5)
  --api-timeout         Per-request timeout in seconds (default: 180)
  --api-max-retries     Max retries per API call (default: 3)
  --only-tier1          Only run Tier-1 annotation
  --only-tier2          Skip Tier-1, reuse cache
  --render-only         Re-render prompts from existing JSON
  --estimate-cost       Dry run: estimate token usage
  --clear-cache         Clear cached results before running
  --sample N            Only process N random chunks
  --since/--until       Date range filter (YYYY-MM-DD)
  --include-chat/--exclude-chat   Chat name filter (repeatable)
  --min-messages N      Skip chats with fewer user messages
```

## How to use the output

The `core_prompt.md` is a compact system prompt meant to be loaded into an AI agent as its base persona. It contains only the essential style rules, vocabulary, and boundaries.

For context-specific data (contacts, groups, topics), load from the JSON files on demand:

```python
import json

# Always loaded as system prompt
core_prompt = open("output/core_prompt.md").read()

# Injected when talking to a specific person
contacts = json.load(open("output/contact_profiles.json"))
if contact_name in contacts:
    context += format_contact(contacts[contact_name])

# Injected when in a specific group
groups = json.load(open("output/group_profiles.json"))
if group_name in groups:
    context += format_group(groups[group_name])
```

## Caching

All LLM calls are cached in `output/.cache/`. Cache keys include content hash, model name, prompt version, and source file hash. Changing prompts or input data automatically invalidates relevant caches.

## License

[MIT](LICENSE)
