# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/telegram_persona/`. `cli.py` is the entry point, `analyzer.py` drives LLM analysis, `parser.py` and `chunker.py` prepare Telegram exports, and `persona.py` plus `prompts.py` assemble the final outputs. Jinja templates for generated prompts live in `templates/`. Repository-level documentation is in `README.md`.

## Build, Test, and Development Commands
Set up a local environment with `python -m venv .venv && source .venv/bin/activate`, then install the package with `pip install -e .`. Run the CLI as `telegram-persona path/to/result.json -o output/` or `python -m telegram_persona path/to/result.json`. Use `telegram-persona --render-only -o output/` to regenerate prompt files without API calls, and `telegram-persona path/to/result.json --estimate-cost` to preview work before a full run.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow the existing code style: 4-space indentation, type hints where useful, `snake_case` for functions and variables, and short module docstrings. Keep modules focused on one stage of the pipeline. Prefer `Path` over raw string paths and preserve the current CLI flag naming style such as `--only-tier2` and `--estimate-cost`. No formatter or linter is configured in `pyproject.toml`, so keep changes consistent with surrounding code.

## Testing Guidelines
There is no committed `tests/` suite yet. Before opening a PR, run a small end-to-end check with `telegram-persona path/to/result.json --sample 10` and validate generated files in `output/`. For prompt-only changes, verify `--render-only` still produces `core_prompt.md` and `system_prompt.md`. If you add automated tests, place them under `tests/` and use `test_*.py` naming.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects such as `Add README` and `Initial commit: telegram-persona`. Keep commits focused and descriptive, preferably under 72 characters. PRs should explain the user-visible change, note any CLI or output format impact, and include example commands or sample output when behavior changes.

## Configuration & Security
Pass secrets through `.env` or CLI flags, not committed files. `OPENAI_API_KEY`, model names, and base URLs should stay local. Avoid checking in Telegram export data, generated `output/` artifacts, or cache contents from `output/.cache/`.
