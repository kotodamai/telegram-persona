from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    tier1_model: str = "gpt-5.4-mini"
    tier2_model: str = "gpt-5.4"
    concurrency: int = 5
    api_timeout: float = 180.0
    api_max_retries: int = 3
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> Config:
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", cls.base_url),
            api_key=os.getenv("OPENAI_API_KEY", cls.api_key),
            tier1_model=os.getenv("TIER1_MODEL", cls.tier1_model),
            tier2_model=os.getenv("TIER2_MODEL", cls.tier2_model),
            concurrency=int(os.getenv("CONCURRENCY", cls.concurrency)),
            api_timeout=float(os.getenv("API_TIMEOUT", cls.api_timeout)),
            api_max_retries=int(os.getenv("API_MAX_RETRIES", cls.api_max_retries)),
        )
