"""集中读取环境变量。仓库内零密钥——所有敏感信息从 .env / 环境变量注入。"""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file():
    """自动加载项目根 .env；不覆盖已有环境变量（export / CI 优先）。

    这样 `uvicorn app.main:app --reload` 直接生效，不必每次加 --env-file；
    python-dotenv 是可选依赖，缺失时静默退化为纯环境变量模式。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


def _env(*names, default=None):
    """按顺序取第一个非空环境变量。"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass
class Settings:
    base_url: str
    api_key: str
    model: str
    max_rounds: int
    max_history_messages: int
    recent_keep: int
    db_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file()
        return cls(
            base_url=_env("LLM_BASE_URL", "OPENAI_BASE_URL",
                          default="https://api.deepseek.com/v1"),
            api_key=_env("LLM_API_KEY", "OPENAI_API_KEY", default=""),
            model=_env("LLM_MODEL", default="deepseek-chat"),
            max_rounds=int(_env("MAX_ROUNDS", default="8")),
            max_history_messages=int(_env("MAX_HISTORY_MESSAGES", default="40")),
            recent_keep=int(_env("RECENT_KEEP", default="20")),
            db_path=_env("DB_PATH", default="data/agent.db"),
        )

    def validate_llm(self) -> None:
        """真实调用 LLM 前校验配置缺失，给出明确报错（而不是跑到一半才挂）。"""
        missing = []
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.model:
            missing.append("LLM_MODEL")
        if missing:
            raise RuntimeError(
                f"缺少必需的环境变量: {', '.join(missing)}。"
                f"请复制 .env.example 为 .env 并填入（或直接 export），"
                f"密钥一律走环境变量，禁止写入代码。"
            )
