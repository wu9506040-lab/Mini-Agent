"""安全防呆：仓库代码中绝不出现 LLM API Key 字面量（题目红线）。"""
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


def test_no_api_key_literals_in_source():
    offenders = []
    for py in APP_DIR.rglob("*.py"):
        content = py.read_text(encoding="utf-8")
        if KEY_PATTERN.search(content):
            offenders.append(str(py))
    assert not offenders, f"源码中发现疑似 API Key: {offenders}"


def test_no_hardcoded_key_env_names_assigned():
    """禁止在源码里直接赋值密钥类变量（只允许从环境变量读取）。"""
    banned = re.compile(r'^(?!.*getenv).*api_key\s*=\s*["\'].+["\']', re.MULTILINE)
    for py in APP_DIR.rglob("*.py"):
        content = py.read_text(encoding="utf-8")
        assert not banned.search(content), f"{py} 中出现疑似硬编码密钥"
