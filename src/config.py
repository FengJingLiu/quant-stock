"""
集中管理所有外部服务配置，统一从环境变量读取。

使用方式:
    from src.config import CH_KWARGS, TUSHARE_TOKEN, AKSHARE_PROXY_HOST, AKSHARE_PROXY_TOKEN

环境变量优先，未设置时报错（不允许硬编码 fallback）。
可通过项目根目录 .env 文件自动加载。
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 自动加载 .env ──────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise EnvironmentError(
            f"环境变量 {name} 未设置。请在 .env 文件或 shell 中配置，参考 .env.example"
        )
    return val


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# ── ClickHouse ─────────────────────────────────────────────────────────
CH_HOST = _get("CH_HOST", "localhost")
CH_HTTP_PORT = int(_get("CH_HTTP_PORT", "8123"))
CH_NATIVE_PORT = int(_get("CH_NATIVE_PORT", "9000"))
CH_USER = _get("CH_USER", "default")
CH_PASSWORD = _require("CH_PASSWORD")
CH_DATABASE = _get("CH_DATABASE", "astock")

CH_HTTP_KWARGS = dict(
    host=CH_HOST,
    port=CH_HTTP_PORT,
    username=CH_USER,
    password=CH_PASSWORD,
    database=CH_DATABASE,
)

CH_NATIVE_KWARGS = dict(
    host=CH_HOST,
    port=CH_NATIVE_PORT,
    user=CH_USER,
    password=CH_PASSWORD,
    compression="lz4",
)

# ── Tushare Pro ────────────────────────────────────────────────────────
TUSHARE_TOKEN = _require("TUSHARE_TOKEN")
TUSHARE_API_URL = _get("TUSHARE_API_URL", "http://api.tushare.pro")

# ── AKShare Proxy ──────────────────────────────────────────────────────
AKSHARE_PROXY_HOST = _require("AKSHARE_PROXY_HOST")
AKSHARE_PROXY_TOKEN = _require("AKSHARE_PROXY_TOKEN")
