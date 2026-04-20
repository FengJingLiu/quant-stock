"""
Tushare Pro 数据访问层

Token 和 API 地址集中管理，其他模块通过 get_pro() 获取已配置的 pro 实例。
"""

from __future__ import annotations

from typing import Any

from src.data_clients import get_tushare_pro


def get_pro() -> Any:
    """返回已配置的 tushare pro 实例（单例）。"""
    return get_tushare_pro()
