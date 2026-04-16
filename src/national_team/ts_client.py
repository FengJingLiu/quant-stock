"""
Tushare Pro 数据访问层

Token 和 API 地址集中管理，其他模块通过 get_pro() 获取已配置的 pro 实例。
"""

from __future__ import annotations

import tushare as ts

_TUSHARE_TOKEN = "***TUSHARE_TOKEN***"
_TUSHARE_API_URL = "http://***TUSHARE_HOST***:8010/"

_pro: ts.pro.client.DataApi | None = None


def get_pro() -> ts.pro.client.DataApi:
    """返回已配置的 tushare pro 实例（单例）。"""
    global _pro
    if _pro is None:
        _pro = ts.pro_api(_TUSHARE_TOKEN)
        _pro._DataApi__http_url = _TUSHARE_API_URL
    return _pro
