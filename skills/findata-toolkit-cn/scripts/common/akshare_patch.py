"""
AKShare Proxy Patch Helper
自动应用代理补丁，确保所有 AKShare 请求通过代理服务器。

使用方法:
    from common import akshare_patch
    ak = akshare_patch.get_akshare()
"""

import os

# 默认配置
TOKEN = os.getenv("AKSHARE_PROXY_TOKEN", "")
PROXY_HOST = os.getenv("AKSHARE_PROXY_HOST", "")
RETRY = int(os.getenv("AKSHARE_PROXY_RETRY", "30"))


class _AkshareProxy:
    """单例模式，确保只安装一次补丁。"""

    _instance = None
    _patched = False
    _akshare = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def install(self):
        """安装 AKShare 代理补丁。"""
        if self._patched:
            return

        try:
            import akshare_proxy_patch

            akshare_proxy_patch.install_patch(PROXY_HOST, TOKEN, retry=RETRY)
            self._patched = True
        except Exception as e:
            print(f"[akshare_patch] Warning: Failed to install proxy patch: {e}")
            # 继续运行，不中断流程

    def get_akshare(self):
        """获取已安装补丁的 akshare 模块。"""
        self.install()
        if self._akshare is None:
            import akshare

            self._akshare = akshare
        return self._akshare


# 全局实例
_proxy = _AkshareProxy()


def install():
    """手动安装代理补丁（可选）。"""
    _proxy.install()


def get_akshare():
    """
    获取已安装代理补丁的 akshare 模块。

    Usage:
        from common import akshare_patch
        ak = akshare_patch.get_akshare()
        df = ak.stock_zh_a_spot_em()
    """
    return _proxy.get_akshare()
