from __future__ import annotations

import types
import unittest
from unittest import mock

import pyarrow as pa

from src import data_clients


class TestDataClients(unittest.TestCase):
    def setUp(self) -> None:
        data_clients._tushare_pro = None
        data_clients._tushare_key = None
        data_clients._akshare_module = None
        data_clients._akshare_key = None
        data_clients._akshare_patch_key = None

    def test_create_clickhouse_http_client_merges_overrides(self) -> None:
        with mock.patch.object(
            data_clients,
            "CH_HTTP_KWARGS",
            {"host": "localhost", "port": 8123, "username": "default"},
        ), mock.patch.object(data_clients.clickhouse_connect, "get_client", return_value="client") as get_client:
            client = data_clients.create_clickhouse_http_client({"port": 9000, "database": "astock"})

        self.assertEqual(client, "client")
        get_client.assert_called_once_with(
            host="localhost",
            port=9000,
            username="default",
            database="astock",
        )

    def test_query_clickhouse_arrow_df_uses_existing_client(self) -> None:
        fake_client = mock.Mock()
        fake_client.query_arrow.return_value = pa.table({"symbol": ["510300.SH"], "close": [4.2]})

        df = data_clients.query_clickhouse_arrow_df(
            "SELECT symbol, close FROM t",
            parameters={"sym": "510300.SH"},
            client=fake_client,
        )

        fake_client.query_arrow.assert_called_once_with(
            "SELECT symbol, close FROM t",
            parameters={"sym": "510300.SH"},
        )
        self.assertEqual(df["symbol"].to_list(), ["510300.SH"])
        self.assertEqual(df["close"].to_list(), [4.2])

    def test_query_clickhouse_rows_and_scalar_use_existing_client(self) -> None:
        fake_client = mock.Mock()
        fake_client.query.side_effect = [
            types.SimpleNamespace(result_rows=[("510300.SH", 4.2)]),
            types.SimpleNamespace(result_rows=[(4.2,)]),
        ]

        rows = data_clients.query_clickhouse_rows(
            "SELECT symbol, close FROM t",
            parameters={"sym": "510300.SH"},
            client=fake_client,
        )
        scalar = data_clients.query_clickhouse_scalar(
            "SELECT close FROM t",
            parameters={"sym": "510300.SH"},
            client=fake_client,
        )

        self.assertEqual(rows, [("510300.SH", 4.2)])
        self.assertEqual(scalar, 4.2)
        self.assertEqual(fake_client.query.call_count, 2)

    def test_get_tushare_pro_is_singleton_per_token_and_url(self) -> None:
        fake_pro1 = types.SimpleNamespace()
        fake_pro2 = types.SimpleNamespace()

        with mock.patch.object(
            data_clients.ts,
            "pro_api",
            side_effect=[fake_pro1, fake_pro2],
        ) as pro_api:
            first = data_clients.get_tushare_pro(token="t1", api_url="http://api.example")
            second = data_clients.get_tushare_pro(token="t1", api_url="http://api.example")
            third = data_clients.get_tushare_pro(token="t2", api_url="http://api.example")

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(getattr(first, "_DataApi__http_url"), "http://api.example")
        self.assertEqual(pro_api.call_count, 2)

    def test_get_akshare_installs_proxy_patch_once_per_key(self) -> None:
        fake_patch = mock.Mock()
        fake_akshare = mock.Mock()

        def fake_import(name: str):
            if name == "akshare_proxy_patch":
                return fake_patch
            if name == "akshare":
                return fake_akshare
            raise AssertionError(name)

        with mock.patch.object(data_clients.importlib, "import_module", side_effect=fake_import) as import_module:
            first = data_clients.get_akshare(proxy_host="proxy", token="token", retry=30)
            second = data_clients.get_akshare(proxy_host="proxy", token="token", retry=30)
            third = data_clients.get_akshare(proxy_host="proxy", token="token", retry=31)

        self.assertIs(first, fake_akshare)
        self.assertIs(second, fake_akshare)
        self.assertIs(third, fake_akshare)
        self.assertEqual(fake_patch.install_patch.call_count, 2)
        fake_patch.install_patch.assert_any_call("proxy", "token", retry=30)
        fake_patch.install_patch.assert_any_call("proxy", "token", retry=31)
        self.assertEqual(import_module.call_count, 4)


if __name__ == "__main__":
    unittest.main()
