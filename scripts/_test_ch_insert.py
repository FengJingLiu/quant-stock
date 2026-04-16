"""临时验证脚本：测试修复后的 transform + insert 流程"""
import zipfile, io
import pandas as pd
from clickhouse_driver import Client

CHUNK = 200_000
COLS = ['symbol','trade_date','datetime','open','high','low','close','volume','amount','vwap']
zp = '/home/autumn/quant/stock/data/基金分钟数据/ETF分钟数据_汇总/ETF_1min_2005_2022.zip'

c = Client(host='localhost', port=9000, user='default', password='***CH_PASSWORD***', compression='lz4')
c.execute('TRUNCATE TABLE astock.klines_1m_etf')

def transform(df):
    df = df.copy()
    df['_dt'] = pd.to_datetime(df['时间'], errors='coerce')
    df = df.dropna(subset=['_dt','开盘价','收盘价','最高价','最低价']).reset_index(drop=True)
    vol = df['成交量'].astype('float64')
    amt = df['成交额'].astype('float64')
    vwap = (amt / vol.where(vol>0)).fillna(df['收盘价'].astype('float64'))
    out = pd.DataFrame({
        'symbol': df['代码'].astype(str),
        'trade_date': df['_dt'].dt.date,
        'datetime': df['_dt'].tolist(),
        'open': df['开盘价'].astype('float32'), 'high': df['最高价'].astype('float32'),
        'low':  df['最低价'].astype('float32'), 'close':df['收盘价'].astype('float32'),
        'volume':vol, 'amount':amt, 'vwap':vwap.astype('float32'),
    })
    return out[out['close']>0].reset_index(drop=True)

with zipfile.ZipFile(zp) as zf:
    for name in ['159001.SZ.csv','159003.SZ.csv','159005.SZ.csv','510050.SH.csv']:
        buf = io.BytesIO(zf.read(name))
        total = 0
        for chunk in pd.read_csv(buf, chunksize=CHUNK, low_memory=False):
            out = transform(chunk)
            data = [out[col].tolist() for col in COLS]
            c.execute(f'INSERT INTO astock.klines_1m_etf ({",".join(COLS)}) VALUES',
                      data, columnar=True, types_check=False)
            total += len(out)
        print(f'{name}: {total} rows OK')

cnt = c.execute('SELECT count() FROM astock.klines_1m_etf')[0][0]
sample = c.execute('SELECT symbol, trade_date, datetime, close, vwap FROM astock.klines_1m_etf ORDER BY trade_date LIMIT 3')
print(f'table total: {cnt:,}')
for row in sample:
    print(' ', row)

# 清理测试数据
c.execute('TRUNCATE TABLE astock.klines_1m_etf')
print('truncated OK — ready for full load')
