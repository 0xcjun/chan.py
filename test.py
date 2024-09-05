import ccxt
import datetime
import time

# exchange = ccxt.binance({
#     'proxies': {
#         'http': 'http://127.0.0.1:53902',
#         'https': 'http://127.0.0.1:53902',
#     }
# })

# # 尝试获取市场数据
# markets = exchange.fetch_ohlcv( symbol='BTCUSDT',
#             timeframe='4h',
#             limit=1500)

timestamp = time.time()
print(int(timestamp)*1000)

