import pandas as pd
from elastic_trader.envs.ipsa_env import IpsaTradingEnv

def test_ipsa_shape():
    data = pd.DataFrame({
        'Date': ['2020-01-01']*2,
        'Open': [1,1],
        'High': [1,1],
        'Low': [1,1],
        'Close': [1,1],
        'Volume': [1,1],
        'Ticker': ['AAA','BBB']
    })
    env = IpsaTradingEnv(data)
    obs, _ = env.reset()
    assert obs.shape[0] == len(env.tickers)*5 + 1
