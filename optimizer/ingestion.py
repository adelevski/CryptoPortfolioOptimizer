import numpy as np
import yfinance as yf


def get_data(assets, start, end, log):
    asset_data = yf.download(assets, start=start, end=end)
    asset_data = asset_data['Adj Close']
    delta_days = len(asset_data)
    if log:
        returns = np.log(asset_data / asset_data.shift(1))    
    else:
        returns = asset_data.pct_change()                    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix, delta_days