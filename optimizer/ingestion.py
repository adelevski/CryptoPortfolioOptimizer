# Functional imports
import numpy as np
import yfinance as yf

# Imports for type hints
import pandas as pd
import datetime as dt
from typing import List, Tuple


def get_data(
    assets: List[str], 
    start: dt.datetime, 
    end: dt.datetime, 
    log: bool
) -> Tuple[pd.core.series.Series, pd.core.frame.DataFrame, int]:
    """
    Fetches pricing data for the given assets from yahoo finance using the
    yfinance package, (for the provided time period start -> end), then 
    calculates and returns the mean returns, covariance matrix, and
    the length of the time period.

    Notes:
      - If fetching crypto assets, yfinance requires '-USD' be attached to the
        ticker, and also some crypto have unconventional namings, with numbers
        attached to the end of the ticker, e.g. SOL1, DOT1, GRT2, etc...
    """
    print("Downloading historical price data...")
    asset_data = yf.download(assets, start=start, end=end)
    asset_data = asset_data['Adj Close'] # Takes care of stock splits/etc...
    delta_days = len(asset_data)
    if log:
        returns = np.log(asset_data / asset_data.shift(1)) # log returns    
    else:
        returns = asset_data.pct_change()                  # simple returns  
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return (mean_returns, cov_matrix, delta_days)