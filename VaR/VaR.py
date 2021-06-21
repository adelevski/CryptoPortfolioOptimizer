import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr

# Importing Data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# Portfolio Performance
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(Time)
    return returns, std

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock+'.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)

returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns = returns.dropna()

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

returns['portfolio'] = returns.dot(weights)

# print(returns)


def historicalVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=5)
    
    else:
        raise TypeError("Expected returns to be dataframe or series")

# print(historicalVaR(returns['portfolio'], alpha=5))

def historicalCVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the CVaR for dataframe / series
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=5)
    
    else:
        raise TypeError("Expected returns to be dataframe or series")

# 1 day    
Time = 1

VaR = -historicalVaR(returns['portfolio'], alpha=5)*np.sqrt(Time)
CVaR = -historicalCVaR(returns['portfolio'], alpha=5)*np.sqrt(Time)
pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)


initialInvestment = 100000
print('Expected Portfolio Return: ', round(initialInvestment*pRet, 2))
print('Value at Risk 95th CI:     ', round(initialInvestment*VaR, 2))
print('Conditional VaR 95th CI:   ', round(initialInvestment*CVaR, 2))



