import numpy as np
import datetime as dt
from pandas_datareader import data as pdr


# Importing data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Adj Close']

    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Mean returns and standard deviations
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*365
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(365)
    return returns, std



####  Script
stocks = ['BTC-USD', 'ETH-USD', 'ADA-USD']

# Time (2 years back)
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=730)

weights = np.array([0.3, 0.3, 0.4])

meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

print(round(returns*100, 2), round(std*100, 2))




