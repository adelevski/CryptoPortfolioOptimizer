import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import scipy.optimize as sc

# Importing data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']

    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Mean returns and standard deviations
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*365
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(365)
    return returns, std

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result 

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minVariance(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    "Minimize the portfolio variance by altering the weights/allocation of assets in the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result



####  Script
stocks = ['BTC-USD', 'ETH-USD', 'ADA-USD']

# Time (2 years back)
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

weights = np.array([0.3, 0.3, 0.4])

meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

result = maxSR(meanReturns, covMatrix)
maxSR, maxWeights = result['fun'], result['x']
print(round(maxSR, 3), [round(x, 3) for x in maxWeights])

minVarResult = minVariance(meanReturns, covMatrix)
minVar, minVarWeights = minVarResult['fun'], minVarResult['x']
print(round(minVar, 3), [round(x, 3) for x in minVarWeights])




