import datetime as dt
from efFuncs import getData, EF_graph



stocks = ['AAPL', 'TSLA', 'MSFT']

# # Time
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)

# EF_graph(meanReturns, covMatrix)

maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)


