import datetime as dt
from efFuncs import getData, EF_graph




stocks = ['AAPL', 'TSLA', 'MSFT']

# Time (2 years back)
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)

EF_graph(meanReturns, covMatrix)