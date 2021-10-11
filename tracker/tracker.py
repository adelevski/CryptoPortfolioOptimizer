import pandas as pd
from collections import defaultdict

import tracker.alts as alts
from tracker.auth import cmc
from tracker.auth import exchanges



# Retrieve relevant information from ccxt response
def parse(response):
    return {k:v for k,v in response['total'].items() if v}


# Aggregate all dictionaries
def dsum(dicts):
    ret = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


# Get total balance 
def get_total():
    all_holdings = [parse(exchange.fetch_balance()) for exchange in exchanges]
    all_holdings.append(alts.voyager_balance)
    all_holdings.append(alts.metamask_balance)
    total_balance = dsum(all_holdings)
    total_balance['LYXe'] = total_balance.pop('LYXE')
    return dict(sorted(total_balance.items()))


# String Constructor function for CMC API price fetching
def string_maker(balance):
    symbol_string = ''
    for key in balance:
        if key != 'USD':
            symbol_string += key + ','
    return symbol_string[:-1]


def get_prices(total_balance):
    symbol_string = string_maker(total_balance)
    data = cmc.cryptocurrency_quotes_latest(symbol=symbol_string)
    prices = {}
    for key in data.data:
        prices[data.data[key]['symbol']] = data.data[key]['quote']['USD']['price']
    return prices


def get_portfolio(format: bool=False):
    # data fetching
    total_balance = get_total()
    prices = get_prices(total_balance)

    # dataframe creation and processing
    df = pd.DataFrame.from_dict(total_balance, columns=['amount'], orient='index')
    df['price'] = df.index.map(prices)
    df.fillna(1.0, inplace=True)
    df['value'] = df['price'] * df['amount']
    total_value = df['value'].sum()
    df['weight'] = (df['value'] / total_value) * 100

    # dataframe formatting for presentation
    if format:
        df['amount'] = df['amount'].map('{:,.4f}'.format)
        df['price'] = df['price'].map('${:,.2f}'.format)
        df['value'] = df['value'].map('${:,.2f}'.format)
        df['weight'] = df['weight'].map('{:,.2f}%'.format)

    return df, total_value


def get_owned():
    port = get_portfolio()
    owned = list(port[0].index)
    faulty = ['KCS','CHSB','LTO','LYXe','REN','FORTH','RFUEL','REVV','USDT', 'USD', 'NMR','STMX']
    rename = {'ATOM': 'ATOM1', 'DOT': 'DOT1', 'GRT': 'GRT2', 'ONE': 'ONE2', 'SOL': 'SOL1'}
    owned = [ticker for ticker in owned if ticker not in faulty]
    owned = sorted([ticker+'-USD' if ticker not in rename.keys() else rename[ticker]+'-USD' for ticker in owned])
    return owned