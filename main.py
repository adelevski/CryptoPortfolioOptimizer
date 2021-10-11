from funcs import get_total
from funcs import get_prices
import pandas as pd


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



if __name__ == "__main__":
    portfolio, total_value = get_portfolio(format=True)
    print(portfolio)
    print(f"Total value: ${total_value:.2f}")
