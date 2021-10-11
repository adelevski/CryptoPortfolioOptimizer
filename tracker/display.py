import pandas as pd

from tracker import get_total
from tracker import get_prices
from tracker import get_portfolio


def display_port():
    portfolio, total_value = get_portfolio(format=True)
    print(portfolio)
    print(f"Total value: ${total_value:.2f}")


if __name__ == "__main__":
    display_port()