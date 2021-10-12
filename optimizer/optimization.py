# Functional imports
import numpy as np
import pandas as pd
import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from numpy.random import default_rng

# Imports for type hints
from typing import List, Tuple


# Global variable for the risk-free rate
RFR = 0.00


def expected_return(
    weights: List[float], 
    mean_returns: pd.core.series.Series,
    delta_days: int,
    sign: float=1.0
) -> float:
    """
    Given a set of weights of assets, their mean returns, and the time period
    for which these statistics have been calculated, this function will 
    calculate and return the expected return for such a portfolio.
    Notes:
        - Instead of calculating annualized returns/volatilities, I decided
        to use the time period instead. I prefer it this way since if we are
        performing research on just the past week (for insights into maybe
        what the next week will hold) annualized returns do not make sense.
        - The "sign" argument is there for the purposes of the 
        "maximize_return" function which passes in "sign = -1.0" for the sake
        of the scipy minimize function.
    """
    return sign*np.sum(mean_returns * weights) * delta_days


def expected_volatility(
    weights: List[float], 
    cov_matrix: pd.core.frame.DataFrame,
    delta_days: int
) -> float:
    """
    Given a set of weights of assets, their covariance matrix, and the time
    period for which these statistics have been calculated, this function will 
    calculate and return the expected volatility for such a portfolio.
    Notes:
        - Uses time period instead of annualization. See docstring of
        expected_return function above.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(delta_days)


def negative_sharpe(weights, mean_returns, cov_matrix, delta_days, rfr=RFR):
    exp_return = expected_return(weights, mean_returns, delta_days)
    exp_volatility = expected_volatility(weights, cov_matrix, delta_days)
    sharpe_ratio = (exp_return - rfr) / exp_volatility
    neg_sharpe = -1 * sharpe_ratio
    return neg_sharpe


def maximize_return(mean_returns, cov_matrix, delta_days, bound=(0,1)):
    num_assets = len(mean_returns)
    initial_weights = num_assets * [1.0 / num_assets]
    arguments = (mean_returns, delta_days, -1.0)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
    result = sc.minimize(expected_return, initial_weights, method='SLSQP', args=arguments, bounds=bounds, constraints=constraints)
    return result


def minimize_volatility(mean_returns, cov_matrix, delta_days, bound=(0,1)):
    num_assets = len(mean_returns)
    initial_weights = num_assets * [1.0 / num_assets]
    arguments = (cov_matrix, delta_days)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
    result = sc.minimize(expected_volatility, initial_weights, method='SLSQP', args=arguments, bounds=bounds, constraints=constraints)
    return result


def maximize_sharpe(mean_returns, cov_matrix, delta_days, bound=(0,1)):
    num_assets = len(mean_returns)
    initial_weights = num_assets * [1.0 / num_assets]
    arguments = (mean_returns, cov_matrix, delta_days)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
    result = sc.minimize(negative_sharpe, initial_weights, method='SLSQP', args=arguments, bounds=bounds, constraints=constraints)
    return result


def simulate_portfolios(mean_returns, cov_matrix, delta_days):
    num_ports = 10000
    num_assets = len(mean_returns)

    sharpe_ratios = np.zeros(num_ports)
    exp_returns = np.zeros(num_ports)
    exp_vols = np.zeros(num_ports)

    weights = np.zeros((num_ports, num_assets))

    single_asset_returns = np.zeros(num_assets)
    single_asset_vols = np.zeros(num_assets)

    rng = default_rng()

    for k in range(num_ports):
        w = rng.uniform(0, 1, size=num_assets)
        weights[k] = w / np.sum(w)
        exp_returns[k] = expected_return(weights[k], mean_returns, delta_days)
        exp_vols[k] = expected_volatility(weights[k], cov_matrix, delta_days)
        sharpe_ratios[k] = (exp_returns[k] - RFR) / exp_vols[k]
    for i in range(num_assets):
        w = np.zeros(num_assets)
        w[i] = 1
        single_asset_returns[i] = expected_return(w, mean_returns, delta_days)
        single_asset_vols[i] = expected_volatility(w, cov_matrix, delta_days)

    return exp_returns, exp_vols, sharpe_ratios, weights, single_asset_returns, single_asset_vols


def efficient_frontier(mean_returns, cov_matrix, delta_days, return_target, bound=(0,1)):
    num_assets = len(mean_returns)
    initial_w = num_assets * [1.0 / num_assets]
    arguments = (cov_matrix, delta_days)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: expected_return(w, mean_returns, delta_days) - return_target})
    opt = sc.minimize(expected_volatility, initial_w, method='SLSQP', args=arguments, bounds=bounds, constraints=constraints)
    return opt['fun']


def get_results(mean_returns, cov_matrix, delta_days):
    max_rets_port = maximize_return(mean_returns, cov_matrix, delta_days)['x']
    max_rets_returns = expected_return(max_rets_port, mean_returns, delta_days)
    max_rets_vol = expected_volatility(max_rets_port, cov_matrix, delta_days)

    min_vol_port = minimize_volatility(mean_returns, cov_matrix, delta_days)['x']
    min_vol_returns = expected_return(min_vol_port, mean_returns, delta_days)
    min_vol_vol = expected_volatility(min_vol_port, cov_matrix, delta_days)

    max_sharpe_port = maximize_sharpe(mean_returns, cov_matrix, delta_days)['x']
    max_sharpe_returns = expected_return(max_sharpe_port, mean_returns, delta_days)
    max_sharpe_vol = expected_volatility(max_sharpe_port, cov_matrix, delta_days)

    frontier_list = []
    target_returns = np.linspace(min_vol_returns, max_rets_returns, 50)
    for return_target in target_returns:
        frontier_list.append(efficient_frontier(mean_returns, cov_matrix, delta_days, return_target))


    max_rets_perf = (max_rets_returns, max_rets_vol, [round(i * 100, 3) for i in max_rets_port])
    min_vol_perf = (min_vol_returns, min_vol_vol, [round(i * 100, 3) for i in min_vol_port])
    max_sharpe_perf = (max_sharpe_returns, max_sharpe_vol, [round(i * 100, 3) for i in max_sharpe_port])

    return max_rets_perf, min_vol_perf, max_sharpe_perf, frontier_list, target_returns


def plot_results(mean_returns, cov_matrix, delta_days, assets):
    assets.sort()
    max_rets_perf, min_vol_perf, max_sharpe_perf, frontier_list, target_returns = get_results(mean_returns, cov_matrix, delta_days)
    sim_returns, sim_vols, sim_sharpe, weights, single_asset_returns, single_asset_vols = simulate_portfolios(mean_returns, cov_matrix, delta_days)
    sim_max_index = sim_sharpe.argmax()

    max_rets_w = max_rets_perf[2]
    max_rets_weights = [{ticker.split('-')[0]:round(x, 1)} for ticker, x in zip(assets, max_rets_w)]
    max_rets_final = [d for d in max_rets_weights if list(d.values())[0] > 0] 

    min_vol_w = min_vol_perf[2]
    min_vol_weights = [{ticker.split('-')[0]:round(x, 1)} for ticker, x in zip(assets, min_vol_w)]
    min_vol_final = [d for d in min_vol_weights if list(d.values())[0] > 0]

    max_sharpe_w = max_sharpe_perf[2]
    max_sharpe_weights = [{ticker.split('-')[0]:round(x, 1)} for ticker, x in zip(assets, max_sharpe_w)]
    max_sharpe_final = [d for d in max_sharpe_weights if list(d.values())[0] > 0]

    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'darkred','size':15}

    x_vals = np.linspace(0, max_sharpe_perf[1] * 1.5, 10)
    slope = (max_sharpe_perf[0] - RFR) / max_sharpe_perf[1]
    y_vals = (slope * x_vals) + RFR

    plt.figure(figsize=(20, 6))
    plt.scatter(sim_vols, sim_returns, c=sim_sharpe)
    plt.title(f"Efficient Frontier - (past {delta_days} trading days)", fontdict=font1)
    plt.xlabel("Expected Volatility %", fontdict=font2)
    plt.ylabel("Expected Returns %", fontdict=font2)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.colorbar().set_label(label="Sharpe Ratio", fontdict=font2)
    plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0], c="red", marker="*", s=200, label="Max Sharpe")
    plt.scatter(max_rets_perf[1], max_rets_perf[0], c="blue", s=80, label="Max Returns")
    plt.scatter(min_vol_perf[1], min_vol_perf[0], c="magenta", s=80, label="Min Volatility")
    for i in range(len(assets)):
        plt.scatter(single_asset_vols[i], single_asset_returns[i], c="black")
        plt.text(single_asset_vols[i], single_asset_returns[i], f"{assets[i].split('-')[0]}")
    plt.plot(frontier_list, target_returns, label="Frontier")
    plt.plot(x_vals, y_vals, c="purple", label="CML")
    plt.margins(x=0)
    handles, labels = plt.gca().get_legend_handles_labels()
    black_circle = Line2D([0], [0], marker='o', color='w', label='Single Asset',
                        markerfacecolor='black', markersize=9)
    handles.extend([black_circle])
    plt.legend(handles=handles)
    plt.show()
    print(f"Max Returns: {max_rets_final}")
    print(f"Min Volatility: {min_vol_final}")
    print(f"Max Sharpe: {max_sharpe_final}")