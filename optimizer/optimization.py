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
# Global variable for the default bounds for allocation per asset            
ASSET_BOUND = (0, 1) 
# Global variable for the number of portfolios to be simulated
NUM_SIM_PORTS = 10000
# Global variable for the number of optimizations to make 
# for the efficient frontier
NUM_OPTS = 50


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
    Note:
        - Instead of calculating annualized returns/volatilities, I decided
        to use the time period instead. I prefer it this way since if we are
        performing research on just the past week (for insights into maybe
        what the next week will hold) annualized returns do not make sense.
        - The "sign" argument is there for the purposes of the 
        "maximize_return" function which passes in "sign = -1.0" for the sake
        of the SciPy minimize function.
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
    Note:
        - Uses time period instead of annualization. See docstring of
        expected_return function above.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) \
                * np.sqrt(delta_days)


def negative_sharpe(
    weights: List[float], 
    mean_returns: pd.core.series.Series,
    cov_matrix: pd.core.frame.DataFrame, 
    delta_days: int, 
    rfr: float=RFR
) -> float:
    """ 
    Given a set of weights of assets, their mean returns, their covaraince
    matrix, the time period for which these statistics have been calculated,
    and the risk-free rate, this fuction will get the expected return and 
    volatility of such a portfolio and then calculate and return the 
    corresponding sharpe ratio.
    Note:
        - The function returns the negative sharpe ratio for the sake
        of the SciPy minimize function.
        - By default, he risk-free rate is set to be the global variable RFR
        defined at the top of this file.
    """
    exp_return = expected_return(weights, mean_returns, delta_days)
    exp_volatility = expected_volatility(weights, cov_matrix, delta_days)
    sharpe_ratio = (exp_return - rfr) / exp_volatility 
    return -1 * sharpe_ratio


def maximize_return(
    mean_returns: pd.core.series.Series, 
    delta_days: int, 
    bound: Tuple[int]=ASSET_BOUND
) -> dict:
    """
    Given the mean returns of a set of assets, the time period for which they
    were calculated, and the bounds for allocation, this function will perform
    an optimization using SciPy to find the weight allocation for the assets
    that gives the maximum return.
    Note:
        - The function returns a dictionary which holds the output of the
        optimization function, which contains the result along with metrics
        from the optimization.
    """
    num_assets = len(mean_returns)
    initial_weights = num_assets * [1.0 / num_assets]
    arguments = (mean_returns, delta_days, -1.0)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
    result = sc.minimize(expected_return, initial_weights, method='SLSQP', \
                    args=arguments, bounds=bounds, constraints=constraints)
    return result


def minimize_volatility( 
    cov_matrix: pd.core.frame.DataFrame, 
    delta_days: int, 
    bound: Tuple[int]=ASSET_BOUND
) -> dict:
    """
    Given the covariance matrix of a set of assets, the time period for which
    they were calculated, and the bounds for allocation, this function will
    perform an optimization using SciPy to find the weight allocation for the
    assets that gives the minimum volatility.
    Note:
        - The function returns a dictionary which holds the output of the
        optimization function, which contains the result along with metrics
        from the optimization.
    """
    num_assets = len(cov_matrix)
    initial_weights = num_assets * [1.0 / num_assets]
    arguments = (cov_matrix, delta_days)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
    result = sc.minimize(expected_volatility, initial_weights, method='SLSQP', \
                        args=arguments, bounds=bounds, constraints=constraints)
    return result


def maximize_sharpe(
    mean_returns: pd.core.series.Series, 
    cov_matrix: pd.core.frame.DataFrame, 
    delta_days: int, 
    bound: Tuple[int]=ASSET_BOUND
) -> dict:
    """
    Given the mean returns of a set of assets, their covariance matrix, the
    time period for which they were calculated, and the bounds for
    allocation, this function will perform an optimization using SciPy to find
    the weight allocation for the assets that gives the maximum sharpe ratio.
    Note:
        - The function returns a dictionary which holds the output of the
        optimization function, which contains the result along with metrics
        from the optimization.
    """
    num_assets = len(mean_returns)
    initial_weights = num_assets * [1.0 / num_assets]
    arguments = (mean_returns, cov_matrix, delta_days)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
    result = sc.minimize(negative_sharpe, initial_weights, method='SLSQP', \
                    args=arguments, bounds=bounds, constraints=constraints)
    return result


def simulate_portfolios(
    mean_returns: pd.core.series.Series, 
    cov_matrix: pd.core.frame.DataFrame, 
    delta_days: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, \
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the mean returns of a set of assets, their covariance matrix, and
    the time period for which they were calculated, this function will
    simulate a large amount of portfolios for the sake of graphing purposes.
    Note:
        - The function also returns the portfolios for which a single asset
        takes up the entire allocation (also for graphing purposes)
        - The function returns a tuple that holds 6 np.ndarrays.
        - The number of portfolios the function simulates is provided by the
        global variable declared at the top of this file.
        - The function uses NumPy's random number generator.
    """
    num_ports = NUM_SIM_PORTS
    num_assets = len(mean_returns)

    # NumPy ndarray initialization
    sharpe_ratios = np.zeros(num_ports)
    exp_returns = np.zeros(num_ports)
    exp_vols = np.zeros(num_ports)
    weights = np.zeros((num_ports, num_assets))
    single_asset_returns = np.zeros(num_assets)
    single_asset_vols = np.zeros(num_assets)

    # NumPy random number generator
    rng = default_rng()

    # Main simulation
    for k in range(num_ports):
        w = rng.uniform(0, 1, size=num_assets)
        weights[k] = w / np.sum(w)
        exp_returns[k] = expected_return(weights[k], mean_returns, delta_days)
        exp_vols[k] = expected_volatility(weights[k], cov_matrix, delta_days)
        sharpe_ratios[k] = (exp_returns[k] - RFR) / exp_vols[k]

    # Single asset allocation simulation
    for i in range(num_assets):
        w = np.zeros(num_assets)
        w[i] = 1
        single_asset_returns[i] = expected_return(w, mean_returns, delta_days)
        single_asset_vols[i] = expected_volatility(w, cov_matrix, delta_days)

    return (exp_returns, exp_vols, sharpe_ratios, weights, \
                    single_asset_returns, single_asset_vols)


def efficient_frontier(
    mean_returns: pd.core.series.Series, 
    cov_matrix: pd.core.frame.DataFrame,  
    delta_days: int, 
    return_target: float, 
    bound: Tuple[int]=ASSET_BOUND
) -> float:
    """
    Given the mean returns of a set of assets, their covariance matrix, the
    time period for which they were calculated, the return target in mind, and
    the bounds for allocation, this function will perform an optimization
    using SciPy to find the weight allocation for the assets that gives the
    minimum volatility and returns it.
    Note:
        - The purpose of this function is to return the minimum volatility
        portfolio for a given return target. This does not return the absolute
        minimum volatility, but instead is called many times for the plotting
        of the efficient frontier. Unlike the minimize_volatility function,
        it also does not return the weights of the portfolio, but the
        volatility itself.
    """
    num_assets = len(mean_returns)
    initial_w = num_assets * [1.0 / num_assets]
    arguments = (cov_matrix, delta_days)
    bounds = tuple(bound for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: expected_return(w, \
                            mean_returns, delta_days) - return_target})
    opt = sc.minimize(expected_volatility, initial_w, method='SLSQP', \
                args=arguments, bounds=bounds, constraints=constraints)
    return opt['fun']


def get_results(
    mean_returns: pd.core.series.Series, 
    cov_matrix: pd.core.frame.DataFrame, 
    delta_days: int
) -> Tuple[Tuple, Tuple, Tuple, List, np.ndarray]:
    """
    Given the mean returns of a set of assets, their covariance matrix, and
    the time period for which those statistics were calculated, this function
    will call all other existing optimization functions from this file in 
    order to get the maximum return, minimum volatility, and the maximum
    sharpe ratio portfolios. It also performs 50 more optimizations for
    return targets between the minimum volatility return and the maximum
    return for the purposes of plotting the efficient frontier.
    Note:
        - The function returns a tuple with 5 values. The first 3 are tuples
        themselves, holding the return, volatility, and weights for each
        type of portfolio. The last 2 are the list of volatilities, and
        the corresponding np.ndarray of return targets for the efficient
        frontier.
    """
    # Maximum return weights, return, and volatility
    max_ret_w = maximize_return(mean_returns, delta_days)['x']
    max_ret_ret = expected_return(max_ret_w, mean_returns, delta_days)
    max_ret_vol = expected_volatility(max_ret_w, cov_matrix, delta_days)

    # Minimum volatility weights, return, and volatility
    min_vol_w = minimize_volatility(cov_matrix, delta_days)['x']
    min_vol_ret = expected_return(min_vol_w, mean_returns, delta_days)
    min_vol_vol = expected_volatility(min_vol_w, cov_matrix, delta_days)

    # Maximum sharpe ratio weights, return, and volatility
    max_sr_w = maximize_sharpe(mean_returns, cov_matrix, delta_days)['x']
    max_sr_ret = expected_return(max_sr_w, mean_returns, delta_days)
    max_sr_vol = expected_volatility(max_sr_w, cov_matrix, delta_days)

    # All portfolios for the efficient frontier
    frontier_list = []
    target_returns = np.linspace(min_vol_ret, max_ret_ret, NUM_OPTS)
    for return_target in target_returns:
        frontier_list.append(efficient_frontier(mean_returns, cov_matrix, \
                                                delta_days, return_target))

    # Each tuple contains the return, volatility, and list of rounded 
    # weights for the corresponding portfolio
    max_ret = (max_ret_ret, max_ret_vol, [round(i*100, 3) for i in max_ret_w])
    min_vol = (min_vol_ret, min_vol_vol, [round(i*100, 3) for i in min_vol_w])
    max_sr = (max_sr_ret, max_sr_vol, [round(i*100, 3) for i in max_sr_w])

    return (max_ret, min_vol, max_sr, frontier_list, target_returns)


def format_weights(assets: List[str], weights: List[float]) -> List[dict]:
    """ Function to format weights and filter out any that are 0.0 """
    proper_form = [{ticker.split('-')[0]:round(x, 1)} \
                   for ticker, x in zip(assets, weights)]
    filtered = [d for d in proper_form if list(d.values())[0] > 0]
    return filtered


def plot_results(
    mean_returns: pd.core.series.Series, 
    cov_matrix: pd.core.frame.DataFrame, 
    delta_days: int, 
    assets: List[str]
) -> None:
    """
    Main function of the file. Takes in the mean returns of a set of assets,
    their covariance matrix, the time period for which those were calculated,
    and the assets themselves and calls all needed functions in order to 
    display the plot of the efficient frontier and all portfolios and all
    simulated portfolios. It also plots the Capital Market Line, and the
    single asset allocation locations on the plot. It also displays a legend,
    and the sharpe ratio bar for visualization. 
    """
    # Make sure assets are sorted for returning allcations since
    # the yfinance package sorts them too.
    assets.sort()

    # All optimization results
    max_ret, min_vol, max_sr, \
    frontier_list, target_returns = \
    get_results(mean_returns, cov_matrix, delta_days)

    # All simulation results
    sim_returns, sim_vols, \
    sim_sharpe, weights, \
    single_asset_returns, single_asset_vols = \
    simulate_portfolios(mean_returns, cov_matrix, delta_days)
    sim_max_index = sim_sharpe.argmax()

    # Clean formatting for display
    max_rets_weights = format_weights(assets, max_ret[2])
    min_vol_weights = format_weights(assets, min_vol[2])
    max_sharpe_weights = format_weights(assets, max_sr[2])

    # Fonts for plot
    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'darkred','size':15}

    # Plotting the CML (Capital Market Line)
    x_vals = np.linspace(0, max_sr[1] * 1.5, 10)
    slope = (max_sr[0] - RFR) / max_sr[1]
    y_vals = (slope * x_vals) + RFR

    # Plotting optimization and simulation results
    plt.figure(figsize=(20, 6))
    plt.scatter(sim_vols, sim_returns, c=sim_sharpe)
    plt.scatter(max_sr[1], max_sr[0], c="red", marker="*", s=200, label="Max Sharpe")
    plt.scatter(max_ret[1], max_ret[0], c="blue", s=80, label="Max Returns")
    plt.scatter(min_vol[1], min_vol[0], c="magenta", s=80, label="Min Volatility")
    for i in range(len(assets)):
        plt.scatter(single_asset_vols[i], single_asset_returns[i], c="black")
        plt.text(single_asset_vols[i], single_asset_returns[i], f"{assets[i].split('-')[0]}")
    
    # Plotting frontier and CML
    plt.plot(frontier_list, target_returns, label="Frontier")
    plt.plot(x_vals, y_vals, c="purple", label="CML")

    # Plot title, labels, and colorbar
    plt.title(f"Efficient Frontier - (past {delta_days} trading days)", fontdict=font1)
    plt.xlabel("Expected Volatility %", fontdict=font2)
    plt.ylabel("Expected Returns %", fontdict=font2)
    plt.colorbar().set_label(label="Sharpe Ratio", fontdict=font2)
    
    # Plot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    black_circle = Line2D([0], [0], marker='o', color='w', label='Single Asset',
                        markerfacecolor='black', markersize=9)
    handles.extend([black_circle])
    plt.legend(handles=handles)

    # Plot formatting 
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.margins(x=0)

    # Plot and results displaying
    plt.show()
    print(f"Max Returns: {max_rets_final}")
    print(f"Min Volatility: {min_vol_final}")
    print(f"Max Sharpe: {max_sharpe_final}")