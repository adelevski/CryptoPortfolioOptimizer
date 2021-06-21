import math
from scipy.stats import norm
import numpy as np


# Underlying price (per share): S; 
# Strike price of the option (per share): K;
# Time to maturity (years): T;
# Continuously compounding risk-free interest rate: r;
# Volatility: sigma;

## define two functions, d1 and d2 in Black-Scholes model
def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+sigma**2/2.)*T)/sigma*sqrt(T)
def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

## define the call options price function
def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

## define the put options price function
def bs_put(S,K,T,r,sigma):
    return K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)

## define the Call_Greeks of an option
def call_delta(S,K,T,r,sigma):
    return norm.cdf(d1(S,K,T,r,sigma))
def call_gamma(S,K,T,r,sigma):
    return norm.pdf(d1(S,K,T,r,sigma))/(S*sigma*sqrt(T))
def call_vega(S,K,T,r,sigma):
    return 0.01*(S*norm.pdf(d1(S,K,T,r,sigma))*sqrt(T))
def call_theta(S,K,T,r,sigma):
    return 0.01*(-(S*norm.pdf(d1(S,K,T,r,sigma))*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma)))
def call_rho(S,K,T,r,sigma):
    return 0.01*(K*T*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma)))


## define the Put_Greeks of an option
def put_delta(S,K,T,r,sigma):
    return -norm.cdf(-d1(S,K,T,r,sigma))
def put_gamma(S,K,T,r,sigma):
    return norm.pdf(d1(S,K,T,r,sigma))/(S*sigma*sqrt(T))
def put_vega(S,K,T,r,sigma):
    return 0.01*(S*norm.pdf(d1(S,K,T,r,sigma))*sqrt(T))
def put_theta(S,K,T,r,sigma):
    return 0.01*(-(S*norm.pdf(d1(S,K,T,r,sigma))*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))
def put_rho(S,K,T,r,sigma):
    return 0.01*(-K*T*exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))

## to calculate the volatility of a put/call option
def implied_volatility(Price,S,K,T,r, option):
    sigma = 0.001
    print (np.array([['Price', 'S', 'K', 'T', 'r'], [Price, S, K, T, r]]))
    if option == 'C':
        while sigma < 1:
            Price_implied = S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
            if Price-(Price_implied) < 0.001:
                return sigma
            sigma += 0.001
        return "It could not find the right volatility of the call option."
    else:
        while sigma < 1:
            Price_implied = K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)
            if Price-(Price_implied) < 0.001:
                return sigma
            sigma += 0.001
        return "It could not find the right volatility of the put option."
    return

## define Cox_Ross_Rubinstein binomial model
def Cox_Ross_Rubinstein_Tree (S,K,T,r,sigma,N, Option_type):
    
    # Underlying price (per share): S; 
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Number of binomial steps: N;

        # The factor by which the price rises (assuming it rises) = u ;
        # The factor by which the price falls (assuming it falls) = d ;
        # The probability of a price rise = pu ;
        # The probability of a price fall = pd ;
        # discount rate = disc ;
    
    u=math.exp(sigma*math.sqrt(T/N))
    d=math.exp(-sigma*math.sqrt(T/N))
    pu=((math.exp(r*T/N))-d)/(u-d)
    pd=1-pu
    disc=math.exp(-r*T/N)

    St = [0] * (N+1)
    C = [0] * (N+1)
    
    St[0]=S*d**N
    
    for j in range(1, N+1): 
        St[j] = St[j-1] * u/d
    
    for j in range(1, N+1):
        if Option_type == 'P':
            C[j] = max(K-St[j],0)
        elif Option_type == 'C':
            C[j] = max(St[j]-K,0)
    
    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc*(pu*C[j+1]+pd*C[j])
            
    return C[0]


## define Jarrow_Rudd binomial model    
def Jarrow_Rudd_Tree (S,K,T,r,sigma,N, Option_type):

    # Underlying price (per share): S; 
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Steps: N;
    
        # The factor by which the price rises (assuming it rises) = u ;
        # The factor by which the price falls (assuming it falls) = d ;
        # The probability of a price rise = pu ;
        # The probability of a price fall = pd ;
        # discount rate = disc ;
        
    u=math.exp((r-(sigma**2/2))*T/N+sigma*math.sqrt(T/N))
    d=math.exp((r-(sigma**2/2))*T/N-sigma*math.sqrt(T/N))
    pu=0.5
    pd=1-pu
    disc=math.exp(-r*T/N)

    St = [0] * (N+1)
    C = [0] * (N+1)
    
    St[0]=S*d**N
    
    for j in range(1, N+1): 
        St[j] = St[j-1] * u/d
    
    for j in range(1, N+1):
        if Option_type == 'P':
            C[j] = max(K-St[j],0)
        elif Option_type == 'C':
            C[j] = max(St[j]-K,0)
    
    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc*(pu*C[j+1]+pd*C[j])
            
    return C[0]