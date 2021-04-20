from datetime import datetime, date
from pandas import DataFrame
from options_funcs import *

## input the current stock price and check if it is a number.
S = input("What is the current stock price? ")
while True:
    try:
        S = float(S)
        break
    except:
        print("The current stock price has to be a NUMBER.")
        S = input("What is the current stock price? ")

## input the strike price and check if it is a number.
K = input("What is the strike price? ")
while True:
    try:
        K = float(K)
        break
    except:
        print("The the strike price has to be a NUMBER.")
        K = input("What is the strike price? ")


## input the expiration_date and calculate the days between today and the expiration date.
while True:
    expiration_date = input("What is the expiration date of the options? (mm-dd-yyyy) ")
    try:
        expiration_date = datetime.strptime(expiration_date, "%m-%d-%Y")
    except ValueError as e:
        print("error: %s\nTry again." % (e,))
    else:
        break
T = (expiration_date - datetime.utcnow()).days / 365


## input the continuously compounding risk-free interest rate and check if it is a number.
r = input("What is the continuously compounding risk-free interest rate in percentage(%)? ")
while True:
    try:
        r = float(r)
        break
    except:
        print("The continuously compounding risk-free interest rate has to be a NUMBER.")
        r = input("What is the continuously compounding risk-free interest rate in percentage(%)? ")
        

## input the volatility and check if it is a number.
sigma = input("What is the volatility in percentage(%)? "); 
while True:
    try:
        sigma = float(sigma)
        if sigma > 100 or sigma < 0:
            print ( "The range of sigma has to be in [0,100].")
            sigma = input("What is the volatility in percentage(%)? ")
        break
    except:
        print("The volatility has to be a NUMBER.")
        sigma = input("What is the volatility in percentage(%)? ")


## make a DataFrame of these inputs
data = {'Symbol': ['S', 'K', 'T', 'r', 'sigma'],
        'Input': [S, K, T , r , sigma]}
input_frame = DataFrame(data, columns=['Symbol', 'Input'], 
                   index=['Underlying price', 'Strike price', 'Time to maturity', 'Risk-free interest rate', 'Volatility'])
print(input_frame)

## calculate the call / put option price and the greeks of the call / put option
r = r/100
sigma = sigma/100
price_and_greeks = {'Call' : [bs_call(S,K,T,r,sigma), call_delta(S,K,T,r,sigma), call_gamma(S,K,T,r,sigma),call_vega(S,K,T,r,sigma), call_rho(S,K,T,r,sigma), call_theta(S,K,T,r,sigma)],
                    'Put' : [bs_put(S,K,T,r,sigma), put_delta(S,K,T,r,sigma), put_gamma(S,K,T,r,sigma),put_vega(S,K,T,r,sigma), put_rho(S,K,T,r,sigma), put_theta(S,K,T,r,sigma)]}
price_and_greeks_frame = DataFrame(price_and_greeks, columns=['Call','Put'], index=['Price', 'delta', 'gamma','vega','rho','theta'])
print(price_and_greeks_frame)

## input a put or call option price
option = input ("Put or Call option? (P/C)  ")
while option != 'P' and option !='C' :
    print ("error: this option does not match the format (P/C) \nTry again.")
    option = input ("Put or Call option? (P/C)  ")

Price = input("What is the option price? ")
while True:
    try:
        Price = float(Price)
        break
    except:
        print("The the option price has to be a NUMBER.")
        Price = input("What is the option price? ")


print ("The implied volatility is " + str (100* implied_volatility(Price,S,K,T,r,option)) + " %.") 