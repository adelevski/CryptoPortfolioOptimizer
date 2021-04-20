from datetime import datetime, date
from options_funcs import *
from pandas import DataFrame
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


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


data = {'Symbol': ['S', 'K', 'T', 'r', 'sigma'],
        'Input': [S, K, T , r , sigma]}
input_frame = DataFrame(data, columns=['Symbol', 'Input'], 
                   index=['Underlying price', 'Strike price', 'Time to maturity', 'Risk-free interest rate', 'Volatility'])
print(input_frame)

r = r/100
sigma = sigma/100
binomial_model_pricing = {'Option' : ['Call', 'Put', 'Call', 'Put'],
                          'Price': [Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma,1000,'C'), Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma,1000,'P'),
                                     Jarrow_Rudd_Tree(S, K, T, r, sigma,1000,'C'), Jarrow_Rudd_Tree(S, K, T, r, sigma,1000,'P')]}
binomial_model_pricing_frame = DataFrame(binomial_model_pricing, columns=[ 'Option', 'Price'], 
                   index = ['Cox-Ross-Rubinstein','Cox-Ross-Rubinstein', 'Jarrow-Rudd', 'Jarrow-Rudd'])                                        
print(binomial_model_pricing_frame)


## call option with different steps 
runs1 = list(range(50,5000,50))
CRR1 = []
JR1 = []

for i in runs1:
    CRR1.append(Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma,i ,'C'))
    JR1.append(Jarrow_Rudd_Tree(S, K, T, r, sigma,i ,'C'))

plt.plot(runs1, CRR1, label='Cox_Ross_Rubinstein')
plt.plot(runs1, JR1, label='Jarrow_Rudd')
plt.legend(loc='upper right')
plt.show()

## put option with different steps 
runs2 = list(range(50,5000,50))
CRR2 = []
JR2 = []

for i in runs2:
    CRR2.append(Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma,i ,'P'))
    JR2.append(Jarrow_Rudd_Tree(S, 110, T, r, sigma,i ,'P'))

plt.plot(runs2, CRR2, label='Cox_Ross_Rubinstein')
plt.plot(runs2, JR2, label='Jarrow_Rudd')
plt.legend(loc='upper right')
plt.show()