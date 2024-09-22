import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st


# Function to calculate d1 and d2 in Black-Scholes Model
def black_scholes_d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def black_scholes_d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

# Black-Scholes formula for Call and Put options
def call_option_price(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = black_scholes_d2(d1, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_option_price(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = black_scholes_d2(d1, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Generate a 10x10 range of stock prices and volatilities

def Option_Pricing_Chart(S, S_min, S_max, sigma0, sigma_min, sigma_max, K, T, r):

    selected_call_price = round(call_option_price(S, K, T, r, sigma0),3)
    selected_put_price = round(put_option_price(S, K, T, r, sigma0),3)

    S_values = np.linspace(S_min, S_max, 10)
    sigma_values = np.linspace(sigma_min, sigma_max, 10)

    # Create a grid of stock prices and volatilities
    S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

    # Calculate call option prices over the grid
    call_prices_grid = np.array([[call_option_price(S, K, T, r, sigma) for S in S_values] for sigma in sigma_values])

    # Calculate put option prices over the grid
    put_prices_grid = np.array([[put_option_price(S, K, T, r, sigma) for S in S_values] for sigma in sigma_values])

    # Heatmap with smaller font size (Call Option Prices)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(call_prices_grid, annot=True, fmt=".2f", cmap='viridis', xticklabels=np.round(S_values, 1), 
                yticklabels=np.round(sigma_values, 2), ax=ax1, annot_kws={"size": 10})
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Volatility (σ)')

    # Heatmap with smaller font size (Put Option Prices)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(put_prices_grid, annot=True, fmt=".2f", cmap='plasma', xticklabels=np.round(S_values, 1), 
                yticklabels=np.round(sigma_values, 2), ax=ax2, annot_kws={"size": 10})
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Volatility (σ)')

    return selected_call_price,selected_put_price, fig1, fig2