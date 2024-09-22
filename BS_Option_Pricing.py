import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# Streamlit UI
st.title('Option Pricing Visualization')
st.sidebar.header('Input Parameters')

# User inputs for the option pricing model
S = st.sidebar.slider('Stock Price (S)', min_value=10.0, max_value=500.0, value=100.0, step=1.0)
K = st.sidebar.slider('Strike Price (K)', min_value=10.0, max_value=500.0, value=100.0, step=1.0)
T = st.sidebar.slider('Time to Expiration (T in years)', min_value=0.01, max_value=2.0, value=1.0, step=0.01)
r = st.sidebar.slider('Risk-Free Interest Rate (r)', min_value=0.0, max_value=0.2, value=0.05, step=0.01)
sigma = st.sidebar.slider('Volatility (σ)', min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# Calculate option prices
call_price = call_option_price(S, K, T, r, sigma)
put_price = put_option_price(S, K, T, r, sigma)

# Display results
st.write(f"### Call Option Price: {call_price:.2f}")
st.write(f"### Put Option Price: {put_price:.2f}")

# Plot the effect of changing stock price on option prices
S_values = np.linspace(10, 500, 100)
call_prices = [call_option_price(S, K, T, r, sigma) for S in S_values]
put_prices = [put_option_price(S, K, T, r, sigma) for S in S_values]

# Plotting
fig, ax = plt.subplots()
ax.plot(S_values, call_prices, label='Call Option Price', color='blue')
ax.plot(S_values, put_prices, label='Put Option Price', color='red')
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Option Price')
ax.legend()
ax.set_title('Option Prices vs Stock Price')

# Clear previous plot
plt.tight_layout()

# Display plot in Streamlit
st.pyplot(fig)
