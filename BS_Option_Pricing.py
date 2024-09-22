import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm
from Option_Pricing.Option_Pricing import Option_Pricing_Chart

# Page selection
st.set_page_config(layout="wide")
page = st.sidebar.selectbox("Choose a page", ["Option Pricing", "Monte Carlo Simulation"])
st.sidebar.divider()

if page == "Option Pricing":
    # Streamlit UI for Option Pricing Heatmap
    

    # User inputs for the option pricing model
    st.sidebar.header('Input Parameters')
    S = st.sidebar.number_input('Spot Price (S)', min_value=0.1, max_value=10000.0, value=100.0, step=0.1)
    K = st.sidebar.number_input('Strike Price (K)', min_value=0.1, max_value=1000.0, value=100.0, step=1.0)
    T = st.sidebar.number_input('Time to Expiration (Years)', min_value=0.0001, max_value=1.0, value=1.0, step=1.0)
    sigma0 = st.sidebar.number_input('Volatility (σ)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    r = st.sidebar.number_input('Risk-Free Interest Rate (r)', min_value=0.0, max_value=0.2, value=0.05, step=0.01)

    st.sidebar.divider()
    
    # Heatmap axis limits
    st.sidebar.subheader('Heatmap Axis Ranges')
    S_min = st.sidebar.number_input('Minimum Stock Price (S)', min_value=0.1, max_value=10000.0, value=S*0.9, step=0.1)
    S_max = st.sidebar.number_input('Maximum Stock Price (S)', min_value=0.1, max_value=10000.0, value=S*1.1, step=0.1)
    sigma_min = st.sidebar.number_input('Minimum Volatility (σ)', min_value=0.01, max_value=1.0, value=sigma0 - 0.1, step=0.01)
    sigma_max = st.sidebar.number_input('Maximum Volatility (σ)', min_value=0.01, max_value=1.0, value=sigma0 + 0.1, step=0.01)

    
    call_price, put_price, fig1, fig2 = Option_Pricing_Chart(S, S_min, S_max, sigma0, sigma_min, sigma_max, K, T, r)
    

    # Display Call and Put Prices with nicer formatting
    st.markdown("<h1 style='text-align: left;'>Black Scholes Option Pricing Model</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label='Call Price', value=f"${call_price:.2f}", delta=None)
    with col2:
        st.metric(label='Put Price', value=f"${put_price:.2f}", delta=None)

    # Separator for visual appeal
    st.markdown("---")

    # Section Title for Heatmap
    st.markdown("<h1 style='text-align: left;'>Option Pricing Interactive Heatmap</h1>", unsafe_allow_html=True)

    # Create columns for heatmap display
    col1, col2 = st.columns(2)

    # Display heatmaps in Streamlit
    with col1:
        st.subheader('Call Option Prices')
        st.pyplot(fig1)

    with col2:
        st.subheader('Put Option Prices')
        st.pyplot(fig2)

    # Add some additional space at the bottom
    st.markdown("<br>", unsafe_allow_html=True)

elif page == "Monte Carlo Simulation":

    st.title('Monte Carlo Simulation')

    # Monte Carlo Simulation Parameters
    st.sidebar.header('Monte Carlo Simulation Parameters')
    S0 = st.sidebar.number_input('Initial Stock Price (S0)', min_value=0.1, max_value=10000.0, value=100.0, step=0.1)
    T = st.sidebar.number_input('Time to Expiration (T in years)', min_value=0.01, max_value=2.0, value=1.0, step=0.01)
    r = st.sidebar.number_input('Risk-Free Rate (r)', min_value=0.0, max_value=0.2, value=0.05, step=0.01)
    sigma = st.sidebar.number_input('Volatility (σ)', min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    simulations = st.sidebar.number_input('Number of Simulations', min_value=100, max_value=10000, value=1000, step=100)
    
    # Perform Monte Carlo Simulation
    np.random.seed(42)
    dt = T / 252
    S = np.zeros((simulations, 252))
    S[:, 0] = S0

    # Simulate price paths
    for t in range(1, 252):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(simulations))

    # Plot the results of Monte Carlo simulation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S.T, lw=1)
    ax.set_title('Monte Carlo Simulation of Stock Price Paths')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Stock Price')
    
    # Display Monte Carlo simulation plot
    st.pyplot(fig)
