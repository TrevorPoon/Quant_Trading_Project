import streamlit as st

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from BSM.Option_Pricing import Option_Pricing_Chart

def BSM_Streamlit_Page():
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
        st.metric(label='Call Price', value=f"${call_price:.3f}", delta=None)
    with col2:
        st.metric(label='Put Price', value=f"${put_price:.3f}", delta=None)

    # Separator for visual appeal
    st.markdown("---")

    # Section Title for Heatmap
    st.markdown("<h1 style='text-align: left;'>Option Pricing Interactive Heatmap</h1>", unsafe_allow_html=True)
    st.caption("Discover how option prices change with different spot prices and volatility levels through interactive heatmap parameters, while keeping the 'Strike Price' constant.")
    # Create columns for heatmap display
    col1, col2 = st.columns(2)

    # Display heatmaps in Streamlit
    with col1:
        st.subheader('Call Option Prices Heatmap')
        st.pyplot(fig1)

    with col2:
        st.subheader('Put Option Prices Heatmap')
        st.pyplot(fig2)

    # Add some additional space at the bottom
    st.markdown("<br>", unsafe_allow_html=True)