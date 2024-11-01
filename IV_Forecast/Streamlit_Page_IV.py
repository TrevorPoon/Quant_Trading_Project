import streamlit as st

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from IV_Forecast.ML_IV_Forecast import IV_Forecast

def IV_Streamlit_Page():

    # Sidebar input for user details
    st.sidebar.header('Input Information')

    # Text input for ticker symbol
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

    start_date = st.sidebar.text_input("Enter Start Date", value="2024-01-01")

    # Submit button
    if st.sidebar.button('Submit'):
        
        if ticker:
            try:
                
                market_option_price, price_garch, price_rf, price_lstm, market_option_iv, sigma_garch, sigma_rf, sigma_lstm, fg= IV_Forecast(ticker, start_date)
                st.write(market_option_price, price_garch, price_rf, price_lstm)
                st.write(market_option_iv, sigma_garch, sigma_rf, sigma_lstm)
                st.pyplot(fg)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter a valid stock ticker.")