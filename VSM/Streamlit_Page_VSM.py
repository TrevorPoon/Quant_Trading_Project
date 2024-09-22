import streamlit as st

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from VSM.VolatilitySurfaceModel import Volatility_Surface_Model

def VSM_Streamlit_Page():

    # Sidebar input for user details
    st.sidebar.header('Input Information')

    # Text input for ticker symbol
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

    # Dropdown or radio button for selecting option type (Call or Put)
    option_type = st.sidebar.selectbox("Select Option Type", ['Call', 'Put'])

    st.sidebar.header('Axis Control')

    y_min = st.sidebar.number_input('Days To Expire (min)', min_value=0, max_value=10000, value=0, step=1)
    y_max = st.sidebar.number_input('Days To Expire (max)', min_value=0, max_value=10000, value=100, step=1)
    x_min = st.sidebar.number_input('Strike Price (min)', min_value=0.1, max_value=1000.0, value=0.1, step=0.1)
    x_max = st.sidebar.number_input('Strike Price (max)', min_value=0.1, max_value=1000.0, value=1000.0, step=0.1)

    st.title(ticker+' Volatility Surface Model')
    st.caption("Select your stock ticker (US listed), option type in the sidebar and submit to view the ticker's option volatility surface.")

    # Submit button
    if st.sidebar.button('Submit'):
        # On submit, fetch data and perform volatility modeling
        if ticker:
            # Call the volatility surface model from your imported module
            try:
                # Assuming Volatility_Surface_Model is the main function to calculate the volatility surface
                
                st.plotly_chart(Volatility_Surface_Model(ticker, option_type.lower(), x_min, x_max, y_min, y_max), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter a valid stock ticker.")