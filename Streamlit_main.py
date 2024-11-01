import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm
from BSM.Streamlit_Page_BSM import BSM_Streamlit_Page
from VSM.Streamlit_Page_VSM import VSM_Streamlit_Page
from IV_Forecast.Streamlit_Page_IV import IV_Streamlit_Page

# Page selection
st.set_page_config(layout="wide")
page = st.sidebar.selectbox("Choose a page", ["Option Pricing", "Volatility Surface Modelling", "Implied Volatility Forecast"])
st.sidebar.divider()

if page == "Option Pricing":

    BSM_Streamlit_Page()

elif page == "Volatility Surface Modelling":

    VSM_Streamlit_Page()

elif page == "Implied Volatility Forecast":

    IV_Streamlit_Page()

