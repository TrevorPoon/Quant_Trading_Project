import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm
from Option_Pricing.Streamlit_Page import BSM_Streamlit_Page

# Page selection
st.set_page_config(layout="wide")
page = st.sidebar.selectbox("Choose a page", ["Option Pricing", "Implied Volatility Estimation"])
st.sidebar.divider()

if page == "Option Pricing":

    BSM_Streamlit_Page()

elif page == "Implied Volatility Estimation":

    st.write('hi')

