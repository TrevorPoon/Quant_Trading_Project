import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

# Fetch AAPL options data
def fetch_options_data(ticker,call_or_put):
    stock = yf.Ticker(ticker)
    expiries = stock.options  # Get all expiration dates
    
    options_data = []
    
    for expiry in expiries[:5]:  # Fetch first 5 expiry dates to avoid large data (can increase later)
        opt_chain = stock.option_chain(expiry)
        if call_or_put == "call":
            opts = opt_chain.calls
        elif call_or_put == "put":
            opts = opt_chain.puts
        opts['expiry'] = expiry
        options_data.append(opts)
    
    options_df = pd.concat(options_data, ignore_index=True)
    return options_df

# Filter required columns and clean data
def preprocess_data(options_df):
    relevant_columns = ['strike', 'lastPrice', 'impliedVolatility', 'expiry']
    options_df = options_df[relevant_columns].dropna()
    
    # Remove negative or zero implied volatility
    options_df = options_df[options_df['impliedVolatility'] > 0]
    
    # Convert expiry date to number of days until expiry
    options_df['expiry'] = pd.to_datetime(options_df['expiry'])
    options_df['days_to_expiry'] = (options_df['expiry'] - pd.Timestamp.today()).dt.days
    
    return options_df

### SVI Model
def svi(params, k):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def svi_loss(params, strikes, implied_vols):
    k = np.log(strikes)
    svi_vols = svi(params, k)
    return np.mean((svi_vols - implied_vols) ** 2)

def fit_svi_surface(options_data):
    strikes = options_data['strike'].values
    implied_vols = options_data['impliedVolatility'].values
    initial_params = [0.1, 0.1, 0, np.log(np.mean(strikes)), 0.1]  # Initial guess
    
    result = minimize(svi_loss, initial_params, args=(strikes, implied_vols), method='L-BFGS-B')
    return result.x

def svi_surface(params, strikes, expiries):

    # Convert strikes to log-strikes
    log_strikes = np.log(strikes)
    
    # Use SVI model to calculate volatilities for each log-strike
    vol_surface = svi(params, log_strikes)
    
    return vol_surface

### SABR Model
def sabr_vol(alpha, beta, rho, nu, F, K, T):
    # Element-wise comparison: use np.where to apply different formulas depending on F == K
    epsilon = 1e-7  # Small constant to avoid division by zero
    
    # Apply the formula for F != K
    z = (nu / alpha) * ((F * K) ** ((1 - beta) / 2)) * np.log(F / K + epsilon)
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho) + epsilon)
    
    # SABR volatility formula
    vol = (alpha * (F * K) ** ((beta - 1) / 2) *
           (1 + ((T * ((1 - beta)**2 * alpha**2) / (24 * (F * K)**(1 - beta)) +
                 rho * beta * nu * alpha / (4 * (F * K)**((1 - beta) / 2)) +
                 (2 - 3 * rho**2) * nu**2 / 24) * T)) * (z / x_z))
    
    # Special case: When F == K, apply a different formula
    vol_F_eq_K = alpha * F ** (beta - 1)
    
    # Use np.where to handle the case where F == K element-wise
    return np.where(np.abs(F - K) < epsilon, vol_F_eq_K, vol)

def sabr_loss(params, F, K, T, market_vols):
    alpha, beta, rho, nu = params
    sabr_vols = sabr_vol(alpha, beta, rho, nu, F, K, T)
    return np.mean((sabr_vols - market_vols) ** 2)

def fit_sabr_surface(options_data, forward_price):
    strikes = options_data['strike'].values
    implied_vols = options_data['impliedVolatility'].values
    expiries = options_data['days_to_expiry'].values / 365  # Convert to years
    initial_params = [0.2, 0.5, 0, 0.2]  # Initial guess for alpha, beta, rho, nu

    result = minimize(sabr_loss, initial_params, args=(forward_price, strikes, expiries, implied_vols), method='L-BFGS-B')
    return result.x

### Local Volatility Model
def local_volatility_surface(strikes, expiries, implied_vols):
    # Finite difference approximation for local volatility using Dupire's formula
    local_vols = np.zeros_like(implied_vols)
    for i in range(1, len(strikes) - 1):
        for j in range(1, len(expiries) - 1):
            dK = (strikes[i + 1] - strikes[i - 1]) / 2
            dT = (expiries[j + 1] - expiries[j - 1]) / 2
            dVdK = (implied_vols[j, i + 1] - implied_vols[j, i - 1]) / (2 * dK)
            dVdT = (implied_vols[j + 1, i] - implied_vols[j - 1, i]) / (2 * dT)
            local_vols[j, i] = implied_vols[j, i] / np.sqrt(1 - strikes[i] * dVdK * dT / implied_vols[j, i])
    return local_vols


def Volatility_Surface_Model(ticker, call_or_put, x_min, x_max, y_min, y_max):
    # Get the options data for AAPL
    aapl_options_data = fetch_options_data(ticker,call_or_put)
    # print(aapl_options_data.head())

    # Preprocess the options data
    cleaned_data = preprocess_data(aapl_options_data)
    # print(cleaned_data.head())

    svi_params = fit_svi_surface(cleaned_data)
    # print("Fitted SVI parameters:", svi_params)

    forward_price = cleaned_data['strike'].mean()  # Assume forward price ~ mean strike price for simplicity
    sabr_params = fit_sabr_surface(cleaned_data, forward_price)
    # print("Fitted SABR parameters:", sabr_params)

    # Generate a grid of strikes and expiries
    strike_grid = np.linspace(cleaned_data['strike'].min(), cleaned_data['strike'].max(), 100)
    expiry_grid = np.linspace(cleaned_data['days_to_expiry'].min(), cleaned_data['days_to_expiry'].max(), 100)

    # Create meshgrid for 3D plotting
    strike_mesh, expiry_mesh = np.meshgrid(strike_grid, expiry_grid)

    # Calculate SVI surface
    svi_volatility_mesh = svi_surface(svi_params, strike_mesh, expiry_mesh) * 100

    print(max(strike_mesh[0]))

    # Calculate SABR surface
    # sabr_volatility_mesh = sabr_vol(sabr_params[0], sabr_params[1], sabr_params[2], sabr_params[3], forward_price, strike_mesh, expiry_mesh / 365)

    fig = go.Figure(data=[go.Surface(
        z=svi_volatility_mesh, 
        x=strike_mesh, 
        y=expiry_mesh, 
        colorscale='Viridis',

        contours = {
        "x": {"show": True, "color": "black", "highlight": True, "highlightcolor": "white"},
        "y": {"show": True, "color": "black", "highlight": True, "highlightcolor": "white"},
        "z": {"show": True, "color": "black", "highlight": True, "highlightcolor": "white"}
        },
        # Add hoverinfo to show details when hovering
        hovertemplate='<b>Strike Price:</b> %{x}<br><b>Days to Expiry:</b> %{y}<br><b>Implied Volatility:</b> %{z:.2f}%<extra></extra>'
        )])

    # Update layout for better appearance and gridlines
    fig.update_layout(
        title='Stochastic Implied Volatility -- Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiry',
            zaxis_title='Implied Volatility (%)',

            xaxis=dict(
                range=[x_min, min(x_max,max(strike_mesh[0]))],
            ),

            yaxis=dict(
                range=[y_min, min(y_max,expiry_mesh[-1][0])],
            ),
            
            # Add gridlines on the surface itself
            xaxis_showspikes=True,
            yaxis_showspikes=True,
            zaxis_showspikes=True,
        ),
        
        # Set overall background color
        autosize=True,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # SABR Volatility Surface
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot_surface(strike_mesh, expiry_mesh, sabr_volatility_mesh, cmap='plasma')
    # ax2.set_xlabel('Strike Price')
    # ax2.set_ylabel('Days to Expiry')
    # ax2.set_zlabel('Implied Volatility')
    # ax2.set_title('SABR Volatility Surface')

    return fig
