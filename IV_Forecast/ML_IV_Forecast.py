# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.stats import norm
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price for European options.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def IV_Forecast(ticker, start_date):
    # Fetch historical stock data
    data = yf.download(ticker, start=start_date)
    end_date = data.index.max()

    # Compute log returns
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data.dropna(inplace=True)

    # ----------------------- GARCH Model -----------------------
    # Fit GARCH(1,1) model
    am = arch_model(data['Log_Returns'] * 100, vol='GARCH', p=1, q=1)
    res = am.fit(update_freq=5)
    # Forecast future volatility and returns
    garch_forecasts = res.forecast(horizon=30, method='simulation')
    garch_vol_forecast = np.sqrt(garch_forecasts.variance.values[-1][0])
    # Simulate future returns
    garch_simulated_returns = garch_forecasts.simulations.values[-1].mean(axis=0) / 100
    # Generate price forecast
    last_price = data['Adj Close'][-1]
    garch_price_forecast = last_price * np.cumprod(np.exp(garch_simulated_returns))

    # ----------------------- Random Forest Model -----------------------
    # Prepare features and target variable
    data['Price_Change'] = data['Adj Close'].pct_change()
    data.dropna(inplace=True)

    features = []
    target = []
    window_size = 10  # Number of past days to consider
    for i in range(window_size, len(data)):
        features.append(data['Price_Change'].iloc[i - window_size:i].values)
        target.append(data['Price_Change'].iloc[i])
    features = np.array(features)
    target = np.array(target)

    # Split data into training and testing sets
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train_rf, y_train_rf)

    # Forecast future price changes
    rf_price_changes = []
    last_window = features[-1]
    for _ in range(30):
        pred_change = rf_model.predict(last_window.reshape(1, -1))[0]
        rf_price_changes.append(pred_change)
        last_window = np.append(last_window[1:], pred_change)

    # Generate price forecast
    rf_price_forecast = last_price * np.cumprod(1 + np.array(rf_price_changes))

    # ----------------------- LSTM Model -----------------------
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

    look_back = 60  # You can adjust the look-back period
    X_lstm, y_lstm = create_dataset(scaled_data, look_back)

    # Reshape input to be [samples, time steps, features]
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    # Split into train and test sets
    train_size = int(len(X_lstm) * 0.8)
    X_train_lstm = X_lstm[:train_size]
    X_test_lstm = X_lstm[train_size:]
    y_train_lstm = y_lstm[:train_size]
    y_test_lstm = y_lstm[train_size:]

    # Build and train LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)

    # Forecast future prices
    lstm_price_forecast = []
    last_sequence = scaled_data[-look_back:]
    for _ in range(30):
        pred_price = lstm_model.predict(last_sequence.reshape(1, look_back, 1))[0][0]
        lstm_price_forecast.append(pred_price)
        last_sequence = np.append(last_sequence[1:], pred_price)
    lstm_price_forecast = scaler.inverse_transform(np.array(lstm_price_forecast).reshape(-1, 1)).flatten()

    # ----------------------- Option Pricing -----------------------
    # Get current stock price
    S = data['Adj Close'].iloc[-1]

    # Define option parameters
    K = S  # At-the-money option
    r = 0.01  # Risk-free interest rate (1%)

    # Get option chain data
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    expiry_date = options_dates[0]  # Earliest expiry date
    options_chain = stock.option_chain(expiry_date)
    calls = options_chain.calls

    # Find the call option with strike price closest to K
    call_option = calls.iloc[(calls['strike'] - K).abs().argsort()[:1]]
    market_option_price = call_option['lastPrice'].values[0]
    market_option_iv = call_option['impliedVolatility'].values[0]
    T = (pd.to_datetime(expiry_date) - data.index[-1]).days / 365  # Time to maturity in years

    # Convert forecasted volatilities to decimals
    sigma_garch = garch_vol_forecast / 100
    sigma_rf = data['Log_Returns'].std() * np.sqrt(252)  # Use historical volatility
    sigma_lstm = data['Log_Returns'].std() * np.sqrt(252)  # Use historical volatility

    # Calculate option prices
    price_garch = black_scholes(S, K, T, r, sigma_garch, option_type='call')
    price_rf = black_scholes(S, K, T, r, sigma_rf, option_type='call')
    price_lstm = black_scholes(S, K, T, r, sigma_lstm, option_type='call')


    # ----------------------- Plotting Price Forecasts -----------------------
    fg = plt.figure(figsize=(14, 7))

    # Plot historical prices
    plt.plot(data['Adj Close'][-100:], label='Historical Price')

    # Future dates for plotting
    future_dates = pd.date_range(start=data.index[-1], periods=31, freq='B')[1:]

    # GARCH Price Forecast
    plt.plot(future_dates, garch_price_forecast, label='GARCH Price Forecast')

    # Random Forest Price Forecast
    plt.plot(future_dates, rf_price_forecast, label='Random Forest Price Forecast')

    # LSTM Price Forecast
    plt.plot(future_dates, lstm_price_forecast, label='LSTM Price Forecast')

    plt.title(f"{ticker} Price Forecasts")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return market_option_price, price_garch, price_rf, price_lstm, market_option_iv, sigma_garch, sigma_rf, sigma_lstm, fg

