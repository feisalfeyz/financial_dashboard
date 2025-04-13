import io
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


from prophet import Prophet

# ----------- SETUP ----------
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>📈 Financial Data Analysis Dashboard</h1>", unsafe_allow_html=True)
st.title("By Feisal Mussa Vuai, Economics and Financial Analyst")

# ----------- SIDEBAR ----------
section = st.sidebar.radio("📌 Choose Section", [
    "📊 Portfolio Simulator", 
    "📉 Volatility & Risk", 
    "📁 Upload CSV", 
    "📤 Export Results",
    "🔮 Forecast Prices"
])

default_tickers = ['AAPL', 'TSLA', 'BTC-USD', 'EURUSD=X', '^GSPC']
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# ----------- PORTFOLIO SIMULATOR ----------
if section == "📊 Portfolio Simulator":
    st.header("📊 Portfolio Allocation Simulator")
    tickers = st.multiselect("Select Assets", default_tickers, default=['AAPL', 'TSLA'])

    if tickers:
        weights = []
        for t in tickers:
            weights.append(st.slider(f"Weight for {t}", 0.0, 1.0, 1 / len(tickers)))

        weights = np.array(weights)
        weights /= weights.sum()  # Normalize

        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
        returns = data.pct_change().dropna()
        portfolio_returns = returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        st.subheader("💹 Cumulative Portfolio Return")
        st.line_chart(cumulative_returns)

        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)

        st.metric("📈 Annualized Return", f"{ann_return:.2%}")
        st.metric("📉 Annualized Volatility", f"{ann_vol:.2%}")
    else:
        st.warning("Please select at least one asset.")

# ----------- VOLATILITY & RISK ----------

elif section == "📉 Volatility & Risk":
    st.header("📉 Risk Metrics & Volatility")

    tickers = st.multiselect("Select Assets", default_tickers, default=['AAPL', 'TSLA'], key="risk_tickers")

    if tickers:
        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
        returns = data.pct_change().dropna()

        weights = np.array([1 / len(tickers)] * len(tickers))
        weights /= weights.sum()

        portfolio_returns = returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Rolling volatility
        rolling_vol = returns.rolling(window=21).std()
        portfolio_rolling_vol = rolling_vol.dot(weights)

        st.subheader("📉 Portfolio vs. Volatility")
        fig, ax = plt.subplots()
        ax.plot(cumulative_returns.index, cumulative_returns, label="Portfolio", color='blue')
        ax.plot(portfolio_rolling_vol.index, portfolio_rolling_vol, label="Rolling Volatility", color='orange', alpha=0.6)
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        # Max drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_dd = drawdown.min()

        # Beta vs. S&P 500
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].pct_change().dropna()

        # ✅ FIX: Flatten inputs to 1D for DataFrame creation
        portfolio_returns_1d = np.array(portfolio_returns).ravel()
        sp500_1d = np.array(sp500).ravel()

        aligned = pd.DataFrame({
            'Portfolio': portfolio_returns_1d,
            'SP500': sp500_1d
        }).dropna()

        beta = np.cov(aligned['Portfolio'], aligned['SP500'])[0, 1] / np.var(aligned['SP500'])

        st.metric("📉 Max Drawdown", f"{max_dd:.2%}")
        st.metric("📊 Beta vs S&P 500", f"{beta:.2f}")

# ----------- CSV UPLOAD ----------
elif section == "📁 Upload CSV":
    st.header("📁 Upload Your Financial Data (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

# ----------- EXPORT RESULTS ----------
elif section == "📤 Export Results":
    st.header("📤 Export Portfolio Data to Excel")
    tickers = st.multiselect("Select Assets to Export", default_tickers, default=['AAPL', 'TSLA'], key="export_tickers")

    if tickers:
        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()

        def to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=True, sheet_name='Portfolio Data')
            return output.getvalue()

        st.dataframe(data.tail())

        st.download_button(
            label="📥 Download Excel",
            data=to_excel(data),
            file_name="portfolio_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Please select at least one asset.")

# ----------- FORECASTING ----------

elif section == "🔮 Forecast Prices":
    st.header("🔮 Asset Price Forecasting with Prophet")

    ticker = st.selectbox("Select an asset to forecast", default_tickers, index=0)
    forecast_period = st.slider("Forecast Period (Days)", 30, 365, 90)

    try:
        # Download data
        df = yf.download(ticker, start="2022-01-01", end=pd.to_datetime("today"), group_by="ticker")

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # Try to get the close price
        close_cols = [col for col in df.columns if 'Close' in col]
        if not close_cols:
            st.error("No 'Close' column found for forecasting.")
        else:
            close_col = close_cols[0]
            df.reset_index(inplace=True)

            if 'Date' not in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            prophet_df = df[['Date', close_col]].copy()
            prophet_df.rename(columns={'Date': 'ds', close_col: 'y'}, inplace=True)

            prophet_df.dropna(subset=['ds', 'y'], inplace=True)
            prophet_df = prophet_df.sort_values('ds')

            if len(prophet_df) < 100:
                st.warning("Not enough historical data for reliable forecasting.")
            elif prophet_df['y'].std() < 1e-3:
                st.warning("Selected column has too little variation for meaningful forecasting.")
            elif len(prophet_df) < (forecast_period / 2):
                st.warning("Too little data for this long of a forecast. Consider reducing forecast period.")
            else:
                model = Prophet(daily_seasonality=True)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)

                st.subheader(f"📈 {ticker} Close Price Forecast (Next {forecast_period} Days)")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                st.subheader("📊 Forecast Components")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                st.subheader("📄 Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
