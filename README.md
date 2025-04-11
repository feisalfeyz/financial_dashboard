# 📈 Financial Analytics Dashboard

A fully interactive financial dashboard built with **Python**, **Streamlit**, and **Prophet** that allows users to:

✅ Simulate portfolio allocations  
✅ Analyze risk and volatility metrics  
✅ Forecast asset prices using Prophet  
✅ Upload custom CSV data  
✅ Export results to Excel  

---

## 🚀 Live Demo

Try the app live here:  
👉 [https://your-streamlit-url.streamlit.app](https://your-streamlit-url.streamlit.app)

---

## 📌 Features

### 1. 💼 Portfolio Allocation Simulator
- Allocate weights across multiple assets
- Visualize cumulative portfolio returns
- Annualized return and volatility metrics

### 2. 📉 Volatility & Risk Metrics
- Rolling 21-day volatility overlay
- Max Drawdown calculation
- Beta against S&P 500

### 3. 🔮 Price Forecasting with Prophet
- Select any asset (e.g. AAPL, BTC-USD, EURUSD=X)
- Forecast future prices up to 365 days
- Visualize forecast trends + uncertainty intervals

### 4. 📁 CSV Upload & Viewer
- Upload your own financial time series
- Preview and work with your own data

### 5. 📤 Export Results to Excel
- Download selected asset data as `.xlsx`

---

## 🛠 Tech Stack

- **Python 3.12**
- [Streamlit](https://streamlit.io/)
- [yFinance](https://github.com/ranaroussi/yfinance)
- [Prophet](https://facebook.github.io/prophet/)
- NumPy, Pandas, Matplotlib, XlsxWriter

---

## 📂 How to Run Locally

1. Clone the repo

```bash
git clone https://github.com/your-username/financial-dashboard.git
cd financial-dashboard
