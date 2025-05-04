# 🚀 CryptoPortfolio Optimizer

CryptoPortfolio Optimizer is a powerful, interactive web application built with Streamlit that helps users optimize their cryptocurrency portfolios using Modern Portfolio Theory, assess risk with advanced metrics, and analyze market sentiment.

![CryptoPortfolio Optimizer](https://cdn-icons-png.flaticon.com/512/2091/2091665.png)

## 🔍 Features

### ✅ **Portfolio Optimization**
- **Modern Portfolio Theory** implementation for optimal asset allocation
- **Efficient Frontier** visualization to understand risk-return tradeoffs
- **Multiple optimization strategies** (Max Sharpe Ratio, Min Volatility, Max Return)
- **Real-time data** from CoinGecko API for the top 100 cryptocurrencies

### 📊 **Risk Assessment**
- **Key risk metrics** calculation (Sharpe ratio, Sortino ratio, VaR, CVaR)
- **Monte Carlo simulation** for portfolio performance prediction
- **Drawdown analysis** to understand historical worst-case scenarios
- **Returns distribution** visualization with statistical analysis

### 🧠 **Market Sentiment Analysis**
- **Fear & Greed Index** tracking for overall market sentiment
- **Cryptocurrency-specific sentiment** analysis with radar charts
- **Sentiment comparison** between different cryptocurrencies
- **Historical sentiment trends** visualization

## 🎨 User Experience Highlights

- 🧭 **Intuitive Navigation**: Easy-to-follow workflows for every task
- 💡 **Help Sections**: Built-in tips for beginners and pros alike
- 🎬 **Lottie Animations**: Crypto-themed visual enhancements
- 🌙 **Dark-Themed UI**: Clean and visually appealing design
- 📥 **Downloadable Results**: Export your optimized portfolios and risk reports

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit**
- **pandas / numpy**
- **plotly**
- **PyPortfolioOpt**
- **CoinGecko API**
- **streamlit-lottie**

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ExalaFrost/CryptoPortfolio-Optimizer.git
cd CryptoPortfolio-Optimizer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📋 Project Structure

```
CryptoPortfolio-Optimizer/
├── app.py                  # Main application file
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── modules/                # Application modules
│   ├── data_fetcher.py     # API integration for crypto data
│   ├── portfolio_opt.py    # Portfolio optimization algorithms
│   ├── risk_assessment.py  # Risk metrics and simulations
│   └── sentiment.py        # Sentiment analysis functions
└── utils/                  # Utility functions
    ├── visualization.py    # Plotting functions
    └── helpers.py          # Helper functions
```

## 🚀 How to Use

1. **Home Page**: View market overview and key features
2. **Portfolio Optimization**:
   - Select cryptocurrencies for your portfolio
   - Choose analysis timeframe
   - Set optimization parameters
   - View and download optimal portfolio allocation
3. **Risk Assessment**:
   - Analyze portfolio risk metrics
   - Run Monte Carlo simulations
   - View drawdown analysis and returns distribution
4. **Market Sentiment**:
   - Track Fear & Greed Index
   - Analyze cryptocurrency-specific sentiment
   - Compare sentiment across different assets

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not financial advice, and you should always do your own research before making investment decisions.

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome! Please feel free to submit a pull request or open an issue.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, feedback, or suggestions, please contact:
- Email: i222272@nu.edu.pk
- GitHub: [ExalaFrost](https://github.com/ExalaFrost)
