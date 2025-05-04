import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import requests
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO

# Import custom modules
from modules.data_fetcher import CryptoDataFetcher
from modules.portfolio_opt import PortfolioOptimizer
from modules.risk_assessment import RiskAssessor
from modules.sentiment import SentimentAnalyzer
from modules.price_prediction import run_price_prediction
from modules.clustering import run_clustering
from utils.visualization import (
    plot_price_history, plot_returns_heatmap, plot_portfolio_composition,
    plot_risk_return_scatter, plot_efficient_frontier_with_assets,
    plot_monte_carlo_simulation, plot_risk_metrics_radar, plot_drawdown_chart
)
from utils.helpers import (
    format_currency, format_percentage, format_large_number,
    get_timeframe_days, get_date_range, load_lottie_url,
    get_crypto_icon_url, get_dataframe_download_link,
    calculate_portfolio_value, calculate_portfolio_weights,
    get_risk_free_rate, get_crypto_market_cap_data
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="CryptoPortfolio Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = CryptoDataFetcher()

if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

if 'portfolio_returns' not in st.session_state:
    st.session_state.portfolio_returns = None

if 'portfolio_weights' not in st.session_state:
    st.session_state.portfolio_weights = None

if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = None

if 'selected_coins' not in st.session_state:
    st.session_state.selected_coins = []

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Function to load crypto animation
def load_crypto_animation():
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_kcsr6fcp.json"
    return load_lottie_url(lottie_url)

# Function to get crypto icon
def get_crypto_icon(coin_id):
    icon_url = get_crypto_icon_url(coin_id)
    if icon_url:
        response = requests.get(icon_url)
        return Image.open(BytesIO(response.content))
    return None

# Sidebar navigation
st.sidebar.title("CryptoPortfolio Optimizer")

# Add crypto animation to sidebar
try:
    from streamlit_lottie import st_lottie
    crypto_animation = load_crypto_animation()
    if crypto_animation:
        with st.sidebar:
            st_lottie(crypto_animation, height=200, key="crypto_animation")
except ImportError:
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2091/2091665.png", width=100)

# Navigation options
nav_option = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Portfolio Optimization", "📈 Risk Assessment", "🔍 Market Sentiment", "🧠 ML Analysis", "ℹ️ About"]
)

# Sidebar - Help & Info
with st.sidebar.expander("💡 Help & Tips"):
    st.markdown("""
    **How to use this app:**
    1. Select cryptocurrencies to analyze
    2. Choose your analysis timeframe
    3. Optimize your portfolio based on risk/return preferences
    4. Analyze risk metrics and market sentiment
    5. Use ML tools for price prediction and clustering
    6. Download your optimized portfolio allocation
    """)

# Main content area
if nav_option == "🏠 Home":
    st.title("🚀 CryptoPortfolio Optimizer")
    st.markdown("### Your All-in-One Cryptocurrency Portfolio Optimization Tool")
    
    # Market overview section
    st.subheader("📊 Crypto Market Overview")
    
    # Fetch market data
    market_data = get_crypto_market_cap_data()
    
    # Display market metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Market Cap",
            format_currency(market_data['total_market_cap']),
            f"{market_data['market_cap_change_percentage_24h_usd']:.2f}%"
        )
    
    with col2:
        st.metric(
            "24h Trading Volume",
            format_currency(market_data['total_volume'])
        )
    
    with col3:
        # Get Fear & Greed Index
        fear_greed = st.session_state.sentiment_analyzer.get_fear_greed_index()
        if fear_greed:
            st.metric(
                "Fear & Greed Index",
                f"{fear_greed['value']} - {fear_greed['value_classification']}"
            )
    
    # Display market dominance pie chart
    if market_data['market_cap_percentage']:
        fig = px.pie(
            values=list(market_data['market_cap_percentage'].values()),
            names=list(market_data['market_cap_percentage'].keys()),
            title='Market Dominance',
            template='plotly_dark',
            hole=0.4
        )
        fig.update_layout(legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    
    # App features section
    st.markdown("---")
    st.subheader("🛠️ Key Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        #### 📊 Portfolio Optimization
        - Modern Portfolio Theory implementation
        - Efficient Frontier visualization
        - Optimal portfolio weight allocation
        - Multiple optimization strategies
        """)
        
        st.markdown("""
        #### 📈 Risk Assessment
        - Key risk metrics calculation
        - Monte Carlo simulation
        - Drawdown analysis
        - Value at Risk (VaR) calculation
        """)
    
    with feature_col2:
        st.markdown("""
        #### 🔍 Market Sentiment Analysis
        - Fear & Greed Index tracking
        - Cryptocurrency sentiment analysis
        - Market trend visualization
        - Sentiment comparison between assets
        """)
        
        st.markdown("""
        #### 🧠 Machine Learning Analysis
        - Price prediction with Prophet
        - Cryptocurrency clustering with K-means
        - Interactive ML visualizations
        """)
    
    # Get started button
    st.markdown("---")
    st.subheader("Ready to optimize your crypto portfolio?")
    
    get_started_col1, get_started_col2 = st.columns([2, 1])
    
    with get_started_col1:
        st.markdown("""
        Click the button to start optimizing your cryptocurrency portfolio.
        You'll be able to select cryptocurrencies, analyze their performance,
        and get optimal portfolio allocations based on your risk preferences.
        """)
    
    with get_started_col2:
        if st.button("🚀 Get Started", key="get_started"):
            st.session_state.nav_option = "📊 Portfolio Optimization"
            st.experimental_rerun()

elif nav_option == "📊 Portfolio Optimization":
    st.title("📊 Portfolio Optimization")
    
    # Step 1: Select cryptocurrencies
    st.subheader("Step 1: Select Cryptocurrencies")
    
    # Fetch top coins
    with st.spinner("Fetching top cryptocurrencies..."):
        top_coins = st.session_state.data_fetcher.get_top_coins(limit=100)
    
    # Create a list of coin options
    coin_options = [f"{coin['name']} ({coin['symbol']})" for coin in top_coins]
    coin_ids = [coin['id'] for coin in top_coins]
    
    # Allow user to select coins
    selected_options = st.multiselect(
        "Select cryptocurrencies for your portfolio (2-10 recommended):",
        options=coin_options,
        default=coin_options[:5] if len(coin_options) >= 5 else coin_options
    )
    
    # Get selected coin IDs
    selected_indices = [coin_options.index(option) for option in selected_options]
    selected_coin_ids = [coin_ids[i] for i in selected_indices]
    
    # Store selected coins in session state
    st.session_state.selected_coins = selected_coin_ids
    
    # Step 2: Select timeframe and fetch data
    if selected_coin_ids:
        st.subheader("Step 2: Select Analysis Timeframe")
        
        timeframe = st.select_slider(
            "Select historical data timeframe:",
            options=["1m", "3m", "6m", "1y", "2y"],
            value="1y"
        )
        
        days = get_timeframe_days(timeframe)
        
        # Fetch historical data
        if st.button("Fetch Data and Analyze", key="fetch_data"):
            with st.spinner(f"Fetching {days} days of historical data for {len(selected_coin_ids)} cryptocurrencies..."):
                # Get historical returns data
                returns_data = st.session_state.data_fetcher.get_portfolio_historical_data(
                    coin_ids=selected_coin_ids,
                    days=days
                )
                
                # Store returns data in session state
                st.session_state.portfolio_returns = returns_data
                
                # Success message
                st.success(f"Successfully fetched data for {len(selected_coin_ids)} cryptocurrencies over {days} days!")
        
        # Step 3: Portfolio Optimization
        if st.session_state.portfolio_returns is not None and not st.session_state.portfolio_returns.empty:
            st.subheader("Step 3: Portfolio Optimization")
            
            # Optimization parameters
            # Using fixed values for optimization strategy, expected returns method, and risk model
            optimization_criterion = "Maximum Sharpe Ratio"
            expected_returns_method = "Mean Historical Return"
            risk_model_method = "Sample Covariance"
            
            # Display the fixed strategies
            st.info("Using Maximum Sharpe Ratio optimization with Mean Historical Return and Sample Covariance")
            
            # Only allow adjusting the risk-free rate
            risk_free_rate = st.slider(
                "Risk-Free Rate (%):",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.1
            ) / 100
            
            # Map selection to method names (keeping these for code compatibility)
            criterion_map = {
                "Maximum Sharpe Ratio": "sharpe",
                "Minimum Volatility": "min_volatility",
                "Maximum Return": "max_return"
            }
            
            returns_method_map = {
                "Mean Historical Return": "mean_historical_return",
                "Capital Asset Pricing Model (CAPM)": "capm_return"
            }
            
            risk_model_map = {
                "Sample Covariance": "sample_cov",
                "Semicovariance": "semicovariance",
                "Exponential Covariance": "exp_cov"
            }
            
            # Run optimization
            if st.button("Optimize Portfolio", key="optimize_portfolio"):
                with st.spinner("Optimizing portfolio..."):
                    # Create portfolio optimizer
                    optimizer = PortfolioOptimizer(st.session_state.portfolio_returns)
                    
                    # Store optimizer in session state
                    st.session_state.optimizer = optimizer
                    
                    # Run optimization
                    optimization_results = optimizer.optimize_portfolio(
                        risk_free_rate=risk_free_rate,
                        optimization_criterion=criterion_map[optimization_criterion],
                        expected_returns_method=returns_method_map[expected_returns_method],
                        risk_model_method=risk_model_map[risk_model_method]
                    )
                    
                    # Store results in session state
                    st.session_state.optimization_results = optimization_results
                    st.session_state.portfolio_weights = optimization_results['weights']
                    
                    # Success message
                    st.success("Portfolio optimization complete!")
            
            # Display optimization results
            if st.session_state.optimization_results is not None:
                st.subheader("Optimization Results")
                
                results = st.session_state.optimization_results
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Expected Annual Return",
                        f"{results['expected_annual_return']*100:.2f}%"
                    )
                
                with metric_col2:
                    st.metric(
                        "Annual Volatility",
                        f"{results['annual_volatility']*100:.2f}%"
                    )
                
                with metric_col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results['sharpe_ratio']:.2f}"
                    )
                
                # Display portfolio weights
                st.subheader("Optimal Portfolio Allocation")
                
                # Create pie chart
                fig = plot_portfolio_composition(results['weights'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Display weights in a table
                weights_df = pd.DataFrame({
                    'Cryptocurrency': list(results['weights'].keys()),
                    'Weight (%)': [f"{w*100:.2f}%" for w in results['weights'].values()]
                })
                st.table(weights_df)
                
                # Plot efficient frontier
                st.subheader("Efficient Frontier")
                
                # Calculate efficient frontier
                ef_returns, ef_volatilities = st.session_state.optimizer.calculate_efficient_frontier(
                    risk_free_rate=risk_free_rate,
                    expected_returns_method=returns_method_map[expected_returns_method],
                    risk_model_method=risk_model_map[risk_model_method]
                )
                
                # Plot efficient frontier
                fig = st.session_state.optimizer.plot_efficient_frontier(
                    risk_free_rate=risk_free_rate,
                    expected_returns_method=returns_method_map[expected_returns_method],
                    risk_model_method=risk_model_map[risk_model_method]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot correlation matrix
                st.subheader("Asset Correlation Matrix")
                fig = st.session_state.optimizer.plot_correlation_matrix()
                st.plotly_chart(fig, use_container_width=True)
                
                # Download portfolio allocation
                st.subheader("Download Portfolio Allocation")
                
                # Create DataFrame with allocation
                allocation_df = pd.DataFrame({
                    'Cryptocurrency': list(results['weights'].keys()),
                    'Weight': list(results['weights'].values()),
                    'Expected Annual Return': results['expected_annual_return'],
                    'Annual Volatility': results['annual_volatility'],
                    'Sharpe Ratio': results['sharpe_ratio']
                })
                
                # Add download button
                csv = allocation_df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio Allocation (CSV)",
                    data=csv,
                    file_name="crypto_portfolio_allocation.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Please select at least one cryptocurrency to continue.")

elif nav_option == "📈 Risk Assessment":
    st.title("📈 Risk Assessment")
    
    # Check if portfolio weights are available
    if st.session_state.portfolio_weights is None:
        st.warning("Please optimize your portfolio first in the Portfolio Optimization section.")
        if st.button("Go to Portfolio Optimization"):
            st.session_state.nav_option = "📊 Portfolio Optimization"
            st.experimental_rerun()
    else:
        # Display portfolio weights
        st.subheader("Current Portfolio Allocation")
        
        # Create pie chart
        fig = plot_portfolio_composition(st.session_state.portfolio_weights)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment parameters
        st.subheader("Risk Assessment Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_free_rate = st.slider(
                "Risk-Free Rate (%):",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="risk_rf_rate"
            ) / 100
            
            confidence_level = st.slider(
                "VaR/CVaR Confidence Level (%):",
                min_value=90,
                max_value=99,
                value=95,
                step=1
            ) / 100
        
        with col2:
            time_horizon = st.slider(
                "Monte Carlo Simulation Time Horizon (Days):",
                min_value=30,
                max_value=365,
                value=252,
                step=30
            )
            
            initial_investment = st.number_input(
                "Initial Investment ($):",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
        
        # Run risk assessment
        if st.button("Run Risk Assessment", key="run_risk_assessment"):
            with st.spinner("Calculating risk metrics..."):
                # Create risk assessor
                risk_assessor = RiskAssessor(
                    st.session_state.portfolio_returns,
                    st.session_state.portfolio_weights
                )
                
                # Calculate risk metrics
                risk_metrics = risk_assessor.get_risk_metrics(
                    risk_free_rate=risk_free_rate
                )
                
                # Store risk metrics in session state
                st.session_state.risk_metrics = risk_metrics
                
                # Success message
                st.success("Risk assessment complete!")
        
        # Display risk metrics
        if st.session_state.risk_metrics is not None:
            st.subheader("Risk Metrics")
            
            metrics = st.session_state.risk_metrics
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Annual Return", f"{metrics['annualized_return']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Value at Risk (95%)", f"{metrics['var_95']*100:.2f}%")
            
            with metric_col2:
                st.metric("Annual Volatility", f"{metrics['annualized_volatility']*100:.2f}%")
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                st.metric("Conditional VaR (95%)", f"{metrics['cvar_95']*100:.2f}%")
            
            with metric_col3:
                st.metric("Maximum Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
            
            # Create risk assessor
            risk_assessor = RiskAssessor(
                st.session_state.portfolio_returns,
                st.session_state.portfolio_weights
            )
            
            # Monte Carlo simulation
            st.subheader("Monte Carlo Simulation")
            
            with st.spinner("Running Monte Carlo simulation..."):
                # Plot Monte Carlo simulation
                fig = risk_assessor.plot_monte_carlo_simulation(
                    num_simulations=1000,
                    time_horizon=time_horizon,
                    initial_investment=initial_investment
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown analysis
            st.subheader("Drawdown Analysis")
            
            # Plot drawdown
            fig = risk_assessor.plot_drawdown()
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            
            # Plot returns distribution
            fig = risk_assessor.plot_returns_distribution()
            st.plotly_chart(fig, use_container_width=True)
            
            # Download risk metrics
            st.subheader("Download Risk Assessment Report")
            
            # Create DataFrame with risk metrics
            risk_df = pd.DataFrame({
                'Metric': [
                    'Annual Return',
                    'Annual Volatility',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Maximum Drawdown',
                    'Value at Risk (95%)',
                    'Conditional VaR (95%)'
                ],
                'Value': [
                    f"{metrics['annualized_return']*100:.2f}%",
                    f"{metrics['annualized_volatility']*100:.2f}%",
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"{metrics['sortino_ratio']:.2f}",
                    f"{metrics['max_drawdown']*100:.2f}%",
                    f"{metrics['var_95']*100:.2f}%",
                    f"{metrics['cvar_95']*100:.2f}%"
                ]
            })
            
            # Add download button
            csv = risk_df.to_csv(index=False)
            st.download_button(
                label="Download Risk Assessment Report (CSV)",
                data=csv,
                file_name="crypto_risk_assessment.csv",
                mime="text/csv"
            )

elif nav_option == "🔍 Market Sentiment":
    st.title("🔍 Market Sentiment Analysis")
    
    # Fear & Greed Index
    st.subheader("Crypto Fear & Greed Index")
    
    # Plot Fear & Greed gauge
    fig = st.session_state.sentiment_analyzer.plot_fear_greed_gauge()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical Fear & Greed Index
    st.subheader("Historical Fear & Greed Index")
    
    # Select timeframe
    timeframe = st.select_slider(
        "Select timeframe:",
        options=["7d", "14d", "30d", "60d", "90d"],
        value="30d"
    )
    
    days = get_timeframe_days(timeframe)
    
    # Plot historical Fear & Greed Index
    fig = st.session_state.sentiment_analyzer.plot_fear_greed_historical(days=days)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Coin-specific sentiment analysis
    st.subheader("Cryptocurrency Sentiment Analysis")
    
    # Check if selected coins are available
    if not st.session_state.selected_coins:
        st.warning("Please select cryptocurrencies in the Portfolio Optimization section first.")
    else:
        # Select coin for sentiment analysis
        coin_id = st.selectbox(
            "Select cryptocurrency for sentiment analysis:",
            options=st.session_state.selected_coins
        )
        
        if coin_id:
            # Plot coin sentiment radar
            fig = st.session_state.sentiment_analyzer.plot_coin_sentiment_radar(coin_id)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment comparison
            st.subheader("Sentiment Comparison")
            
            # Plot sentiment comparison
            fig = st.session_state.sentiment_analyzer.plot_sentiment_comparison(st.session_state.selected_coins)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

elif nav_option == "🧠 ML Analysis":
    st.title("🧠 Machine Learning Analysis")
    
    # Check if data is available
    if st.session_state.portfolio_returns is None:
        st.warning("Please fetch data in the Portfolio Optimization section first.")
        if st.button("Go to Portfolio Optimization"):
            st.session_state.nav_option = "📊 Portfolio Optimization"
            st.experimental_rerun()
    else:
        # Get price data
        price_data = st.session_state.data_fetcher.get_price_data(
            coin_ids=st.session_state.selected_coins,
            days=get_timeframe_days("1y")
        )
        
        # ML Analysis tabs
        ml_tab1, ml_tab2 = st.tabs([
            "📈 Price Prediction", 
            "🔍 Crypto Clustering"
        ])
        
        # Price Prediction tab
        with ml_tab1:
            st.subheader("Cryptocurrency Price Prediction")
            st.markdown("""
            This feature uses Facebook Prophet, a time series forecasting model, to predict future cryptocurrency prices.
            The model analyzes historical price patterns, seasonality, and trends to make predictions.
            """)
            
            # Select cryptocurrency for prediction
            coin_id = st.selectbox(
                "Select cryptocurrency for price prediction:",
                options=st.session_state.selected_coins,
                key="prediction_coin"
            )
            
            # Prediction parameters
            col1, col2 = st.columns(2)
            
            with col1:
                forecast_days = st.slider(
                    "Forecast Horizon (Days):",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=7
                )
            
            with col2:
                seasonality_mode = st.selectbox(
                    "Seasonality Mode:",
                    options=["multiplicative", "additive"],
                    index=0
                )
            
            # Run prediction
            if st.button("Run Price Prediction", key="run_prediction"):
                with st.spinner("Running price prediction model..."):
                    try:
                        # Get price data for selected coin
                        coin_price_data = price_data[[coin_id]]
                        
                        # Run price prediction
                        forecast_fig, components_fig, metrics = run_price_prediction(
                            price_data=coin_price_data,
                            coin_id=coin_id,
                            forecast_days=forecast_days
                        )
                        
                        # Store results in session state
                        st.session_state.forecast_fig = forecast_fig
                        st.session_state.components_fig = components_fig
                        st.session_state.forecast_metrics = metrics
                        
                        # Success message
                        st.success("Price prediction complete!")
                    except Exception as e:
                        st.error(f"Error running price prediction: {str(e)}")
            
            # Display prediction results
            if 'forecast_fig' in st.session_state and 'forecast_metrics' in st.session_state:
                # Display metrics
                st.subheader("Prediction Metrics")
                
                metrics = st.session_state.forecast_metrics
                
                # Display metrics in columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Forecast End Price",
                        f"${metrics['forecast_end_price']:.2f}"
                    )
                
                with metric_col2:
                    st.metric(
                        "Price Change",
                        f"${metrics['price_change']:.2f}",
                        f"{metrics['price_change_pct']:.2f}%"
                    )
                
                with metric_col3:
                    st.metric(
                        "Forecast Volatility",
                        f"{metrics['volatility']:.2f}%"
                    )
                
                # Display forecast plot
                st.subheader("Price Forecast")
                st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
                
                # Display components plot
                st.subheader("Trend Component")
                st.plotly_chart(st.session_state.components_fig, use_container_width=True)
        
        # Crypto Clustering tab
        with ml_tab2:
            st.subheader("Cryptocurrency Clustering Analysis")
            st.markdown("""
            This feature uses K-means clustering to group cryptocurrencies based on their statistical properties.
            The algorithm identifies natural groupings of cryptocurrencies with similar characteristics.
            """)
            
            # Clustering parameters
            n_clusters = st.slider(
                "Number of Clusters:",
                min_value=2,
                max_value=10,
                value=3,
                step=1
            )
            
            # Run clustering
            if st.button("Run Clustering Analysis", key="run_clustering"):
                with st.spinner("Running clustering analysis..."):
                    try:
                        # Run clustering
                        results, elbow_fig, plot_2d, plot_3d, profiles = run_clustering(
                            returns_data=st.session_state.portfolio_returns,
                            n_clusters=n_clusters
                        )
                        
                        # Store results in session state
                        st.session_state.clustering_results = results
                        st.session_state.elbow_fig = elbow_fig
                        st.session_state.cluster_plot_2d = plot_2d
                        st.session_state.cluster_plot_3d = plot_3d
                        st.session_state.cluster_profiles = profiles
                        
                        # Success message
                        st.success("Clustering analysis complete!")
                    except Exception as e:
                        st.error(f"Error running clustering analysis: {str(e)}")
            
            # Display clustering results
            if 'clustering_results' in st.session_state:
                # Display elbow method plot
                st.subheader("Optimal Number of Clusters")
                st.plotly_chart(st.session_state.elbow_fig, use_container_width=True)
                
                # Display 2D cluster plot if available
                if st.session_state.cluster_plot_2d is not None:
                    st.subheader("Cryptocurrency Clusters (2D)")
                    st.plotly_chart(st.session_state.cluster_plot_2d, use_container_width=True)
                
                # Display 3D cluster plot if available
                if st.session_state.cluster_plot_3d is not None:
                    st.subheader("Cryptocurrency Clusters (3D)")
                    st.plotly_chart(st.session_state.cluster_plot_3d, use_container_width=True)
                
                # Display cluster profiles
                st.subheader("Cluster Profiles")
                st.plotly_chart(st.session_state.cluster_profiles, use_container_width=True)
                
                # Display clustering results table
                st.subheader("Clustering Results")
                st.dataframe(st.session_state.clustering_results)
        
        # No Anomaly Detection tab

elif nav_option == "ℹ️ About":
    st.title("ℹ️ About CryptoPortfolio Optimizer")
    
    st.markdown("""
    ## Overview
    
    **CryptoPortfolio Optimizer** is an all-in-one tool for cryptocurrency portfolio optimization and risk assessment. 
    It uses Modern Portfolio Theory and advanced risk metrics to help you build an optimal cryptocurrency portfolio 
    based on your risk preferences.
    
    ## Features
    
    - **Portfolio Optimization**: Optimize your cryptocurrency portfolio using Modern Portfolio Theory
    - **Risk Assessment**: Analyze portfolio risk using various metrics and Monte Carlo simulation
    - **Market Sentiment Analysis**: Track market sentiment and cryptocurrency-specific sentiment
    - **Machine Learning Analysis**: Predict prices, cluster cryptocurrencies, and detect anomalies
    - **Real-time Data**: Fetch real-time cryptocurrency data from CoinGecko API
    
    ## Technologies Used
    
    - **Streamlit**: Web application framework
    - **Pandas/NumPy**: Data manipulation and analysis
    - **Plotly**: Interactive data visualization
    - **PyPortfolioOpt**: Portfolio optimization
    - **CoinGecko API**: Cryptocurrency data
    
    ## How It Works
    
    1. **Data Collection**: The app fetches historical cryptocurrency data from CoinGecko API
    2. **Portfolio Optimization**: Using Modern Portfolio Theory, the app calculates the optimal portfolio weights
    3. **Risk Assessment**: The app calculates various risk metrics and runs Monte Carlo simulations
    4. **Sentiment Analysis**: The app analyzes market sentiment using Fear & Greed Index and other metrics
    5. **Machine Learning Analysis**: The app applies ML algorithms for price prediction, clustering, and anomaly detection
    
    ## Disclaimer
    
    This app is for educational and informational purposes only. It is not financial advice, and you should always do your own research before making investment decisions.
    
    ## Contact
    
    For questions, feedback, or suggestions, please contact:
    
    - Email: i222272@nu.edu.pk
    - GitHub: [ExalaFrost](https://github.com/ExalaFrost)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center>CryptoPortfolio Optimizer v1.0 | Made with ❤️ using Streamlit | © 2025</center>",
    unsafe_allow_html=True
)
