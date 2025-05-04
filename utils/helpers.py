import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

def format_currency(value, currency="$"):
    """
    Format a value as currency
    
    Args:
        value (float): Value to format
        currency (str): Currency symbol
        
    Returns:
        str: Formatted currency string
    """
    if value >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    else:
        return f"{currency}{value:.2f}"

def format_percentage(value):
    """
    Format a value as percentage
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value*100:.2f}%"

def format_large_number(value):
    """
    Format a large number with K, M, B suffixes
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted number string
    """
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"

def get_timeframe_days(timeframe):
    """
    Convert timeframe string to number of days
    
    Args:
        timeframe (str): Timeframe string (e.g., '1w', '1m', '3m', '6m', '1y')
        
    Returns:
        int: Number of days
    """
    if timeframe.endswith('d'):
        return int(timeframe[:-1])
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 7
    elif timeframe.endswith('m'):
        return int(timeframe[:-1]) * 30
    elif timeframe.endswith('y'):
        return int(timeframe[:-1]) * 365
    else:
        return 30  # Default to 30 days

def get_date_range(timeframe):
    """
    Get start and end dates for a timeframe
    
    Args:
        timeframe (str): Timeframe string (e.g., '1w', '1m', '3m', '6m', '1y')
        
    Returns:
        tuple: Tuple with start and end dates
    """
    days = get_timeframe_days(timeframe)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

def load_lottie_url(url):
    """
    Load Lottie animation from URL
    
    Args:
        url (str): URL to Lottie animation JSON
        
    Returns:
        dict: Lottie animation JSON
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"Error loading Lottie animation: {e}")
        return None

def get_crypto_icon_url(coin_id):
    """
    Get cryptocurrency icon URL from CoinGecko
    
    Args:
        coin_id (str): CoinGecko coin ID
        
    Returns:
        str: URL to cryptocurrency icon
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        response = requests.get(url)
        data = response.json()
        return data.get('image', {}).get('small', '')
    except Exception as e:
        print(f"Error fetching coin icon: {e}")
        return ""

def get_plotly_fig_as_json(fig):
    """
    Convert Plotly figure to JSON
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure
        
    Returns:
        str: JSON string
    """
    return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

def get_dataframe_as_csv(df):
    """
    Convert DataFrame to CSV string
    
    Args:
        df (pandas.DataFrame): DataFrame
        
    Returns:
        str: CSV string
    """
    return df.to_csv(index=True)

def get_dataframe_as_excel(df):
    """
    Convert DataFrame to Excel bytes
    
    Args:
        df (pandas.DataFrame): DataFrame
        
    Returns:
        bytes: Excel file as bytes
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    return output.getvalue()

def get_dataframe_download_link(df, filename, format="csv"):
    """
    Create download link for DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame
        filename (str): Filename
        format (str): Format ('csv' or 'excel')
        
    Returns:
        str: HTML download link
    """
    if format == "csv":
        csv = get_dataframe_as_csv(df)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    elif format == "excel":
        excel = get_dataframe_as_excel(df)
        b64 = base64.b64encode(excel).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
    else:
        href = ""
    
    return href

def calculate_portfolio_value(holdings, prices):
    """
    Calculate portfolio value
    
    Args:
        holdings (dict): Dictionary with asset holdings
        prices (dict): Dictionary with asset prices
        
    Returns:
        float: Portfolio value
    """
    value = 0
    for asset, amount in holdings.items():
        if asset in prices:
            value += amount * prices[asset]
    return value

def calculate_portfolio_weights(holdings, prices):
    """
    Calculate portfolio weights
    
    Args:
        holdings (dict): Dictionary with asset holdings
        prices (dict): Dictionary with asset prices
        
    Returns:
        dict: Dictionary with portfolio weights
    """
    total_value = calculate_portfolio_value(holdings, prices)
    weights = {}
    
    if total_value > 0:
        for asset, amount in holdings.items():
            if asset in prices:
                weights[asset] = (amount * prices[asset]) / total_value
    
    return weights

def calculate_portfolio_returns(holdings, historical_prices):
    """
    Calculate historical portfolio returns
    
    Args:
        holdings (dict): Dictionary with asset holdings
        historical_prices (dict): Dictionary with historical prices for each asset
        
    Returns:
        pandas.Series: Historical portfolio returns
    """
    # Create DataFrame with historical prices
    price_df = pd.DataFrame()
    
    for asset, amount in holdings.items():
        if asset in historical_prices:
            # Add price column for asset
            price_df[asset] = historical_prices[asset]['price']
    
    # Calculate portfolio value over time
    portfolio_value = pd.Series(0, index=price_df.index)
    
    for asset, amount in holdings.items():
        if asset in price_df.columns:
            portfolio_value += amount * price_df[asset]
    
    # Calculate returns
    portfolio_returns = portfolio_value.pct_change()
    
    return portfolio_returns

def get_risk_free_rate():
    """
    Get current risk-free rate (US 10-Year Treasury Yield)
    
    Returns:
        float: Risk-free rate
    """
    # This is a placeholder - in a real application, you would fetch this from an API
    return 0.02  # 2% as a default

def get_market_index_returns(timeframe='1y'):
    """
    Get market index returns (e.g., S&P 500)
    
    Args:
        timeframe (str): Timeframe string
        
    Returns:
        pandas.Series: Market index returns
    """
    # This is a placeholder - in a real application, you would fetch this from an API
    days = get_timeframe_days(timeframe)
    dates = pd.date_range(end=datetime.now(), periods=days)
    returns = np.random.normal(0.0005, 0.01, days)  # Simulated daily returns
    return pd.Series(returns, index=dates)

def get_crypto_market_cap_data():
    """
    Get cryptocurrency market cap data
    
    Returns:
        dict: Dictionary with market cap data
    """
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url)
        data = response.json()
        
        return {
            'total_market_cap': data['data']['total_market_cap']['usd'],
            'total_volume': data['data']['total_volume']['usd'],
            'market_cap_percentage': data['data']['market_cap_percentage'],
            'market_cap_change_percentage_24h_usd': data['data']['market_cap_change_percentage_24h_usd']
        }
    except Exception as e:
        print(f"Error fetching market cap data: {e}")
        return {
            'total_market_cap': 0,
            'total_volume': 0,
            'market_cap_percentage': {},
            'market_cap_change_percentage_24h_usd': 0
        }
