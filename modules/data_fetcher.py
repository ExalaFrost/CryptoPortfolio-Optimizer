import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from pycoingecko import CoinGeckoAPI

class CryptoDataFetcher:
    """
    A class to fetch cryptocurrency data from CoinGecko API
    """
    
    def __init__(self):
        """Initialize the CoinGecko API client"""
        self.cg = CoinGeckoAPI()
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # Cache duration in seconds (5 minutes)
    
    def get_top_coins(self, limit=100):
        """
        Get the top cryptocurrencies by market cap
        
        Args:
            limit (int): Number of top coins to retrieve
            
        Returns:
            list: List of coin dictionaries with id, symbol, and name
        """
        cache_key = f"top_coins_{limit}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch top coins by market cap
            coins = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=limit,
                page=1,
                sparkline=False
            )
            
            # Extract relevant information
            result = [{'id': coin['id'], 'symbol': coin['symbol'].upper(), 'name': coin['name']} 
                     for coin in coins]
            
            # Update cache
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return result
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return []
    
    def get_coin_history(self, coin_id, days=30, vs_currency='usd'):
        """
        Get historical price data for a specific coin
        
        Args:
            coin_id (str): CoinGecko coin ID
            days (int): Number of days of historical data
            vs_currency (str): Currency to get prices in
            
        Returns:
            pandas.DataFrame: DataFrame with historical price data
        """
        cache_key = f"history_{coin_id}_{days}_{vs_currency}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch historical market data
            market_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Extract price data
            prices = market_data['prices']
            volumes = market_data['total_volumes']
            market_caps = market_data['market_caps']
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['volume'] = [v[1] for v in volumes]
            df['market_cap'] = [m[1] for m in market_caps]
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate daily returns
            df['daily_return'] = df['price'].pct_change()
            
            # Update cache
            self.cache[cache_key] = df
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return df
        except Exception as e:
            print(f"Error fetching historical data for {coin_id}: {e}")
            return pd.DataFrame()
    
    def get_current_prices(self, coin_ids):
        """
        Get current prices for a list of coins
        
        Args:
            coin_ids (list): List of CoinGecko coin IDs
            
        Returns:
            dict: Dictionary mapping coin IDs to current prices in USD
        """
        if not coin_ids:
            return {}
            
        cache_key = f"prices_{'_'.join(sorted(coin_ids))}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch current prices
            prices = self.cg.get_price(
                ids=coin_ids,
                vs_currencies='usd'
            )
            
            # Extract USD prices
            result = {coin_id: data['usd'] for coin_id, data in prices.items()}
            
            # Update cache
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return result
        except Exception as e:
            print(f"Error fetching current prices: {e}")
            return {}
    
    def get_portfolio_historical_data(self, coin_ids, days=30):
        """
        Get historical data for multiple coins for portfolio analysis
        
        Args:
            coin_ids (list): List of CoinGecko coin IDs
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame: DataFrame with historical returns for all coins
        """
        if not coin_ids:
            return pd.DataFrame()
            
        cache_key = f"portfolio_{'_'.join(sorted(coin_ids))}_{days}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Create an empty DataFrame to store returns
            all_returns = pd.DataFrame()
            
            # Fetch historical data for each coin
            for coin_id in coin_ids:
                coin_data = self.get_coin_history(coin_id, days)
                if not coin_data.empty:
                    # Extract daily returns and rename column
                    returns = coin_data['daily_return'].rename(coin_id)
                    
                    # Add to the combined DataFrame
                    if all_returns.empty:
                        all_returns = pd.DataFrame(returns)
                    else:
                        all_returns = all_returns.join(returns, how='outer')
            
            # Fill missing values with 0
            all_returns.fillna(0, inplace=True)
            
            # Update cache
            self.cache[cache_key] = all_returns
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return all_returns
        except Exception as e:
            print(f"Error fetching portfolio historical data: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self):
        """
        Get overall market sentiment data
        
        Returns:
            dict: Dictionary with market sentiment metrics
        """
        cache_key = "market_sentiment"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch global market data
            global_data = self.cg.get_global()
            
            # Extract relevant sentiment metrics
            sentiment = {
                'market_cap_change_percentage_24h': global_data.get('market_cap_change_percentage_24h_usd', 0),
                'market_cap_percentage': global_data.get('market_cap_percentage', {}),
                'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume': global_data.get('total_volume', {}).get('usd', 0)
            }
            
            # Update cache
            self.cache[cache_key] = sentiment
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return sentiment
        except Exception as e:
            print(f"Error fetching market sentiment: {e}")
            return {}
    
    def get_price_data(self, coin_ids, days=365, vs_currency='usd'):
        """
        Get historical price data for multiple coins
        
        Args:
            coin_ids (list): List of CoinGecko coin IDs
            days (int): Number of days of historical data
            vs_currency (str): Currency to get prices in
            
        Returns:
            pandas.DataFrame: DataFrame with historical price data for all coins
        """
        if not coin_ids:
            return pd.DataFrame()
            
        cache_key = f"price_data_{'_'.join(sorted(coin_ids))}_{days}_{vs_currency}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Create an empty DataFrame to store prices
            all_prices = pd.DataFrame()
            
            # Fetch historical data for each coin
            for coin_id in coin_ids:
                coin_data = self.get_coin_history(coin_id, days, vs_currency)
                if not coin_data.empty:
                    # Extract price and rename column
                    prices = coin_data['price'].rename(coin_id)
                    
                    # Add to the combined DataFrame
                    if all_prices.empty:
                        all_prices = pd.DataFrame(prices)
                    else:
                        all_prices = all_prices.join(prices, how='outer')
            
            # Fill missing values with forward fill then backward fill
            all_prices.fillna(method='ffill', inplace=True)
            all_prices.fillna(method='bfill', inplace=True)
            
            # Update cache
            self.cache[cache_key] = all_prices
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return all_prices
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame()
