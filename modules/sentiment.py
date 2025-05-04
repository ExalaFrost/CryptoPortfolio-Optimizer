import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

class SentimentAnalyzer:
    """
    A class to analyze cryptocurrency market sentiment
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # Cache duration in seconds (5 minutes)
    
    def get_fear_greed_index(self):
        """
        Get the Fear & Greed Index for crypto market
        
        Returns:
            dict: Dictionary with Fear & Greed Index data
        """
        cache_key = "fear_greed_index"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch Fear & Greed Index data
            url = "https://api.alternative.me/fng/"
            response = requests.get(url)
            data = response.json()
            
            if data.get("metadata", {}).get("error") is None:
                result = {
                    'value': int(data['data'][0]['value']),
                    'value_classification': data['data'][0]['value_classification'],
                    'timestamp': data['data'][0]['timestamp'],
                    'time_until_update': data['data'][0]['time_until_update']
                }
                
                # Update cache
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = time.time() + self.cache_duration
                
                return result
            else:
                print(f"Error fetching Fear & Greed Index: {data['metadata']['error']}")
                return None
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None
    
    def get_fear_greed_historical(self, days=30):
        """
        Get historical Fear & Greed Index data
        
        Args:
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame: DataFrame with historical Fear & Greed Index data
        """
        cache_key = f"fear_greed_historical_{days}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch historical Fear & Greed Index data
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url)
            data = response.json()
            
            if data.get("metadata", {}).get("error") is None:
                # Create DataFrame
                df = pd.DataFrame(data['data'])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Convert value to numeric
                df['value'] = pd.to_numeric(df['value'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Update cache
                self.cache[cache_key] = df
                self.cache_expiry[cache_key] = time.time() + self.cache_duration
                
                return df
            else:
                print(f"Error fetching historical Fear & Greed Index: {data['metadata']['error']}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching historical Fear & Greed Index: {e}")
            return pd.DataFrame()
    
    def get_coin_sentiment(self, coin_id):
        """
        Get sentiment data for a specific coin from CoinGecko
        
        Args:
            coin_id (str): CoinGecko coin ID
            
        Returns:
            dict: Dictionary with coin sentiment data
        """
        cache_key = f"coin_sentiment_{coin_id}"
        
        # Check if data is in cache and not expired
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Fetch coin data from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            response = requests.get(url)
            data = response.json()
            
            # Extract sentiment data
            sentiment = {
                'sentiment_votes_up_percentage': data.get('sentiment_votes_up_percentage', 0),
                'sentiment_votes_down_percentage': data.get('sentiment_votes_down_percentage', 0),
                'market_cap_rank': data.get('market_cap_rank', 0),
                'coingecko_rank': data.get('coingecko_rank', 0),
                'coingecko_score': data.get('coingecko_score', 0),
                'developer_score': data.get('developer_score', 0),
                'community_score': data.get('community_score', 0),
                'liquidity_score': data.get('liquidity_score', 0),
                'public_interest_score': data.get('public_interest_score', 0)
            }
            
            # Update cache
            self.cache[cache_key] = sentiment
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return sentiment
        except Exception as e:
            print(f"Error fetching sentiment data for {coin_id}: {e}")
            return {}
    
    def get_market_sentiment(self, coin_ids):
        """
        Get market sentiment data for multiple coins
        
        Args:
            coin_ids (list): List of CoinGecko coin IDs
            
        Returns:
            dict: Dictionary with market sentiment data
        """
        # Get Fear & Greed Index
        fear_greed = self.get_fear_greed_index()
        
        # Get sentiment data for each coin
        coin_sentiments = {}
        for coin_id in coin_ids:
            coin_sentiments[coin_id] = self.get_coin_sentiment(coin_id)
        
        # Combine data
        sentiment = {
            'fear_greed_index': fear_greed,
            'coin_sentiments': coin_sentiments
        }
        
        return sentiment
    
    def plot_fear_greed_gauge(self):
        """
        Plot the Fear & Greed Index gauge
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with Fear & Greed gauge
        """
        # Get Fear & Greed Index
        fear_greed = self.get_fear_greed_index()
        
        if fear_greed is None:
            return None
        
        # Create figure
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fear_greed['value'],
            title={'text': "Crypto Fear & Greed Index"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': 'red'},
                    {'range': [25, 50], 'color': 'orange'},
                    {'range': [50, 75], 'color': 'yellow'},
                    {'range': [75, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': fear_greed['value']
                }
            }
        ))
        
        # Add classification annotation
        fig.add_annotation(
            x=0.5,
            y=0.25,
            text=fear_greed['value_classification'],
            showarrow=False,
            font=dict(size=20)
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def plot_fear_greed_historical(self, days=30):
        """
        Plot historical Fear & Greed Index data
        
        Args:
            days (int): Number of days of historical data
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with historical Fear & Greed Index
        """
        # Get historical Fear & Greed Index data
        df = self.get_fear_greed_historical(days)
        
        if df.empty:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['value'],
                mode='lines+markers',
                name='Fear & Greed Index',
                line=dict(color='blue', width=2),
                marker=dict(
                    size=8,
                    color=df['value'],
                    colorscale=[
                        [0, 'red'],
                        [0.25, 'orange'],
                        [0.5, 'yellow'],
                        [0.75, 'green'],
                        [1, 'green']
                    ],
                    cmin=0,
                    cmax=100
                )
            )
        )
        
        # Add reference lines
        fig.add_shape(
            type='line',
            x0=df['timestamp'].min(),
            y0=25,
            x1=df['timestamp'].max(),
            y1=25,
            line=dict(color='red', dash='dash')
        )
        
        fig.add_shape(
            type='line',
            x0=df['timestamp'].min(),
            y0=50,
            x1=df['timestamp'].max(),
            y1=50,
            line=dict(color='yellow', dash='dash')
        )
        
        fig.add_shape(
            type='line',
            x0=df['timestamp'].min(),
            y0=75,
            x1=df['timestamp'].max(),
            y1=75,
            line=dict(color='green', dash='dash')
        )
        
        # Add annotations
        fig.add_annotation(
            x=df['timestamp'].min(),
            y=12.5,
            text="Extreme Fear",
            showarrow=False,
            xanchor='left'
        )
        
        fig.add_annotation(
            x=df['timestamp'].min(),
            y=37.5,
            text="Fear",
            showarrow=False,
            xanchor='left'
        )
        
        fig.add_annotation(
            x=df['timestamp'].min(),
            y=62.5,
            text="Greed",
            showarrow=False,
            xanchor='left'
        )
        
        fig.add_annotation(
            x=df['timestamp'].min(),
            y=87.5,
            text="Extreme Greed",
            showarrow=False,
            xanchor='left'
        )
        
        # Update layout
        fig.update_layout(
            title='Historical Crypto Fear & Greed Index',
            xaxis_title='Date',
            yaxis_title='Index Value',
            template='plotly_dark',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def plot_coin_sentiment_radar(self, coin_id):
        """
        Plot coin sentiment radar chart
        
        Args:
            coin_id (str): CoinGecko coin ID
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with coin sentiment radar chart
        """
        # Get coin sentiment data
        sentiment = self.get_coin_sentiment(coin_id)
        
        if not sentiment:
            return None
        
        # Create radar chart data
        categories = [
            'Developer Score', 'Community Score', 'Liquidity Score',
            'Public Interest', 'Positive Sentiment'
        ]
        
        values = [
            sentiment.get('developer_score', 0),
            sentiment.get('community_score', 0),
            sentiment.get('liquidity_score', 0),
            sentiment.get('public_interest_score', 0),
            sentiment.get('sentiment_votes_up_percentage', 0) / 100 * 10  # Scale to 0-10
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add radar trace
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=coin_id
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'Sentiment Analysis for {coin_id.capitalize()}',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def plot_sentiment_comparison(self, coin_ids):
        """
        Plot sentiment comparison for multiple coins
        
        Args:
            coin_ids (list): List of CoinGecko coin IDs
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with sentiment comparison
        """
        if not coin_ids:
            return None
        
        # Get sentiment data for each coin
        sentiments = {}
        for coin_id in coin_ids:
            sentiments[coin_id] = self.get_coin_sentiment(coin_id)
        
        # Create DataFrame for comparison
        data = []
        for coin_id, sentiment in sentiments.items():
            if sentiment:
                data.append({
                    'coin': coin_id,
                    'developer_score': sentiment.get('developer_score', 0),
                    'community_score': sentiment.get('community_score', 0),
                    'liquidity_score': sentiment.get('liquidity_score', 0),
                    'public_interest_score': sentiment.get('public_interest_score', 0),
                    'positive_sentiment': sentiment.get('sentiment_votes_up_percentage', 0) / 10  # Scale to 0-10
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Melt DataFrame for plotting
        df_melted = pd.melt(
            df,
            id_vars=['coin'],
            value_vars=[
                'developer_score', 'community_score', 'liquidity_score',
                'public_interest_score', 'positive_sentiment'
            ],
            var_name='metric',
            value_name='score'
        )
        
        # Create figure
        fig = px.bar(
            df_melted,
            x='metric',
            y='score',
            color='coin',
            barmode='group',
            title='Sentiment Metrics Comparison',
            labels={
                'metric': 'Metric',
                'score': 'Score (0-10)',
                'coin': 'Cryptocurrency'
            }
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            xaxis_title='',
            yaxis=dict(range=[0, 10])
        )
        
        return fig
