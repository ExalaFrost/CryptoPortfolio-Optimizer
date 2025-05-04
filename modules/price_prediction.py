import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from datetime import datetime, timedelta
import streamlit as st

class PricePredictionModel:
    """
    A class for cryptocurrency price prediction using Facebook Prophet
    """
    
    def __init__(self):
        """
        Initialize the price prediction model
        """
        self.model = None
        self.forecast = None
        self.historical_data = None
        self.coin_id = None
    
    def prepare_data(self, price_data, coin_id):
        """
        Prepare data for Prophet model
        
        Args:
            price_data (pandas.DataFrame): DataFrame with price data
            coin_id (str): Cryptocurrency ID
            
        Returns:
            pandas.DataFrame: DataFrame prepared for Prophet
        """
        self.coin_id = coin_id
        self.historical_data = price_data.copy()
        
        # Prophet requires columns named 'ds' and 'y'
        prophet_data = pd.DataFrame({
            'ds': price_data.index,
            'y': price_data[coin_id]
        }).reset_index(drop=True)
        
        return prophet_data
    
    def train_model(self, prophet_data, changepoint_prior_scale=0.05, seasonality_mode='multiplicative'):
        """
        Train Prophet model
        
        Args:
            prophet_data (pandas.DataFrame): DataFrame prepared for Prophet
            changepoint_prior_scale (float): Flexibility of the trend
            seasonality_mode (str): 'additive' or 'multiplicative'
            
        Returns:
            Prophet: Trained Prophet model
        """
        # Create and train model
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        self.model.fit(prophet_data)
        
        return self.model
    
    def make_future_dataframe(self, periods=30):
        """
        Create future dataframe for prediction
        
        Args:
            periods (int): Number of periods to forecast
            
        Returns:
            pandas.DataFrame: Future dataframe
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        return self.model.make_future_dataframe(periods=periods)
    
    def predict(self, future_df):
        """
        Make predictions
        
        Args:
            future_df (pandas.DataFrame): Future dataframe
            
        Returns:
            pandas.DataFrame: Forecast
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        self.forecast = self.model.predict(future_df)
        
        return self.forecast
    
    def plot_forecast(self):
        """
        Plot forecast
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with forecast
        """
        if self.forecast is None or self.historical_data is None:
            raise ValueError("Prediction must be made first")
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=self.historical_data.index,
                y=self.historical_data[self.coin_id],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            )
        )
        
        # Add forecast
        forecast_dates = pd.to_datetime(self.forecast['ds'])
        historical_dates = self.historical_data.index
        
        # Filter forecast to only include future dates
        future_mask = forecast_dates > historical_dates.max()
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates[future_mask],
                y=self.forecast['yhat'][future_mask],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            )
        )
        
        # Add prediction intervals
        fig.add_trace(
            go.Scatter(
                x=forecast_dates[future_mask],
                y=self.forecast['yhat_upper'][future_mask],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates[future_mask],
                y=self.forecast['yhat_lower'][future_mask],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty',
                name='95% Confidence Interval'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'Price Forecast for {self.coin_id.capitalize()}',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_components(self):
        """
        Plot forecast components
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with components
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Prediction must be made first")
        
        # Create subplots
        fig = go.Figure()
        
        # Trend component
        fig.add_trace(
            go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='blue', width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Trend Component',
            xaxis_title='Date',
            yaxis_title='Trend',
            template='plotly_dark'
        )
        
        return fig
    
    def get_forecast_metrics(self, days=30):
        """
        Get forecast metrics
        
        Args:
            days (int): Number of days to forecast
            
        Returns:
            dict: Dictionary with forecast metrics
        """
        if self.forecast is None:
            raise ValueError("Prediction must be made first")
        
        # Get last historical date
        last_date = self.historical_data.index.max()
        
        # Get last historical price
        last_price = self.historical_data[self.coin_id].iloc[-1]
        
        # Get forecast for specified days
        future_forecast = self.forecast[self.forecast['ds'] > last_date].iloc[:days]
        
        # Calculate metrics
        forecast_end_price = future_forecast['yhat'].iloc[-1]
        price_change = forecast_end_price - last_price
        price_change_pct = (price_change / last_price) * 100
        
        # Calculate average daily return
        daily_returns = future_forecast['yhat'].pct_change().dropna()
        avg_daily_return = daily_returns.mean() * 100
        
        # Calculate volatility
        volatility = daily_returns.std() * 100
        
        return {
            'forecast_end_price': forecast_end_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility
        }

def run_price_prediction(price_data, coin_id, forecast_days=30):
    """
    Run price prediction for a cryptocurrency
    
    Args:
        price_data (pandas.DataFrame): DataFrame with price data
        coin_id (str): Cryptocurrency ID
        forecast_days (int): Number of days to forecast
        
    Returns:
        tuple: Tuple with forecast figure, components figure, and metrics
    """
    # Create model
    model = PricePredictionModel()
    
    # Prepare data
    prophet_data = model.prepare_data(price_data, coin_id)
    
    # Train model
    model.train_model(prophet_data)
    
    # Make future dataframe
    future_df = model.make_future_dataframe(periods=forecast_days)
    
    # Make prediction
    forecast = model.predict(future_df)
    
    # Get forecast figure
    forecast_fig = model.plot_forecast()
    
    # Get components figure
    components_fig = model.plot_components()
    
    # Get forecast metrics
    metrics = model.get_forecast_metrics(days=forecast_days)
    
    return forecast_fig, components_fig, metrics
