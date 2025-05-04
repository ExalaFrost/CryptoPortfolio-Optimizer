import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st

class AnomalyDetector:
    """
    A class for cryptocurrency price anomaly detection using Isolation Forest
    """
    
    def __init__(self):
        """
        Initialize the anomaly detector
        """
        self.scaler = StandardScaler()
        self.model = None
        self.data = None
        self.scaled_data = None
        self.anomalies = None
    
    def prepare_data(self, price_data, window=10):
        """
        Prepare data for anomaly detection
        
        Args:
            price_data (pandas.DataFrame): DataFrame with price data
            window (int): Window size for feature engineering
            
        Returns:
            pandas.DataFrame: DataFrame with features for anomaly detection
        """
        self.data = price_data.copy()
        
        # Create features for anomaly detection
        features = pd.DataFrame(index=self.data.index)
        
        # For each cryptocurrency
        for coin in self.data.columns:
            # Price
            features[f'{coin}_price'] = self.data[coin]
            
            # Returns
            features[f'{coin}_return'] = self.data[coin].pct_change()
            
            # Volatility (rolling standard deviation)
            features[f'{coin}_volatility'] = self.data[coin].pct_change().rolling(window=window).std()
            
            # Price momentum (ratio of current price to moving average)
            features[f'{coin}_momentum'] = self.data[coin] / self.data[coin].rolling(window=window).mean()
            
            # Price acceleration (change in returns)
            features[f'{coin}_acceleration'] = self.data[coin].pct_change().diff()
        
        # Drop rows with NaN values
        features = features.dropna()
        
        self.features = features
        
        return features
    
    def scale_data(self):
        """
        Scale data for anomaly detection
        
        Returns:
            numpy.ndarray: Scaled data
        """
        if self.features is None:
            raise ValueError("Data must be prepared first")
        
        self.scaled_data = self.scaler.fit_transform(self.features)
        
        return self.scaled_data
    
    def detect_anomalies(self, contamination=0.05):
        """
        Detect anomalies using Isolation Forest
        
        Args:
            contamination (float): Expected proportion of anomalies
            
        Returns:
            numpy.ndarray: Anomaly labels (-1 for anomalies, 1 for normal)
        """
        if self.scaled_data is None:
            raise ValueError("Data must be scaled first")
        
        # Create and fit model
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.anomalies = self.model.fit_predict(self.scaled_data)
        
        return self.anomalies
    
    def get_anomaly_results(self):
        """
        Get anomaly detection results
        
        Returns:
            pandas.DataFrame: DataFrame with anomaly detection results
        """
        if self.anomalies is None or self.features is None:
            raise ValueError("Anomalies must be detected first")
        
        # Create DataFrame with results
        results = pd.DataFrame(index=self.features.index)
        
        # Add anomaly labels
        results['anomaly'] = self.anomalies
        
        # Add anomaly score
        results['anomaly_score'] = self.model.decision_function(self.scaled_data)
        
        # Add original data
        for col in self.data.columns:
            results[col] = self.data.loc[results.index, col]
        
        return results
    
    def plot_anomalies(self, coin_id):
        """
        Plot anomalies for a specific cryptocurrency
        
        Args:
            coin_id (str): Cryptocurrency ID
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with anomalies
        """
        if self.anomalies is None or self.features is None:
            raise ValueError("Anomalies must be detected first")
        
        # Get results
        results = self.get_anomaly_results()
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results[coin_id],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=1)
            )
        )
        
        # Add anomalies
        anomalies = results[results['anomaly'] == -1]
        
        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies[coin_id],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='circle',
                    line=dict(color='black', width=1)
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'Price Anomalies for {coin_id.capitalize()}',
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
    
    def plot_anomaly_scores(self):
        """
        Plot anomaly scores
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with anomaly scores
        """
        if self.anomalies is None or self.features is None:
            raise ValueError("Anomalies must be detected first")
        
        # Get results
        results = self.get_anomaly_results()
        
        # Create figure
        fig = go.Figure()
        
        # Add anomaly score line
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['anomaly_score'],
                mode='lines',
                name='Anomaly Score',
                line=dict(color='blue', width=1)
            )
        )
        
        # Add threshold line (using the decision function scores)
        # For Isolation Forest, anomalies have scores < 0
        threshold = 0
        
        fig.add_trace(
            go.Scatter(
                x=[results.index.min(), results.index.max()],
                y=[threshold, threshold],
                mode='lines',
                name='Threshold (0)',
                line=dict(color='red', width=1, dash='dash')
            )
        )
        
        # Add anomalies
        anomalies = results[results['anomaly'] == -1]
        
        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies['anomaly_score'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='circle',
                    line=dict(color='black', width=1)
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Anomaly Scores',
            xaxis_title='Date',
            yaxis_title='Anomaly Score',
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
    
    def plot_feature_importance(self):
        """
        Plot feature importance
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with feature importance
        """
        if self.model is None or self.features is None:
            raise ValueError("Model must be trained first")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'feature': self.features.columns,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Create figure
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance',
            template='plotly_dark'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Feature',
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def get_anomaly_stats(self):
        """
        Get anomaly statistics
        
        Returns:
            dict: Dictionary with anomaly statistics
        """
        if self.anomalies is None:
            raise ValueError("Anomalies must be detected first")
        
        # Get results
        results = self.get_anomaly_results()
        
        # Calculate statistics
        total_points = len(results)
        anomaly_points = len(results[results['anomaly'] == -1])
        anomaly_percentage = (anomaly_points / total_points) * 100
        
        # Get dates of anomalies
        anomaly_dates = results[results['anomaly'] == -1].index.tolist()
        
        # Calculate average anomaly score
        avg_anomaly_score = results[results['anomaly'] == -1]['anomaly_score'].mean()
        
        return {
            'total_points': total_points,
            'anomaly_points': anomaly_points,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_dates': anomaly_dates,
            'avg_anomaly_score': avg_anomaly_score
        }

def run_anomaly_detection(price_data, contamination=0.05):
    """
    Run anomaly detection on cryptocurrency price data
    
    Args:
        price_data (pandas.DataFrame): DataFrame with price data
        contamination (float): Expected proportion of anomalies
        
    Returns:
        tuple: Tuple with anomaly results, plots, and statistics
    """
    # Create anomaly detector
    detector = AnomalyDetector()
    
    # Prepare data
    detector.prepare_data(price_data)
    
    # Scale data
    detector.scale_data()
    
    # Detect anomalies
    detector.detect_anomalies(contamination=contamination)
    
    # Get results
    results = detector.get_anomaly_results()
    
    # Create plots for each cryptocurrency
    anomaly_plots = {}
    for coin in price_data.columns:
        anomaly_plots[coin] = detector.plot_anomalies(coin)
    
    # Create anomaly score plot
    score_plot = detector.plot_anomaly_scores()
    
    # Create feature importance plot
    importance_plot = detector.plot_feature_importance()
    
    # Get statistics
    stats = detector.get_anomaly_stats()
    
    return results, anomaly_plots, score_plot, importance_plot, stats
