import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

class CryptoClustering:
    """
    A class for cryptocurrency clustering using K-means
    """
    
    def __init__(self):
        """
        Initialize the clustering model
        """
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.data = None
        self.scaled_data = None
        self.features = None
        self.labels = None
        self.pca_components = None
    
    def prepare_data(self, returns_data):
        """
        Prepare data for clustering
        
        Args:
            returns_data (pandas.DataFrame): DataFrame with returns data
            
        Returns:
            pandas.DataFrame: DataFrame with features for clustering
        """
        self.data = returns_data.copy()
        
        # Calculate features for clustering
        features = pd.DataFrame(index=self.data.columns)
        
        # Mean daily return
        features['mean_return'] = self.data.mean()
        
        # Volatility (standard deviation of returns)
        features['volatility'] = self.data.std()
        
        # Skewness
        features['skewness'] = self.data.skew()
        
        # Kurtosis
        features['kurtosis'] = self.data.kurtosis()
        
        # Maximum drawdown
        cumulative_returns = (1 + self.data).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        features['max_drawdown'] = drawdown.min()
        
        # Correlation with Bitcoin (if available)
        if 'bitcoin' in self.data.columns:
            features['btc_correlation'] = self.data.corrwith(self.data['bitcoin'])
        
        # Drop any rows with NaN values
        features = features.dropna()
        
        self.features = features
        
        return features
    
    def scale_data(self):
        """
        Scale data for clustering
        
        Returns:
            numpy.ndarray: Scaled data
        """
        if self.features is None:
            raise ValueError("Data must be prepared first")
        
        self.scaled_data = self.scaler.fit_transform(self.features)
        
        return self.scaled_data
    
    def apply_pca(self, n_components=2):
        """
        Apply PCA for dimensionality reduction
        
        Args:
            n_components (int): Number of components
            
        Returns:
            numpy.ndarray: PCA components
        """
        if self.scaled_data is None:
            raise ValueError("Data must be scaled first")
        
        self.pca = PCA(n_components=n_components)
        self.pca_components = self.pca.fit_transform(self.scaled_data)
        
        return self.pca_components
    
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using elbow method
        
        Args:
            max_clusters (int): Maximum number of clusters to try
            
        Returns:
            tuple: Tuple with number of clusters and inertia values
        """
        if self.scaled_data is None:
            raise ValueError("Data must be scaled first")
        
        inertia = []
        k_values = range(1, max_clusters + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertia.append(kmeans.inertia_)
        
        return k_values, inertia
    
    def cluster_data(self, n_clusters=3):
        """
        Cluster data using K-means
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        if self.scaled_data is None:
            raise ValueError("Data must be scaled first")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.scaled_data)
        
        return self.labels
    
    def get_cluster_results(self):
        """
        Get clustering results
        
        Returns:
            pandas.DataFrame: DataFrame with clustering results
        """
        if self.labels is None or self.features is None:
            raise ValueError("Data must be clustered first")
        
        # Create DataFrame with results
        results = self.features.copy()
        results['cluster'] = self.labels
        
        # Add PCA components if available
        if self.pca_components is not None:
            for i in range(self.pca_components.shape[1]):
                results[f'pca_{i+1}'] = self.pca_components[:, i]
        
        return results
    
    def plot_elbow_method(self, k_values, inertia):
        """
        Plot elbow method for finding optimal number of clusters
        
        Args:
            k_values (list): List of k values
            inertia (list): List of inertia values
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with elbow method plot
        """
        # Create figure
        fig = go.Figure()
        
        # Add line
        fig.add_trace(
            go.Scatter(
                x=list(k_values),
                y=inertia,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Elbow Method for Optimal K',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            template='plotly_dark'
        )
        
        return fig
    
    def plot_clusters_2d(self):
        """
        Plot clusters in 2D
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with 2D cluster plot
        """
        if self.labels is None or self.pca_components is None or self.pca_components.shape[1] < 2:
            raise ValueError("Data must be clustered and PCA applied with at least 2 components")
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'pca_1': self.pca_components[:, 0],
            'pca_2': self.pca_components[:, 1],
            'cluster': self.labels,
            'crypto': self.features.index
        })
        
        # Create figure
        fig = px.scatter(
            results,
            x='pca_1',
            y='pca_2',
            color='cluster',
            hover_name='crypto',
            title='Cryptocurrency Clusters (2D)',
            template='plotly_dark',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            legend_title='Cluster'
        )
        
        return fig
    
    def plot_clusters_3d(self):
        """
        Plot clusters in 3D
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with 3D cluster plot
        """
        if self.labels is None or self.pca_components is None or self.pca_components.shape[1] < 3:
            raise ValueError("Data must be clustered and PCA applied with at least 3 components")
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'pca_1': self.pca_components[:, 0],
            'pca_2': self.pca_components[:, 1],
            'pca_3': self.pca_components[:, 2],
            'cluster': self.labels,
            'crypto': self.features.index
        })
        
        # Create figure
        fig = px.scatter_3d(
            results,
            x='pca_1',
            y='pca_2',
            z='pca_3',
            color='cluster',
            hover_name='crypto',
            title='Cryptocurrency Clusters (3D)',
            template='plotly_dark',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3'
            ),
            legend_title='Cluster'
        )
        
        return fig
    
    def plot_cluster_profiles(self):
        """
        Plot cluster profiles
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with cluster profiles
        """
        if self.labels is None or self.features is None:
            raise ValueError("Data must be clustered first")
        
        # Create DataFrame with results
        results = self.features.copy()
        results['cluster'] = self.labels
        
        # Calculate cluster means
        cluster_means = results.groupby('cluster').mean()
        
        # Normalize cluster means for radar chart
        normalized_means = cluster_means.copy()
        for col in normalized_means.columns:
            normalized_means[col] = (normalized_means[col] - normalized_means[col].min()) / (normalized_means[col].max() - normalized_means[col].min())
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each cluster
        for cluster in normalized_means.index:
            cluster_data = normalized_means.loc[cluster]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=cluster_data.values,
                    theta=cluster_data.index,
                    fill='toself',
                    name=f'Cluster {cluster}'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Cluster Profiles',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            template='plotly_dark',
            showlegend=True
        )
        
        return fig

def run_clustering(returns_data, n_clusters=3):
    """
    Run clustering on cryptocurrency returns data
    
    Args:
        returns_data (pandas.DataFrame): DataFrame with returns data
        n_clusters (int): Number of clusters
        
    Returns:
        tuple: Tuple with clustering results, 2D plot, 3D plot, and cluster profiles
    """
    # Create clustering model
    clustering = CryptoClustering()
    
    # Prepare data
    features = clustering.prepare_data(returns_data)
    
    # Ensure n_clusters is not greater than the number of samples
    n_samples = len(features)
    if n_clusters > n_samples:
        n_clusters = n_samples
    
    # Scale data
    clustering.scale_data()
    
    # Find optimal number of clusters (limit max_clusters to number of samples)
    max_clusters = min(10, n_samples)
    k_values, inertia = clustering.find_optimal_clusters(max_clusters=max_clusters)
    elbow_fig = clustering.plot_elbow_method(k_values, inertia)
    
    # Apply PCA for visualization (limit components to number of samples)
    n_components = min(3, n_samples)
    clustering.apply_pca(n_components=n_components)
    
    # Cluster data
    clustering.cluster_data(n_clusters=n_clusters)
    
    # Get results
    results = clustering.get_cluster_results()
    
    # Create plots
    plot_2d = clustering.plot_clusters_2d() if n_components >= 2 else None
    plot_3d = clustering.plot_clusters_3d() if n_components >= 3 else None
    profiles = clustering.plot_cluster_profiles()
    
    return results, elbow_fig, plot_2d, plot_3d, profiles
