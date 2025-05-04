import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    A class to optimize cryptocurrency portfolios using Modern Portfolio Theory
    """
    
    def __init__(self, returns_data):
        """
        Initialize the portfolio optimizer with historical returns data
        
        Args:
            returns_data (pandas.DataFrame): DataFrame with historical returns for each asset
        """
        self.returns_data = returns_data
        self.prices_data = None
        self.weights = None
        self.expected_annual_return = None
        self.annual_volatility = None
        self.sharpe_ratio = None
    
    def set_prices_data(self, prices_data):
        """
        Set the prices data for allocation calculations
        
        Args:
            prices_data (pandas.DataFrame): DataFrame with price data for each asset
        """
        self.prices_data = prices_data
    
    def calculate_expected_returns(self, method='mean_historical_return'):
        """
        Calculate expected returns for each asset
        
        Args:
            method (str): Method to use for expected returns calculation
                          ('mean_historical_return' or 'capm_return')
                          
        Returns:
            pandas.Series: Expected returns for each asset
        """
        if method == 'mean_historical_return':
            # Simple historical mean return
            return self.returns_data.mean() * 252  # Annualize daily returns
        elif method == 'capm_return':
            # Simplified CAPM implementation
            market_returns = self.returns_data.mean(axis=1)
            asset_returns = {}
            
            for col in self.returns_data.columns:
                # Calculate beta (covariance with market / variance of market)
                beta = self.returns_data[col].cov(market_returns) / market_returns.var()
                # Calculate expected return using CAPM formula: rf + beta * (rm - rf)
                # Assuming risk-free rate of 2% and market risk premium of 5%
                asset_returns[col] = 0.02 + beta * 0.05
            
            return pd.Series(asset_returns)
        else:
            raise ValueError("Method must be 'mean_historical_return' or 'capm_return'")
    
    def calculate_risk_model(self, method='sample_cov'):
        """
        Calculate the risk model (covariance matrix)
        
        Args:
            method (str): Method to use for risk model calculation
                          ('sample_cov' or 'semicovariance' or 'exp_cov')
                          
        Returns:
            pandas.DataFrame: Covariance matrix
        """
        if method == 'sample_cov':
            # Simple sample covariance
            return self.returns_data.cov() * 252  # Annualize daily covariance
        elif method == 'semicovariance':
            # Semicovariance (only consider returns below mean)
            mean_returns = self.returns_data.mean()
            below_mean = self.returns_data.copy()
            
            for col in below_mean.columns:
                below_mean.loc[below_mean[col] > mean_returns[col], col] = mean_returns[col]
            
            return below_mean.cov() * 252  # Annualize daily semicovariance
        elif method == 'exp_cov':
            # Exponentially weighted covariance (more weight to recent observations)
            return self.returns_data.ewm(span=60).cov() * 252  # Annualize daily exp. weighted cov.
        else:
            raise ValueError("Method must be 'sample_cov', 'semicovariance', or 'exp_cov'")
    
    def portfolio_return(self, weights):
        """
        Calculate portfolio return given weights
        
        Args:
            weights (numpy.ndarray): Array of weights
            
        Returns:
            float: Portfolio return
        """
        return np.sum(self.calculate_expected_returns('mean_historical_return') * weights)
    
    def portfolio_volatility(self, weights):
        """
        Calculate portfolio volatility given weights
        
        Args:
            weights (numpy.ndarray): Array of weights
            
        Returns:
            float: Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.calculate_risk_model('sample_cov'), weights)))
    
    def portfolio_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """
        Calculate portfolio Sharpe ratio given weights
        
        Args:
            weights (numpy.ndarray): Array of weights
            risk_free_rate (float): Risk-free rate
            
        Returns:
            float: Portfolio Sharpe ratio
        """
        return (self.portfolio_return(weights) - risk_free_rate) / self.portfolio_volatility(weights)
    
    def negative_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """
        Calculate negative portfolio Sharpe ratio (for minimization)
        
        Args:
            weights (numpy.ndarray): Array of weights
            risk_free_rate (float): Risk-free rate
            
        Returns:
            float: Negative portfolio Sharpe ratio
        """
        return -self.portfolio_sharpe_ratio(weights, risk_free_rate)
    
    def optimize_portfolio(self, risk_free_rate=0.02, weight_bounds=(0, 1), 
                           expected_returns_method='mean_historical_return', 
                           risk_model_method='sample_cov',
                           optimization_criterion='sharpe'):
        """
        Optimize the portfolio based on the specified criterion
        
        Args:
            risk_free_rate (float): Risk-free rate
            weight_bounds (tuple): Minimum and maximum weight for each asset
            expected_returns_method (str): Method for expected returns calculation
            risk_model_method (str): Method for risk model calculation
            optimization_criterion (str): Criterion to optimize for
                                         ('sharpe', 'min_volatility', or 'max_return')
                                         
        Returns:
            dict: Dictionary with optimization results
        """
        n_assets = len(self.returns_data.columns)
        
        # Initial guess (equal weights)
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Bounds for weights
        bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))
        
        # Constraint: sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Optimize based on the specified criterion
        if optimization_criterion == 'sharpe':
            # Maximize Sharpe ratio (minimize negative Sharpe ratio)
            result = minimize(
                self.negative_sharpe_ratio,
                init_weights,
                args=(risk_free_rate,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif optimization_criterion == 'min_volatility':
            # Minimize volatility
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif optimization_criterion == 'max_return':
            # Maximize return (minimize negative return)
            result = minimize(
                lambda x: -self.portfolio_return(x),
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            raise ValueError("Criterion must be 'sharpe', 'min_volatility', or 'max_return'")
        
        # Get optimal weights
        optimal_weights = result['x']
        
        # Calculate portfolio performance
        self.expected_annual_return = self.portfolio_return(optimal_weights)
        self.annual_volatility = self.portfolio_volatility(optimal_weights)
        self.sharpe_ratio = self.portfolio_sharpe_ratio(optimal_weights, risk_free_rate)
        
        # Clean weights (remove very small weights)
        cleaned_weights = {}
        for i, asset in enumerate(self.returns_data.columns):
            if optimal_weights[i] > 0.001:  # Threshold for small weights
                cleaned_weights[asset] = optimal_weights[i]
        
        # Normalize cleaned weights to sum to 1
        total_weight = sum(cleaned_weights.values())
        self.weights = {k: v / total_weight for k, v in cleaned_weights.items()}
        
        # Return results
        return {
            'weights': self.weights,
            'expected_annual_return': self.expected_annual_return,
            'annual_volatility': self.annual_volatility,
            'sharpe_ratio': self.sharpe_ratio
        }
    
    def get_discrete_allocation(self, total_portfolio_value=10000):
        """
        Get discrete allocation of assets based on the optimized weights
        
        Args:
            total_portfolio_value (float): Total value of the portfolio
            
        Returns:
            dict: Dictionary with allocation results
        """
        if self.weights is None:
            raise ValueError("Portfolio must be optimized first")
        
        if self.prices_data is None:
            raise ValueError("Prices data must be set first")
        
        # Get latest prices
        latest_prices = {}
        for asset in self.weights.keys():
            if asset in self.prices_data.columns:
                latest_prices[asset] = self.prices_data[asset].iloc[-1]
        
        # Calculate allocation
        allocation = {}
        leftover = total_portfolio_value
        
        for asset, weight in self.weights.items():
            if asset in latest_prices:
                price = latest_prices[asset]
                if price > 0:
                    # Calculate number of units
                    asset_value = total_portfolio_value * weight
                    units = int(asset_value / price)
                    allocation[asset] = units
                    leftover -= units * price
        
        return {
            'allocation': allocation,
            'leftover': leftover
        }
    
    def calculate_efficient_frontier(self, risk_free_rate=0.02, points=100,
                                    expected_returns_method='mean_historical_return',
                                    risk_model_method='sample_cov'):
        """
        Calculate the efficient frontier
        
        Args:
            risk_free_rate (float): Risk-free rate
            points (int): Number of points to calculate
            expected_returns_method (str): Method for expected returns calculation
            risk_model_method (str): Method for risk model calculation
            
        Returns:
            tuple: Tuple with returns and volatilities for the efficient frontier
        """
        n_assets = len(self.returns_data.columns)
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.calculate_expected_returns(method=expected_returns_method)
        cov_matrix = self.calculate_risk_model(method=risk_model_method)
        
        # Get minimum and maximum returns for individual assets
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        
        # Create a range of target returns
        target_returns = np.linspace(min_return, max_return, points)
        
        # Calculate efficient frontier
        efficient_returns = []
        efficient_volatilities = []
        
        for target_return in target_returns:
            # Initial guess (equal weights)
            init_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Bounds for weights
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Constraints: sum of weights = 1, portfolio return = target return
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target_return}
            )
            
            # Minimize volatility for the target return
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result['success']:
                # Calculate portfolio return and volatility
                weights = result['x']
                portfolio_return = self.portfolio_return(weights)
                portfolio_volatility = self.portfolio_volatility(weights)
                
                efficient_returns.append(portfolio_return)
                efficient_volatilities.append(portfolio_volatility)
        
        return efficient_returns, efficient_volatilities
    
    def plot_efficient_frontier(self, risk_free_rate=0.02, points=100,
                               expected_returns_method='mean_historical_return',
                               risk_model_method='sample_cov'):
        """
        Plot the efficient frontier
        
        Args:
            risk_free_rate (float): Risk-free rate
            points (int): Number of points to calculate
            expected_returns_method (str): Method for expected returns calculation
            risk_model_method (str): Method for risk model calculation
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with the efficient frontier
        """
        # Calculate efficient frontier
        returns, volatilities = self.calculate_efficient_frontier(
            risk_free_rate=risk_free_rate,
            points=points,
            expected_returns_method=expected_returns_method,
            risk_model_method=risk_model_method
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add current portfolio if optimized
        if self.annual_volatility is not None and self.expected_annual_return is not None:
            fig.add_trace(
                go.Scatter(
                    x=[self.annual_volatility],
                    y=[self.expected_annual_return],
                    mode='markers',
                    name='Optimized Portfolio',
                    marker=dict(color='red', size=10, symbol='star')
                )
            )
        
        # Add individual assets
        if self.returns_data is not None:
            # Calculate individual asset returns and volatilities
            expected_returns = self.calculate_expected_returns(method=expected_returns_method)
            cov_matrix = self.calculate_risk_model(method=risk_model_method)
            asset_volatilities = np.sqrt(np.diag(cov_matrix))
            
            fig.add_trace(
                go.Scatter(
                    x=asset_volatilities,
                    y=expected_returns,
                    mode='markers+text',
                    name='Individual Assets',
                    marker=dict(color='green', size=8),
                    text=expected_returns.index,
                    textposition="top center"
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Expected Return',
            template='plotly_dark',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of assets
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with the correlation matrix
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set first")
        
        # Calculate correlation matrix
        corr = self.returns_data.corr()
        
        # Create figure
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Asset Correlation Matrix'
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            coloraxis_colorbar=dict(
                title='Correlation',
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="top", y=1,
                ticks="outside"
            )
        )
        
        return fig
    
    def plot_weights_pie(self):
        """
        Plot the optimized portfolio weights as a pie chart
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with the weights pie chart
        """
        if self.weights is None:
            raise ValueError("Portfolio must be optimized first")
        
        # Filter out assets with zero weight
        weights = {k: v for k, v in self.weights.items() if v > 0.001}
        
        # Create figure
        fig = px.pie(
            values=list(weights.values()),
            names=list(weights.keys()),
            title='Optimized Portfolio Weights',
            template='plotly_dark',
            hole=0.3
        )
        
        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
