import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

class RiskAssessor:
    """
    A class to assess risk for cryptocurrency portfolios
    """
    
    def __init__(self, returns_data, weights=None):
        """
        Initialize the risk assessor with historical returns data
        
        Args:
            returns_data (pandas.DataFrame): DataFrame with historical returns for each asset
            weights (dict, optional): Dictionary with portfolio weights
        """
        self.returns_data = returns_data
        self.weights = weights
        self.portfolio_returns = None
        
        # Calculate portfolio returns if weights are provided
        if weights is not None:
            self.calculate_portfolio_returns()
    
    def set_weights(self, weights):
        """
        Set portfolio weights
        
        Args:
            weights (dict): Dictionary with portfolio weights
        """
        self.weights = weights
        self.calculate_portfolio_returns()
    
    def calculate_portfolio_returns(self):
        """
        Calculate historical portfolio returns based on weights
        
        Returns:
            pandas.Series: Historical portfolio returns
        """
        if self.weights is None:
            raise ValueError("Portfolio weights must be set first")
        
        # Convert weights dictionary to Series
        weights_series = pd.Series(self.weights)
        
        # Filter returns data to include only assets in weights
        filtered_returns = self.returns_data[weights_series.index]
        
        # Calculate portfolio returns
        self.portfolio_returns = filtered_returns.dot(weights_series)
        
        return self.portfolio_returns
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate the Sharpe ratio
        
        Args:
            risk_free_rate (float): Risk-free rate
            periods_per_year (int): Number of periods per year
            
        Returns:
            float: Sharpe ratio
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate annualized return and volatility
        mean_return = self.portfolio_returns.mean() * periods_per_year
        volatility = self.portfolio_returns.std() * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate the Sortino ratio
        
        Args:
            risk_free_rate (float): Risk-free rate
            periods_per_year (int): Number of periods per year
            
        Returns:
            float: Sortino ratio
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate annualized return
        mean_return = self.portfolio_returns.mean() * periods_per_year
        
        # Calculate downside deviation
        negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
        
        # Handle case where there are no negative returns
        if downside_deviation == 0:
            return float('inf')
        
        # Calculate Sortino ratio
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
        
        return sortino_ratio
    
    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown
        
        Returns:
            float: Maximum drawdown
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate cumulative returns
        cum_returns = (1 + self.portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def calculate_var(self, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) using historical method
        
        Args:
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            float: Value at Risk
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate VaR
        var = -np.percentile(self.portfolio_returns, 100 * (1 - confidence_level))
        
        return var
    
    def calculate_cvar(self, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) using historical method
        
        Args:
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            float: Conditional Value at Risk
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate VaR
        var = self.calculate_var(confidence_level)
        
        # Calculate CVaR
        cvar = -self.portfolio_returns[self.portfolio_returns <= -var].mean()
        
        return cvar
    
    def run_monte_carlo_simulation(self, num_simulations=1000, time_horizon=252, initial_investment=10000):
        """
        Run Monte Carlo simulation for portfolio returns
        
        Args:
            num_simulations (int): Number of simulations to run
            time_horizon (int): Time horizon in days
            initial_investment (float): Initial investment amount
            
        Returns:
            pandas.DataFrame: DataFrame with simulation results
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate mean and standard deviation of daily returns
        mean_return = self.portfolio_returns.mean()
        std_return = self.portfolio_returns.std()
        
        # Initialize simulation results
        simulation_results = pd.DataFrame()
        
        # Run simulations
        for i in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, time_horizon)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + random_returns).cumprod()
            
            # Calculate portfolio value
            portfolio_value = initial_investment * cumulative_returns
            
            # Add to simulation results
            simulation_results[f'Simulation_{i}'] = portfolio_value
        
        return simulation_results
    
    def plot_monte_carlo_simulation(self, num_simulations=1000, time_horizon=252, initial_investment=10000):
        """
        Plot Monte Carlo simulation results
        
        Args:
            num_simulations (int): Number of simulations to run
            time_horizon (int): Time horizon in days
            initial_investment (float): Initial investment amount
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with simulation results
        """
        # Run Monte Carlo simulation
        simulation_results = self.run_monte_carlo_simulation(
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            initial_investment=initial_investment
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add simulation traces (sample 100 simulations for better visualization)
        sample_size = min(100, num_simulations)
        sample_columns = np.random.choice(simulation_results.columns, sample_size, replace=False)
        
        for col in sample_columns:
            fig.add_trace(
                go.Scatter(
                    y=simulation_results[col],
                    mode='lines',
                    name=col,
                    line=dict(width=0.5, color='rgba(70, 130, 180, 0.2)')
                )
            )
        
        # Add mean, 5th and 95th percentiles
        mean = simulation_results.mean(axis=1)
        percentile_5 = simulation_results.quantile(0.05, axis=1)
        percentile_95 = simulation_results.quantile(0.95, axis=1)
        
        fig.add_trace(
            go.Scatter(
                y=mean,
                mode='lines',
                name='Mean',
                line=dict(width=2, color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                y=percentile_5,
                mode='lines',
                name='5th Percentile',
                line=dict(width=2, color='red')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                y=percentile_95,
                mode='lines',
                name='95th Percentile',
                line=dict(width=2, color='green')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'Monte Carlo Simulation ({num_simulations} runs, {time_horizon} days)',
            xaxis_title='Days',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            showlegend=False
        )
        
        return fig
    
    def plot_drawdown(self):
        """
        Plot portfolio drawdown
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with drawdown
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate cumulative returns
        cum_returns = (1 + self.portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        # Create figure
        fig = go.Figure()
        
        # Add drawdown trace
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            )
        )
        
        # Add zero line
        fig.add_shape(
            type='line',
            x0=drawdown.index[0],
            y0=0,
            x1=drawdown.index[-1],
            y1=0,
            line=dict(color='gray', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            template='plotly_dark'
        )
        
        return fig
    
    def plot_returns_distribution(self):
        """
        Plot the distribution of portfolio returns
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with returns distribution
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=self.portfolio_returns,
                nbinsx=50,
                name='Returns',
                marker=dict(color='blue')
            )
        )
        
        # Add normal distribution
        mean = self.portfolio_returns.mean()
        std = self.portfolio_returns.std()
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        y = stats.norm.pdf(x, mean, std)
        
        # Scale y to match histogram
        hist, bin_edges = np.histogram(self.portfolio_returns, bins=50)
        max_hist = np.max(hist)
        max_pdf = np.max(y)
        y = y * (max_hist / max_pdf)
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            )
        )
        
        # Add VaR line
        var_95 = self.calculate_var(confidence_level=0.95)
        fig.add_shape(
            type='line',
            x0=var_95,
            y0=0,
            x1=var_95,
            y1=max_hist,
            line=dict(color='green', dash='dash', width=2)
        )
        
        # Add annotation for VaR
        fig.add_annotation(
            x=var_95,
            y=max_hist * 0.9,
            text=f'95% VaR: {var_95:.2%}',
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=0
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Returns Distribution',
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            template='plotly_dark'
        )
        
        return fig
    
    def get_risk_metrics(self, risk_free_rate=0.02, periods_per_year=252):
        """
        Get all risk metrics
        
        Args:
            risk_free_rate (float): Risk-free rate
            periods_per_year (int): Number of periods per year
            
        Returns:
            dict: Dictionary with risk metrics
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns must be calculated first")
        
        # Calculate metrics
        sharpe_ratio = self.calculate_sharpe_ratio(risk_free_rate, periods_per_year)
        sortino_ratio = self.calculate_sortino_ratio(risk_free_rate, periods_per_year)
        max_drawdown = self.calculate_max_drawdown()
        var_95 = self.calculate_var(confidence_level=0.95)
        cvar_95 = self.calculate_cvar(confidence_level=0.95)
        
        # Calculate annualized return and volatility
        mean_return = self.portfolio_returns.mean() * periods_per_year
        volatility = self.portfolio_returns.std() * np.sqrt(periods_per_year)
        
        return {
            'annualized_return': mean_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
