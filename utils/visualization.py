import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_price_history(price_data, coin_id, days=30):
    """
    Plot price history for a cryptocurrency
    
    Args:
        price_data (pandas.DataFrame): DataFrame with price data
        coin_id (str): Coin ID or symbol
        days (int): Number of days to display
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with price history
    """
    if price_data.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    # Add volume trace
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['volume'],
            name='Volume',
            marker=dict(color='rgba(100, 100, 255, 0.3)')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'{coin_id.upper()} Price History (Last {days} Days)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    fig.update_xaxes(title_text="Date")
    
    return fig

def plot_returns_heatmap(returns_data):
    """
    Plot correlation heatmap of asset returns
    
    Args:
        returns_data (pandas.DataFrame): DataFrame with returns data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with correlation heatmap
    """
    if returns_data.empty:
        return None
    
    # Calculate correlation matrix
    corr = returns_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Asset Correlation Matrix',
        labels=dict(color="Correlation")
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=600,
        width=800
    )
    
    return fig

def plot_portfolio_composition(weights):
    """
    Plot portfolio composition as a pie chart
    
    Args:
        weights (dict): Dictionary with asset weights
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with portfolio composition
    """
    if not weights:
        return None
    
    # Filter out assets with zero weight
    weights = {k: v for k, v in weights.items() if v > 0.001}
    
    # Create pie chart
    fig = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        title='Portfolio Composition',
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

def plot_risk_return_scatter(returns, volatilities, labels, highlight_index=None):
    """
    Plot risk-return scatter plot
    
    Args:
        returns (list): List of returns
        volatilities (list): List of volatilities
        labels (list): List of labels
        highlight_index (int, optional): Index to highlight
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with risk-return scatter plot
    """
    if not returns or not volatilities or not labels:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter trace
    marker_colors = ['blue'] * len(returns)
    marker_sizes = [10] * len(returns)
    
    # Highlight selected point if specified
    if highlight_index is not None and 0 <= highlight_index < len(returns):
        marker_colors[highlight_index] = 'red'
        marker_sizes[highlight_index] = 15
    
    fig.add_trace(
        go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(
                color=marker_colors,
                size=marker_sizes
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Risk-Return Profile',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        template='plotly_dark'
    )
    
    return fig

def plot_efficient_frontier_with_assets(ef_returns, ef_volatilities, asset_returns, asset_volatilities, asset_labels, 
                                       optimal_return=None, optimal_volatility=None):
    """
    Plot efficient frontier with individual assets
    
    Args:
        ef_returns (list): List of efficient frontier returns
        ef_volatilities (list): List of efficient frontier volatilities
        asset_returns (list): List of individual asset returns
        asset_volatilities (list): List of individual asset volatilities
        asset_labels (list): List of individual asset labels
        optimal_return (float, optional): Optimal portfolio return
        optimal_volatility (float, optional): Optimal portfolio volatility
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with efficient frontier and assets
    """
    if not ef_returns or not ef_volatilities:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add efficient frontier
    fig.add_trace(
        go.Scatter(
            x=ef_volatilities,
            y=ef_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add individual assets
    if asset_returns and asset_volatilities and asset_labels:
        fig.add_trace(
            go.Scatter(
                x=asset_volatilities,
                y=asset_returns,
                mode='markers+text',
                name='Individual Assets',
                marker=dict(color='green', size=8),
                text=asset_labels,
                textposition="top center"
            )
        )
    
    # Add optimal portfolio
    if optimal_return is not None and optimal_volatility is not None:
        fig.add_trace(
            go.Scatter(
                x=[optimal_volatility],
                y=[optimal_return],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='red', size=12, symbol='star')
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

def plot_monte_carlo_simulation(simulation_results, percentiles=[5, 50, 95]):
    """
    Plot Monte Carlo simulation results
    
    Args:
        simulation_results (pandas.DataFrame): DataFrame with simulation results
        percentiles (list): List of percentiles to highlight
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with Monte Carlo simulation
    """
    if simulation_results.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add sample of simulation traces
    sample_size = min(100, simulation_results.shape[1])
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
    
    # Add percentile traces
    colors = ['red', 'blue', 'green']
    for i, p in enumerate(percentiles):
        percentile_values = simulation_results.quantile(p/100, axis=1)
        fig.add_trace(
            go.Scatter(
                y=percentile_values,
                mode='lines',
                name=f'{p}th Percentile',
                line=dict(width=2, color=colors[i % len(colors)])
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'Monte Carlo Simulation ({simulation_results.shape[1]} runs)',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

def plot_risk_metrics_radar(risk_metrics, benchmark_metrics=None):
    """
    Plot risk metrics as a radar chart
    
    Args:
        risk_metrics (dict): Dictionary with risk metrics
        benchmark_metrics (dict, optional): Dictionary with benchmark risk metrics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with risk metrics radar chart
    """
    if not risk_metrics:
        return None
    
    # Define metrics to include
    metrics = [
        'sharpe_ratio',
        'sortino_ratio',
        'annualized_return',
        'annualized_volatility',
        'max_drawdown',
        'var_95'
    ]
    
    # Define display names
    display_names = [
        'Sharpe Ratio',
        'Sortino Ratio',
        'Annual Return',
        'Annual Volatility',
        'Max Drawdown',
        'Value at Risk (95%)'
    ]
    
    # Prepare data
    portfolio_values = []
    for metric in metrics:
        value = risk_metrics.get(metric, 0)
        
        # Invert negative metrics so higher is always better on the radar chart
        if metric in ['max_drawdown', 'annualized_volatility', 'var_95']:
            # Scale to 0-1 range where 1 is best (lowest risk)
            value = max(0, 1 - abs(value))
        
        # Scale other metrics to 0-1 range
        elif metric in ['sharpe_ratio', 'sortino_ratio']:
            # Assume a range of 0 to 3 for ratios
            value = max(0, min(value / 3, 1))
        
        elif metric == 'annualized_return':
            # Assume a range of -0.5 to 0.5 for returns
            value = max(0, min((value + 0.5) / 1, 1))
        
        portfolio_values.append(value)
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio trace
    fig.add_trace(
        go.Scatterpolar(
            r=portfolio_values,
            theta=display_names,
            fill='toself',
            name='Portfolio'
        )
    )
    
    # Add benchmark trace if provided
    if benchmark_metrics:
        benchmark_values = []
        for metric in metrics:
            value = benchmark_metrics.get(metric, 0)
            
            # Apply same scaling as for portfolio values
            if metric in ['max_drawdown', 'annualized_volatility', 'var_95']:
                value = max(0, 1 - abs(value))
            elif metric in ['sharpe_ratio', 'sortino_ratio']:
                value = max(0, min(value / 3, 1))
            elif metric == 'annualized_return':
                value = max(0, min((value + 0.5) / 1, 1))
            
            benchmark_values.append(value)
        
        fig.add_trace(
            go.Scatterpolar(
                r=benchmark_values,
                theta=display_names,
                fill='toself',
                name='Benchmark'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Risk Metrics Comparison',
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

def plot_drawdown_chart(returns_data):
    """
    Plot drawdown chart
    
    Args:
        returns_data (pandas.Series): Series with returns data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with drawdown chart
    """
    if returns_data.empty:
        return None
    
    # Calculate cumulative returns
    cum_returns = (1 + returns_data).cumprod()
    
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
