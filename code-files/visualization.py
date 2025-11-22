import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from prophet.plot import plot_plotly
import plotly.graph_objects as go

class ForecastVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
    
    def create_forecast_chart(self, historical_data, forecast, title="Spending Forecast"):
        """Create a combined historical + forecast chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_data['ds'], historical_data['y'], 
                label='Historical Spending', color=self.colors[0], linewidth=2)
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 
                label='Forecast', color=self.colors[1], linewidth=2)
        
        # Plot confidence interval
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                       alpha=0.2, color=self.colors[1], label='Confidence Interval')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_goal_progress_chart(self, goal_progress):
        """Create a progress chart for goals"""
        if not goal_progress:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        goals = [f"Goal {p['goal']['id']}" for p in goal_progress]
        progress = [p['progress_percentage'] for p in goal_progress]
        colors = ['green' if p['on_track'] else 'orange' for p in goal_progress]
        
        bars = ax.bar(goals, progress, color=colors, alpha=0.7)
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.8, label='Target (100%)')
        
        # Add value labels on bars
        for bar, pct in zip(bars, progress):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{pct:.1f}%', ha='center', va='bottom')
        
        ax.set_ylabel('Progress (%)')
        ax.set_title('Goal Progress Tracking', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 120)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_interactive_forecast(self, historical_data, forecast):
        """Create interactive Plotly chart"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines',
            name='Historical Spending',
            line=dict(color='blue', width=3)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title='Spending Forecast with Confidence Interval',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            hovermode='x unified'
        )
        
        return fig