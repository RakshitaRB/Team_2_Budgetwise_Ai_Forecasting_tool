import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")

from utils.database import (
    get_transactions_time_series, 
    get_category_time_series,
    get_total_spending_time_series,
    get_user_transactions,
    get_forecast_ready_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def can_generate_forecast(email: str, min_transactions: int = 10, min_months: int = 2):
    try:
        transactions = get_user_transactions(email)
        
        print(f"Debug: Found {len(transactions)} total transactions")
        
        if len(transactions) < min_transactions:
            return False, f"Need at least {min_transactions} transactions (currently have {len(transactions)})"
        
        if transactions:
            valid_dates = []
            date_samples = []  
            
            for i, t in enumerate(transactions[:5]):  
                try:
                    date_str = t.get('date', '')
                    date_samples.append(f"Transaction {i}: '{date_str}'")
                    
                    if date_str:
                        parsed_date = pd.to_datetime(date_str, errors='coerce', format='mixed')
                        if not pd.isna(parsed_date):
                            valid_dates.append(parsed_date)
                            print(f"Debug: Successfully parsed '{date_str}' as {parsed_date}")
                        else:
                            print(f"Debug: Failed to parse '{date_str}'")
                            
                except Exception as e:
                    print(f"Debug: Error parsing date in transaction {i}: {e}")
                    continue
            
            print(f"Debug: Found {len(valid_dates)} valid dates out of {len(transactions)} transactions")
            print(f"Debug: Date samples: {date_samples}")
            
            if len(valid_dates) < 2:
                return False, f"Need at least 2 valid dates for forecasting (found {len(valid_dates)} valid dates)"
            
            min_date = min(valid_dates)
            max_date = max(valid_dates)
            date_range = max_date - min_date
            months_of_data = date_range.days / 30.0
            
            print(f"Debug - Date range: {min_date} to {max_date}")
            print(f"Debug - Days difference: {date_range.days}, Months: {months_of_data:.2f}")
            
            if months_of_data < min_months:
                return False, f"Need at least {min_months} months of data (currently have {months_of_data:.1f} months from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
        
        return True, "Sufficient data for forecasting"
        
    except Exception as e:
        logger.error(f"Error checking forecast feasibility: {e}")
        return False, f"Error checking data: {str(e)}"

def prepare_historical_data(time_series_data, target_column='amount_positive'):
    if time_series_data.empty:
        return pd.DataFrame()
    
    try:
        prophet_data = time_series_data[['date', target_column]].copy()
        prophet_data.columns = ['ds', 'y']
        
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        prophet_data = prophet_data.dropna()
        
        return prophet_data
        
    except Exception as e:
        logger.error(f"Error preparing historical data: {e}")
        return pd.DataFrame()

def generate_expense_forecast(email: str, category: str = None, periods: int = 6, frequency: str = 'M'):
    
    if not PROPHET_AVAILABLE:
        return None, "Prophet not installed. Please install with: pip install prophet"
    
    try:
        historical_data = get_forecast_ready_data(email, category, 'expense')
        
        if historical_data.empty:
            return None, "No historical data available for forecasting"
        
        if len(historical_data) < 3:  
            return None, "Insufficient data points for forecasting (need at least 3)"
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        model.fit(historical_data)
        
        if frequency == 'D':
            future_periods = periods
        elif frequency == 'W':
            future_periods = periods * 4  
        else:  
            future_periods = periods
        
        future = model.make_future_dataframe(periods=future_periods, freq=frequency)
        forecast = model.predict(future)
        
        if len(historical_data) > 1:
            x = np.arange(len(historical_data))
            y = historical_data['y'].values
            trend_slope = np.polyfit(x, y, 1)[0]
        else:
            trend_slope = 0
        
        result = {
            'historical': historical_data,
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            'model': model,
            'trend_slope': trend_slope,
            'forecast_periods': periods,
            'frequency': frequency,
            'category': category,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully generated forecast for {email}, category: {category}")
        return result, None
        
    except Exception as e:
        logger.error(f"Error generating expense forecast: {e}")
        return None, f"Forecasting error: {str(e)}"

def generate_income_forecast(email: str, periods: int = 6, frequency: str = 'M'):
    """
    Generate income forecast using Prophet
    
    Args:
        email: User email
        periods: Number of periods to forecast
        frequency: Frequency of data
    
    Returns:
        dict: Forecast results and metadata
    """
    
    if not PROPHET_AVAILABLE:
        return None, "Prophet not installed. Please install with: pip install prophet"
    
    try:
        historical_data = get_forecast_ready_data(email, None, 'income')
        
        if historical_data.empty:
            return None, "No income data available for forecasting"
        
        if len(historical_data) < 3:
            return None, "Insufficient income data points for forecasting (need at least 3)"
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.03,  
            seasonality_prior_scale=8.0
        )
        
        model.fit(historical_data)
        
        if frequency == 'D':
            future_periods = periods
        elif frequency == 'W':
            future_periods = periods * 4
        else:  
            future_periods = periods
        
        future = model.make_future_dataframe(periods=future_periods, freq=frequency)
        
        
        forecast = model.predict(future)
        
        
        if len(historical_data) > 1:
            x = np.arange(len(historical_data))
            y = historical_data['y'].values
            trend_slope = np.polyfit(x, y, 1)[0]
        else:
            trend_slope = 0
        
        
        result = {
            'historical': historical_data,
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            'model': model,
            'trend_slope': trend_slope,
            'forecast_periods': periods,
            'frequency': frequency,
            'category': 'income',
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully generated income forecast for {email}")
        return result, None
        
    except Exception as e:
        logger.error(f"Error generating income forecast: {e}")
        return None, f"Income forecasting error: {str(e)}"

def generate_category_comparison_forecast(email: str, periods: int = 3):
    try:
        transactions = get_user_transactions(email)
        if not transactions:
            return None, "No transactions available"
        
        df = pd.DataFrame(transactions)
        expense_df = df[df['type'] == 'expense']
        
        if expense_df.empty:
            return None, "No expense data available"
        
        category_totals = expense_df.groupby('category')['amount'].apply(lambda x: abs(x).sum())
        top_categories = category_totals.nlargest(5).index.tolist()
        
        category_forecasts = {}
        
        for category in top_categories:
            forecast, error = generate_expense_forecast(email, category, periods, 'M')
            if forecast and not error:
                last_historical = forecast['historical']['y'].iloc[-1] if not forecast['historical'].empty else 0
                avg_forecast = forecast['forecast']['yhat'].mean()
                
                category_forecasts[category] = {
                    'last_historical': last_historical,
                    'avg_forecast': avg_forecast,
                    'trend': 'increasing' if forecast['trend_slope'] > 0 else 'decreasing',
                    'trend_strength': abs(forecast['trend_slope'])
                }
        
        return category_forecasts, None
        
    except Exception as e:
        logger.error(f"Error generating category comparison forecast: {e}")
        return None, f"Category comparison error: {str(e)}"

def calculate_savings_forecast(income_forecast, expense_forecast):
    try:
        if not income_forecast or not expense_forecast:
            return None
        
        
        income_df = income_forecast['forecast'].copy()
        expense_df = expense_forecast['forecast'].copy()
        
        
        merged = pd.merge(income_df, expense_df, on='ds', suffixes=('_income', '_expense'))
        
        
        merged['projected_savings'] = merged['yhat_income'] - merged['yhat_expense']
        merged['savings_lower'] = merged['yhat_lower_income'] - merged['yhat_upper_expense']
        merged['savings_upper'] = merged['yhat_upper_income'] - merged['yhat_lower_expense']
        
        return merged[['ds', 'projected_savings', 'savings_lower', 'savings_upper']]
        
    except Exception as e:
        logger.error(f"Error calculating savings forecast: {e}")
        return None