# utils/database.py
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import uuid
import re

DATA_FILE = 'data/users.json'

def init_database():
    '''Initialize the database file if it doesn't exist'''
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump({'users': {}, 'transactions': {}, 'goals': {}, 'forecasts': {}}, f)

def load_data():
    '''Load all data from JSON file'''
    init_database()
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            
        # Ensure all required keys exist
        required_keys = ['users', 'transactions', 'goals', 'forecasts']
        for key in required_keys:
            if key not in data:
                data[key] = {}
                
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or doesn't exist, reinitialize
        init_database()
        return {'users': {}, 'transactions': {}, 'goals': {}, 'forecasts': {}}

def save_data(data):
    '''Save all data to JSON file'''
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def parse_and_format_date(date_input):
    """
    Helper function to parse and format dates consistently
    
    Args:
        date_input: Date in various formats (string, datetime, etc.)
    
    Returns:
        str: Date in YYYY-MM-DD format
    """
    if date_input is None:
        return datetime.now().date().isoformat()
    
    if isinstance(date_input, datetime):
        return date_input.date().isoformat()
    
    if isinstance(date_input, str):
        # Clean the string
        date_str = date_input.strip()
        
        # If it's already in YYYY-MM-DD format, return as-is
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            try:
                # Validate it's a real date
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except ValueError:
                pass
        
        # Remove time component if present
        if 'T' in date_str:
            date_str = date_str.split('T')[0]
        elif ' ' in date_str:
            date_str = date_str.split(' ')[0]
        
        # Try ISO format first
        try:
            parsed = datetime.strptime(date_str, '%Y-%m-%d')
            return parsed.date().isoformat()
        except ValueError:
            pass
        
        # Try other common formats
        formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%m-%d-%Y', '%d-%m-%Y', '%Y.%m.%d',
            '%m/%d/%y', '%d/%m/%y',
            '%d-%b-%Y', '%d %b %Y', '%b %d, %Y',
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.date().isoformat()
            except ValueError:
                continue
        
        # Try pandas as final fallback
        try:
            parsed = pd.to_datetime(date_str, errors='coerce')
            if not pd.isna(parsed):
                return parsed.strftime('%Y-%m-%d')
        except:
            pass
    
    # Final fallback to today's date
    return datetime.now().date().isoformat()

def add_user(username: str, email: str, hashed_password: str, role: str = 'user'):
    '''Add a new user to the database'''
    data = load_data()
    
    if email in data['users']:
        return False, 'User already exists'
    
    data['users'][email] = {
        'username': username,
        'email': email,
        'hashed_password': hashed_password,
        'role': role,  # NEW: Add role field
        'created_at': datetime.now().isoformat()
    }
    
    # Initialize user-specific data structures
    data['transactions'][email] = []
    data['goals'][email] = []
    data['forecasts'][email] = []
    
    save_data(data)
    return True, 'User created successfully'

def verify_user(email: str, password: str):
    '''Verify user credentials'''
    data = load_data()
    
    if email not in data['users']:
        return False, 'User not found'
    
    user = data['users'][email]
    from utils.auth import verify_password
    if verify_password(password, user['hashed_password']):
        return True, 'Login successful'
    else:
        return False, 'Invalid password'

def add_transaction(email: str, transaction_data: dict):
    '''Add a transaction for a user'''
    data = load_data()
    
    # Ensure user has transactions array
    if email not in data['transactions']:
        data['transactions'][email] = []
    
    # Generate unique ID if not provided
    if 'id' not in transaction_data:
        transaction_data['id'] = str(uuid.uuid4())[:8]
    
    # 🎯 CRITICAL FIX: Use the date from transaction_data, don't override it!
    # Only parse and format the date, don't replace it with today's date
    if 'date' in transaction_data:
        # Use helper function to ensure consistent format, but preserve the original date
        transaction_data['date'] = parse_and_format_date(transaction_data['date'])
    else:
        # Only use today's date if no date was provided
        transaction_data['date'] = parse_and_format_date(datetime.now())
    
    data['transactions'][email].append(transaction_data)
    save_data(data)
    return True, 'Transaction added successfully'

def get_user_transactions(email: str):
    '''Get all transactions for a user'''
    data = load_data()
    
    # Ensure user exists in transactions
    if 'transactions' not in data or email not in data['transactions']:
        return []
    
    transactions = data['transactions'].get(email, [])
    
    # Ensure all transactions have required fields
    for transaction in transactions:
        if 'id' not in transaction:
            transaction['id'] = str(uuid.uuid4())[:8]
        if 'type' not in transaction:
            # Try to infer type from amount
            transaction['type'] = 'income' if float(transaction.get('amount', 0)) >= 0 else 'expense'
        if 'category' not in transaction:
            transaction['category'] = 'Other'
    
    return transactions

def update_user_profile(email: str, profile_data: dict):
    '''Update user profile information'''
    data = load_data()
    
    if email not in data['users']:
        return False, 'User not found'
    
    data['users'][email].update(profile_data)
    save_data(data)
    return True, 'Profile updated successfully'

def manual_categorization(transaction_id: str, new_category: str, email: str):
    '''Manually update transaction category'''
    data = load_data()
    
    if email not in data['transactions']:
        return False, 'No transactions found for user'
    
    # Find and update the transaction
    for transaction in data['transactions'][email]:
        if str(transaction.get('id')) == str(transaction_id):
            transaction['category'] = new_category
            save_data(data)
            return True, 'Category updated successfully'
    
    return False, 'Transaction not found'

# ===== MILESTONE 3: FINANCIAL GOALS FUNCTIONS =====

def add_financial_goal(email: str, goal_data: dict):
    '''Add a new financial goal for the user'''
    data = load_data()
    
    # Ensure goals structure exists for user
    if 'goals' not in data:
        data['goals'] = {}
    if email not in data['goals']:
        data['goals'][email] = []
    
    # Generate unique ID for the goal
    goal_data['id'] = str(uuid.uuid4())[:8]
    goal_data['created_at'] = datetime.now().isoformat()
    goal_data['current_amount'] = float(goal_data.get('current_savings', 0))  # Track current savings
    goal_data['status'] = 'active'  # active, completed, failed
    
    data['goals'][email].append(goal_data)
    save_data(data)
    return True, 'Goal created successfully'

def get_user_goals(email: str):
    '''Get all financial goals for a user - UPDATED WITH FIXED PROGRESS'''
    data = load_data()
    
    # Ensure goals structure exists
    if 'goals' not in data or email not in data['goals']:
        return []
    
    goals = data['goals'].get(email, [])
    
    # Ensure all goals have required fields and calculate current progress
    for goal in goals:
        if 'id' not in goal:
            goal['id'] = str(uuid.uuid4())[:8]
        if 'current_amount' not in goal:
            goal['current_amount'] = 0.0
        if 'created_at' not in goal:
            goal['created_at'] = datetime.now().isoformat()
        if 'status' not in goal:
            goal['status'] = 'active'
        
        # 🎯 USE THE FIXED PROGRESS CALCULATION
        goal['current_progress'] = calculate_savings_goal_progress(email, goal)
        
        # Calculate current total towards this specific goal
        if 'current_towards_goal' not in goal:
            goal['current_towards_goal'] = float(goal.get('current_savings', 0))
    
    return goals

def calculate_goal_progress(email: str, goal: dict):
    """Calculate current progress for a goal based on its type"""
    goal_type = goal.get('goal_type', 'savings_goal')
    
    try:
        if goal_type == 'savings_goal':
            return calculate_savings_goal_progress(email, goal)
        elif goal_type == 'spending_reduction':
            return calculate_spending_reduction_progress(email, goal)
        elif goal_type == 'category_budget':
            return calculate_category_budget_progress(email, goal)
        else:
            return 0.0
    except Exception as e:
        print(f"Error calculating goal progress: {e}")
        return 0.0

def calculate_total_saved_for_goal(email: str, goal: dict):
    """Calculate total saved amount for a savings goal"""
    try:
        if goal.get('goal_type') != 'savings_goal':
            return 0.0
        
        initial_savings = float(goal.get('current_amount', 0))
        goal_created_at = goal.get('created_at')
        
        if not goal_created_at:
            return initial_savings
        
        # Get all income transactions AFTER goal creation
        transactions = get_user_transactions(email)
        if not transactions:
            return initial_savings
        
        goal_created_dt = datetime.fromisoformat(goal_created_at)
        income_after_goal = 0
        
        for transaction in transactions:
            transaction_date = datetime.fromisoformat(transaction['date'])
            # Count income transactions that happened AFTER goal creation
            if (transaction_date >= goal_created_dt and 
                transaction.get('type') == 'income'):
                amount = float(transaction.get('amount', 0))
                income_after_goal += amount
        
        # Total saved = initial savings + income after goal creation
        total_saved = initial_savings + income_after_goal
        return total_saved
        
    except Exception as e:
        print(f"Error calculating total saved for goal: {e}")
        return float(goal.get('current_amount', 0))

def calculate_savings_goal_progress(email: str, goal: dict):
    """Calculate progress for savings goals - FIXED VERSION"""
    try:
        target_amount = float(goal.get('target_amount', 0))
        initial_savings = float(goal.get('current_savings', 0))
        goal_created_at = goal.get('created_at')
        
        if target_amount <= 0:
            return 0.0
        
        if not goal_created_at:
            # If no creation date, use total savings (fallback)
            total_savings = get_total_savings(email)
            progress = min((total_savings / target_amount) * 100, 100)
            return round(progress, 1)
        
        # 🎯 CRITICAL FIX: Calculate savings SINCE goal creation
        savings_since_goal = get_savings_since_goal_creation(email, goal_created_at)
        
        # Total towards goal = initial savings + savings accumulated since goal creation
        total_towards_goal = initial_savings + savings_since_goal
        
        # Progress is based on how much of target we've saved towards this specific goal
        progress = min((total_towards_goal / target_amount) * 100, 100)
        
        # Store the actual amount saved towards this goal for display
        goal['current_towards_goal'] = total_towards_goal
        
        return round(progress, 1)
        
    except Exception as e:
        print(f"Error calculating savings goal progress: {e}")
        return 0.0

def get_savings_since_goal_creation(email: str, goal_created_date: str):
    '''Calculate savings accumulated SINCE a goal was created'''
    transactions = get_user_transactions(email)
    
    if not transactions:
        return 0
    
    try:
        goal_date = datetime.fromisoformat(goal_created_date.replace('Z', '+00:00'))
        
        total_income_since = 0
        total_expenses_since = 0
        
        for transaction in transactions:
            transaction_date = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00'))
            
            # Only count transactions that happened AFTER goal creation
            if transaction_date >= goal_date:
                amount = float(transaction.get('amount', 0))
                if transaction.get('type') == 'income':
                    total_income_since += amount
                else:  # expense
                    total_expenses_since += abs(amount)
        
        savings_since = total_income_since - total_expenses_since
        return max(0, savings_since)  # Don't return negative
        
    except Exception as e:
        print(f"Error calculating savings since goal creation: {e}")
        return 0

def get_total_savings(email: str):
    '''Calculate total savings (income - expenses) for a user - FOR DASHBOARD DISPLAY'''
    transactions = get_user_transactions(email)
    
    if not transactions:
        return 0
    
    try:
        total_income = 0
        total_expenses = 0
        
        for transaction in transactions:
            amount = float(transaction.get('amount', 0))
            if transaction.get('type') == 'income':
                total_income += amount
            else:  # expense
                total_expenses += abs(amount)  # expenses are stored as negative
        
        savings = total_income - total_expenses
        return max(0, savings)  # Don't return negative savings
        
    except Exception as e:
        print(f"Error calculating total savings: {e}")
        return 0

def calculate_spending_reduction_progress(email: str, goal: dict):
    """Calculate progress for spending reduction goals"""
    try:
        # For now, return a simple progress based on transactions
        category = goal.get('category')
        if not category:
            return 0.0
        
        # Count transactions in this category since goal creation
        goal_created_at = goal.get('created_at')
        if not goal_created_at:
            return 0.0
        
        transactions = get_user_transactions(email)
        if not transactions:
            return 0.0
        
        goal_created_dt = datetime.fromisoformat(goal_created_at)
        category_transactions = 0
        
        for transaction in transactions:
            transaction_date = datetime.fromisoformat(transaction['date'])
            if (transaction_date >= goal_created_dt and 
                transaction.get('category') == category and 
                transaction.get('type') == 'expense'):
                category_transactions += 1
        
        # Simple progress: 10% per transaction, max 50% until we implement proper reduction logic
        progress = min(category_transactions * 10, 50)
        return progress
        
    except Exception as e:
        print(f"Error calculating spending reduction progress: {e}")
        return 0.0

def calculate_category_budget_progress(email: str, goal: dict):
    """Calculate progress for category budget goals"""
    try:
        category = goal.get('category')
        budget_limit = float(goal.get('budget_limit', 0))
        
        if not category or budget_limit <= 0:
            return 0.0
        
        # Get current month's spending in the category
        now = datetime.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        current_spending = get_category_spending(email, category, 
                                               start_of_month.isoformat(), 
                                               now.isoformat())
        
        if current_spending == 0:
            return 0.0
        
        progress = (current_spending / budget_limit) * 100
        return min(progress, 100)
        
    except Exception as e:
        print(f"Error calculating category budget progress: {e}")
        return 0.0

def update_goal_progress(email: str, goal_id: str, progress: float):
    '''Update progress for a financial goal'''
    data = load_data()
    
    if email not in data.get('goals', {}):
        return False, 'No goals found for user'
    
    # Find and update the goal
    for goal in data['goals'][email]:
        if str(goal.get('id')) == str(goal_id):
            goal['progress'] = progress
            goal['updated_at'] = datetime.now().isoformat()
            save_data(data)
            return True, 'Goal progress updated successfully'
    
    return False, 'Goal not found'

def delete_financial_goal(email: str, goal_id: str):
    '''Delete a financial goal'''
    data = load_data()
    
    if email not in data.get('goals', {}):
        return False, 'No goals found for user'
    
    # Find and remove the goal
    for i, goal in enumerate(data['goals'][email]):
        if str(goal.get('id')) == str(goal_id):
            del data['goals'][email][i]
            save_data(data)
            return True, 'Goal deleted successfully'
    
    return False, 'Goal not found'

# ===== MILESTONE 3: FORECASTING FUNCTIONS =====

def save_forecast_data(email: str, forecast_data: dict):
    '''Save forecast data for a user'''
    data = load_data()
    
    # Ensure forecasts structure exists
    if 'forecasts' not in data:
        data['forecasts'] = {}
    if email not in data['forecasts']:
        data['forecasts'][email] = []
    
    # Generate unique ID for the forecast
    forecast_data['id'] = str(uuid.uuid4())[:8]
    forecast_data['created_at'] = datetime.now().isoformat()
    forecast_data['user_email'] = email
    
    # Remove old forecasts (keep only last 5 to save space)
    if len(data['forecasts'][email]) >= 5:
        data['forecasts'][email] = data['forecasts'][email][-4:]
    
    data['forecasts'][email].append(forecast_data)
    save_data(data)
    return True, 'Forecast saved successfully'

def get_forecast_data(email: str, limit: int = 5):
    '''Get saved forecast data for a user'''
    data = load_data()
    
    if 'forecasts' not in data or email not in data['forecasts']:
        return []
    
    forecasts = data['forecasts'].get(email, [])
    
    # Return most recent forecasts first
    forecasts_sorted = sorted(forecasts, 
                            key=lambda x: x.get('created_at', ''), 
                            reverse=True)
    
    return forecasts_sorted[:limit]

def get_recent_forecast(email: str):
    '''Get the most recent forecast for a user'''
    data = load_data()
    
    if 'forecasts' not in data or email not in data['forecasts']:
        return None
    
    forecasts = data['forecasts'].get(email, [])
    
    if not forecasts:
        return None
    
    # Return the most recent forecast
    most_recent = max(forecasts, key=lambda x: x.get('created_at', ''))
    return most_recent

# ===== TIME SERIES DATA PREPARATION =====

def get_transactions_time_series(email, frequency='M'):
    """Get transactions as time series data with ROBUST date parsing"""
    transactions = get_user_transactions(email)
    if not transactions:
        return pd.DataFrame()
    
    df = pd.DataFrame(transactions)
    
    # ROBUST DATE PARSING
    def safe_date_parse(date_str):
        if not date_str or pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        # Try multiple date formats
        date_formats = [
            '%Y-%m-%d',           # 2023-01-15
            '%m/%d/%Y',           # 01/15/2023
            '%d/%m/%Y',           # 15/01/2023
            '%Y/%m/%d',           # 2023/01/15
            '%d-%m-%Y',           # 15-01-2023
            '%m-%d-%Y',           # 01-15-2023
            '%Y%m%d',             # 20230115
            '%b %d, %Y',          # Jan 15, 2023
            '%d %b %Y',           # 15 Jan 2023
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Final fallback
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    df['parsed_date'] = df['date'].apply(safe_date_parse)
    
    # Remove invalid dates
    df = df[df['parsed_date'].notna()]
    
    if df.empty:
        return pd.DataFrame()
    
    # Create positive amounts for analysis
    df['amount_positive'] = df['amount'].abs()
    
    # Group by time frequency
    df.set_index('parsed_date', inplace=True)
    
    if frequency == 'D':
        grouped = df.groupby(pd.Grouper(freq='D'))
    elif frequency == 'W':
        grouped = df.groupby(pd.Grouper(freq='W'))
    else:  # Monthly
        grouped = df.groupby(pd.Grouper(freq='M'))
    
    time_series = grouped.agg({
        'amount': 'sum',
        'amount_positive': 'sum',
        'type': 'first'
    }).reset_index()
    
    time_series = time_series.rename(columns={'parsed_date': 'date'})
    
    return time_series

    
def get_category_time_series(email: str, category: str, frequency: str = 'M'):
    """Get time series data for a specific category"""
    ts_data = get_transactions_time_series(email, frequency)
    
    if ts_data.empty:
        return pd.DataFrame()
    
    try:
        category_data = ts_data[ts_data['category'] == category]
        return category_data
    except Exception as e:
        print(f"Error getting category time series: {e}")
        return pd.DataFrame()

def get_total_spending_time_series(email: str, frequency: str = 'M'):
    """Get total spending time series (all categories combined)"""
    ts_data = get_transactions_time_series(email, frequency)
    
    if ts_data.empty:
        return pd.DataFrame()
    
    try:
        total_data = ts_data[ts_data['type'] == 'expense'].groupby('date')['amount_positive'].sum().reset_index()
        return total_data
    except Exception as e:
        print(f"Error getting total spending time series: {e}")
        return pd.DataFrame()

# ===== FORECAST-READY DATA FUNCTION =====

def get_forecast_ready_data(email: str, category: str = None, transaction_type: str = 'expense'):
    """
    Get data ready for forecasting in the required format
    
    Args:
        email: User email
        category: Specific category (None for all)
        transaction_type: 'expense' or 'income'
    
    Returns:
        DataFrame: Data formatted for forecasting with columns ['ds', 'y']
    """
    transactions = get_user_transactions(email)
    
    if not transactions:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
        
        # Remove invalid dates
        df = df.dropna(subset=['date'])
        
        # Filter by type
        df = df[df['type'] == transaction_type]
        
        # Filter by category if specified
        if category:
            df = df[df['category'] == category]
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert to positive amounts for analysis (Prophet requires positive values)
        df['amount_positive'] = df['amount'].abs()
        
        # Aggregate by month (most common frequency for financial forecasting)
        monthly_data = df.groupby(pd.Grouper(key='date', freq='M'))['amount_positive'].sum().reset_index()
        
        # Rename columns to Prophet format: 'ds' for dates, 'y' for values
        monthly_data = monthly_data.rename(columns={'date': 'ds', 'amount_positive': 'y'})
        
        # Remove any rows with zero or negative values (Prophet requires positive values)
        monthly_data = monthly_data[monthly_data['y'] > 0]
        
        # Sort by date
        monthly_data = monthly_data.sort_values('ds').reset_index(drop=True)
        
        return monthly_data
        
    except Exception as e:
        print(f"Error preparing forecast-ready data: {e}")
        return pd.DataFrame()

# ===== HELPER FUNCTIONS FOR GOAL CALCULATIONS =====

def get_user_transactions_by_date_range(email: str, start_date: str, end_date: str):
    '''Get transactions for a user within a specific date range'''
    transactions = get_user_transactions(email)
    
    if not transactions:
        return []
    
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        filtered_transactions = []
        for transaction in transactions:
            transaction_date = datetime.fromisoformat(transaction['date'])
            if start_dt <= transaction_date <= end_dt:
                filtered_transactions.append(transaction)
        
        return filtered_transactions
    except (ValueError, KeyError):
        return transactions

def get_category_spending(email: str, category: str, start_date: str = None, end_date: str = None):
    '''Get total spending for a specific category'''
    transactions = get_user_transactions(email)
    
    if not transactions:
        return 0
    
    try:
        category_transactions = [t for t in transactions if t.get('category') == category and t.get('type') == 'expense']
        
        if start_date and end_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
                category_transactions = [
                    t for t in category_transactions 
                    if start_dt <= datetime.fromisoformat(t['date']) <= end_dt
                ]
            except (ValueError, KeyError):
                pass
        
        total_spending = sum(abs(float(t.get('amount', 0))) for t in category_transactions)
        return total_spending
    except Exception as e:
        print(f"Error calculating category spending: {e}")
        return 0

def get_total_savings_since_date(email: str, since_date: str):
    '''Calculate total savings since a specific date'''
    transactions = get_user_transactions(email)
    
    if not transactions:
        return 0
    
    try:
        since_dt = datetime.fromisoformat(since_date)
        
        total_income = 0
        total_expenses = 0
        
        for transaction in transactions:
            transaction_date = datetime.fromisoformat(transaction['date'])
            if transaction_date >= since_dt:
                amount = float(transaction.get('amount', 0))
                if transaction.get('type') == 'income':
                    total_income += amount
                else:  # expense
                    total_expenses += abs(amount)  # expenses are negative, so use abs
        
        return total_income - total_expenses
    except Exception as e:
        print(f"Error calculating savings since date: {e}")
        return 0

# ===== TRANSACTION STATISTICS FUNCTIONS =====

def get_transaction_statistics(email: str):
    '''Get comprehensive transaction statistics for forecasting'''
    transactions = get_user_transactions(email)
    
    if not transactions:
        return None
    
    try:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
        
        # Remove invalid dates
        df = df.dropna(subset=['date'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Convert expenses to positive for analysis
        df['positive_amount'] = df['amount'].abs()
        
        stats = {
            'total_transactions': len(df),
            'total_income': df[df['type'] == 'income']['positive_amount'].sum(),
            'total_expenses': df[df['type'] == 'expense']['positive_amount'].sum(),
            'avg_daily_spending': df[df['type'] == 'expense'].groupby(df['date'].dt.date)['positive_amount'].sum().mean(),
            'most_common_category': df[df['type'] == 'expense']['category'].mode().iloc[0] if not df[df['type'] == 'expense'].empty else 'Other',
            'transaction_timeline': df.groupby(df['date'].dt.to_period('M'))['positive_amount'].sum().to_dict(),
            'category_breakdown': df[df['type'] == 'expense'].groupby('category')['positive_amount'].sum().to_dict()
        }
        
        return stats
    except Exception as e:
        print(f"Error getting transaction statistics: {e}")
        return None

# ===== ADMIN ANALYTICS FUNCTIONS =====

def get_all_users_stats():
    '''Get statistics about all users (without exposing financial data)'''
    data = load_data()
    
    users_data = data.get('users', {})
    transactions_data = data.get('transactions', {})
    
    # User statistics
    total_users = len(users_data)
    admin_users = sum(1 for user in users_data.values() if user.get('role') == 'admin')
    regular_users = total_users - admin_users
    
    # Registration dates analysis
    registration_dates = []
    for user in users_data.values():
        try:
            reg_date = datetime.fromisoformat(user.get('created_at', ''))
            registration_dates.append(reg_date)
        except:
            pass
    
    # Activity analysis (users with transactions)
    active_users = sum(1 for email in transactions_data if len(transactions_data.get(email, [])) > 0)
    inactive_users = total_users - active_users
    
    # Monthly registration trend
    monthly_registrations = {}
    for date in registration_dates:
        month_key = date.strftime('%Y-%m')
        monthly_registrations[month_key] = monthly_registrations.get(month_key, 0) + 1
    
    return {
        'total_users': total_users,
        'admin_users': admin_users,
        'regular_users': regular_users,
        'active_users': active_users,
        'inactive_users': inactive_users,
        'registration_dates': registration_dates,
        'monthly_registrations': monthly_registrations
    }

def get_platform_financial_insights():
    '''Get aggregated financial insights (no individual user data)'''
    data = load_data()
    transactions_data = data.get('transactions', {})
    
    # Aggregate category usage across all users
    category_usage = {}
    total_transactions = 0
    user_transaction_counts = []
    
    for email, transactions in transactions_data.items():
        user_tx_count = len(transactions)
        user_transaction_counts.append(user_tx_count)
        total_transactions += user_tx_count
        
        # Count category usage
        for transaction in transactions:
            category = transaction.get('category', 'Other')
            category_usage[category] = category_usage.get(category, 0) + 1
    
    # Most common categories (top 10)
    popular_categories = dict(sorted(category_usage.items(), 
                                   key=lambda x: x[1], reverse=True)[:10])
    
    # Transaction type distribution
    expense_count = 0
    income_count = 0
    for transactions in transactions_data.values():
        for transaction in transactions:
            if transaction.get('type') == 'expense':
                expense_count += 1
            else:
                income_count += 1
    
    return {
        'total_transactions': total_transactions,
        'avg_transactions_per_user': total_transactions / max(1, len(transactions_data)),
        'popular_categories': popular_categories,
        'expense_count': expense_count,
        'income_count': income_count,
        'user_transaction_counts': user_transaction_counts
    }

def get_forecasting_analytics():
    '''Get analytics about forecasting usage'''
    data = load_data()
    forecasts_data = data.get('forecasts', {})
    transactions_data = data.get('transactions', {})
    
    # Forecast usage statistics
    users_with_forecasts = sum(1 for forecasts in forecasts_data.values() if len(forecasts) > 0)
    total_forecasts = sum(len(forecasts) for forecasts in forecasts_data.values())
    
    # Most popular forecast categories
    forecast_categories = {}
    forecast_types = {}
    
    for email, forecasts in forecasts_data.items():
        for forecast in forecasts:
            # Count forecast types
            forecast_type = forecast.get('forecast_type', 'unknown')
            forecast_types[forecast_type] = forecast_types.get(forecast_type, 0) + 1
            
            # Count categories for expense forecasts
            if forecast_type == 'expense' and 'category' in forecast:
                category = forecast.get('category', 'Other')
                forecast_categories[category] = forecast_categories.get(category, 0) + 1
    
    return {
        'users_with_forecasts': users_with_forecasts,
        'total_forecasts': total_forecasts,
        'forecast_types': forecast_types,
        'popular_forecast_categories': forecast_categories,
        'forecast_usage_rate': (users_with_forecasts / max(1, len(transactions_data))) * 100
    }

def get_system_health_metrics():
    '''Get system health and performance metrics'''
    data = load_data()
    
    # Database size estimation
    db_size = len(json.dumps(data))
    
    # Data quality metrics
    valid_transactions = 0
    total_transactions = 0
    transactions_with_dates = 0
    transactions_with_categories = 0
    
    for email, transactions in data.get('transactions', {}).items():
        for transaction in transactions:
            total_transactions += 1
            if all(key in transaction for key in ['description', 'amount', 'type']):
                valid_transactions += 1
            if 'date' in transaction and transaction['date']:
                transactions_with_dates += 1
            if 'category' in transaction and transaction['category']:
                transactions_with_categories += 1
    
    # Goal completion rates
    total_goals = 0
    completed_goals = 0
    for email, goals in data.get('goals', {}).items():
        for goal in goals:
            total_goals += 1
            if goal.get('status') == 'completed' or goal.get('current_progress', 0) >= 100:
                completed_goals += 1
    
    data_quality_score = (valid_transactions / max(1, total_transactions)) * 100
    goal_completion_rate = (completed_goals / max(1, total_goals)) * 100
    
    return {
        'database_size_kb': db_size / 1024,
        'total_transactions': total_transactions,
        'valid_transactions': valid_transactions,
        'transactions_with_dates': transactions_with_dates,
        'transactions_with_categories': transactions_with_categories,
        'data_quality_score': data_quality_score,
        'total_goals': total_goals,
        'completed_goals': completed_goals,
        'goal_completion_rate': goal_completion_rate
    }

def get_user_activity_metrics():
    '''Get user engagement and activity metrics'''
    data = load_data()
    users_data = data.get('users', {})
    transactions_data = data.get('transactions', {})
    goals_data = data.get('goals', {})
    forecasts_data = data.get('forecasts', {})
    
    # User engagement analysis
    engaged_users = 0
    highly_engaged_users = 0
    
    for email in users_data:
        tx_count = len(transactions_data.get(email, []))
        goals_count = len(goals_data.get(email, []))
        forecasts_count = len(forecasts_data.get(email, []))
        
        # Basic engagement: has transactions
        if tx_count > 0:
            engaged_users += 1
        
        # High engagement: has transactions + goals or forecasts
        if tx_count > 5 and (goals_count > 0 or forecasts_count > 0):
            highly_engaged_users += 1
    
    # Feature adoption rates
    users_with_goals = sum(1 for goals in goals_data.values() if len(goals) > 0)
    users_with_forecasts = sum(1 for forecasts in forecasts_data.values() if len(forecasts) > 0)
    
    total_users = len(users_data)
    
    return {
        'total_users': total_users,
        'engaged_users': engaged_users,
        'highly_engaged_users': highly_engaged_users,
        'users_with_goals': users_with_goals,
        'users_with_forecasts': users_with_forecasts,
        'engagement_rate': (engaged_users / max(1, total_users)) * 100,
        'goal_adoption_rate': (users_with_goals / max(1, total_users)) * 100,
        'forecast_adoption_rate': (users_with_forecasts / max(1, total_users)) * 100
    }

def get_all_users_stats():
    '''Get statistics about all users (without exposing financial data)'''
    data = load_data()
    
    users_data = data.get('users', {})
    transactions_data = data.get('transactions', {})
    
    # User statistics
    total_users = len(users_data)
    admin_users = sum(1 for user in users_data.values() if user.get('role') == 'admin')
    regular_users = total_users - admin_users
    
    # Registration dates analysis
    registration_dates = []
    for user in users_data.values():
        try:
            created_at = user.get('created_at', '')
            if created_at:
                # Handle different date formats
                if 'T' in created_at:
                    reg_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    reg_date = datetime.fromisoformat(created_at)
                registration_dates.append(reg_date)
        except Exception as e:
            print(f"Error parsing registration date: {e}")
            continue
    
    # Activity analysis (users with transactions)
    active_users = sum(1 for email in transactions_data if transactions_data.get(email) and len(transactions_data.get(email, [])) > 0)
    inactive_users = max(0, total_users - active_users)
    
    # Monthly registration trend
    monthly_registrations = {}
    for date in registration_dates:
        try:
            month_key = date.strftime('%Y-%m')
            monthly_registrations[month_key] = monthly_registrations.get(month_key, 0) + 1
        except:
            continue
    
    # Ensure we have at least some data
    if not monthly_registrations:
        monthly_registrations = {'2024-01': 1}  # Default fallback
    
    return {
        'total_users': total_users,
        'admin_users': admin_users,
        'regular_users': regular_users,
        'active_users': active_users,
        'inactive_users': inactive_users,
        'registration_dates': registration_dates,
        'monthly_registrations': monthly_registrations
    }

def get_platform_financial_insights():
    '''Get aggregated financial insights (no individual user data)'''
    data = load_data()
    transactions_data = data.get('transactions', {})
    
    # Aggregate category usage across all users
    category_usage = {}
    total_transactions = 0
    user_transaction_counts = []
    expense_count = 0
    income_count = 0
    
    for email, transactions in transactions_data.items():
        if not transactions:
            continue
            
        user_tx_count = len(transactions)
        user_transaction_counts.append(user_tx_count)
        total_transactions += user_tx_count
        
        # Count category usage and transaction types
        for transaction in transactions:
            category = transaction.get('category', 'Other')
            category_usage[category] = category_usage.get(category, 0) + 1
            
            tx_type = transaction.get('type', 'expense')
            if tx_type == 'expense':
                expense_count += 1
            else:
                income_count += 1
    
    # Most common categories (top 10)
    popular_categories = dict(sorted(category_usage.items(), 
                                   key=lambda x: x[1], reverse=True)[:10])
    
    # Calculate average transactions per user
    avg_tx_per_user = total_transactions / max(1, len(transactions_data))
    
    return {
        'total_transactions': total_transactions,
        'avg_transactions_per_user': round(avg_tx_per_user, 1),
        'popular_categories': popular_categories,
        'expense_count': expense_count,
        'income_count': income_count,
        'user_transaction_counts': user_transaction_counts
    }

def get_forecasting_analytics():
    '''Get analytics about forecasting usage'''
    data = load_data()
    forecasts_data = data.get('forecasts', {})
    transactions_data = data.get('transactions', {})
    
    
    users_with_forecasts = sum(1 for email, forecasts in forecasts_data.items() if forecasts and len(forecasts) > 0)
    total_forecasts = sum(len(forecasts) for forecasts in forecasts_data.values() if forecasts)
    forecast_categories = {}
    forecast_types = {}
    for email, forecasts in forecasts_data.items():
        if not forecasts:
            continue
        for forecast in forecasts:
            forecast_type = forecast.get('forecast_type', 'unknown')
            forecast_types[forecast_type] = forecast_types.get(forecast_type, 0) + 1
            
            
            if forecast_type == 'expense' and 'category' in forecast:
                category = forecast.get('category', 'Other')
                forecast_categories[category] = forecast_categories.get(category, 0) + 1
            elif 'category' in forecast:
                category = forecast.get('category', 'Other')
                forecast_categories[category] = forecast_categories.get(category, 0) + 1
    
    total_users_with_transactions = sum(1 for tx in transactions_data.values() if tx and len(tx) > 0)
    forecast_usage_rate = (users_with_forecasts / max(1, total_users_with_transactions)) * 100
    if not forecast_types:
        forecast_types = {'expense': 0, 'income': 0}
    
    return {
        'users_with_forecasts': users_with_forecasts,
        'total_forecasts': total_forecasts,
        'forecast_types': forecast_types,
        'popular_forecast_categories': forecast_categories,
        'forecast_usage_rate': round(forecast_usage_rate, 1)
    }

def get_system_health_metrics():
    data = load_data()
    db_size = len(json.dumps(data))
    valid_transactions = 0
    total_transactions = 0
    transactions_with_dates = 0
    transactions_with_categories = 0
    
    for email, transactions in data.get('transactions', {}).items():
        if not transactions:
            continue
            
        for transaction in transactions:
            total_transactions += 1
            if all(key in transaction for key in ['description', 'amount']):
                valid_transactions += 1
                
            if 'date' in transaction and transaction['date']:
                transactions_with_dates += 1
                
            if 'category' in transaction and transaction['category']:
                transactions_with_categories += 1
    
    total_goals = 0
    completed_goals = 0
    for email, goals in data.get('goals', {}).items():
        if not goals:
            continue
            
        for goal in goals:
            total_goals += 1
            progress = goal.get('current_progress', 0)
            if goal.get('status') == 'completed' or progress >= 100:
                completed_goals += 1
    
    data_quality_score = (valid_transactions / max(1, total_transactions)) * 100
    goal_completion_rate = (completed_goals / max(1, total_goals)) * 100 if total_goals > 0 else 0
    
    return {
        'database_size_kb': round(db_size / 1024, 1),
        'total_transactions': total_transactions,
        'valid_transactions': valid_transactions,
        'transactions_with_dates': transactions_with_dates,
        'transactions_with_categories': transactions_with_categories,
        'data_quality_score': round(data_quality_score, 1),
        'total_goals': total_goals,
        'completed_goals': completed_goals,
        'goal_completion_rate': round(goal_completion_rate, 1)
    }

def get_user_activity_metrics():
    '''Get user engagement and activity metrics'''
    data = load_data()
    users_data = data.get('users', {})
    transactions_data = data.get('transactions', {})
    goals_data = data.get('goals', {})
    forecasts_data = data.get('forecasts', {})
    
    engaged_users = 0
    highly_engaged_users = 0
    
    for email in users_data:
        tx_count = len(transactions_data.get(email, []))
        goals_count = len(goals_data.get(email, []))
        forecasts_count = len(forecasts_data.get(email, []))
        if tx_count > 0:
            engaged_users += 1
        if tx_count > 5 and (goals_count > 0 or forecasts_count > 0):
            highly_engaged_users += 1
    
    users_with_goals = sum(1 for goals in goals_data.values() if goals and len(goals) > 0)
    users_with_forecasts = sum(1 for forecasts in forecasts_data.values() if forecasts and len(forecasts) > 0)
    
    total_users = len(users_data)
    
    engagement_rate = (engaged_users / max(1, total_users)) * 100
    goal_adoption_rate = (users_with_goals / max(1, total_users)) * 100
    forecast_adoption_rate = (users_with_forecasts / max(1, total_users)) * 100
    
    return {
        'total_users': total_users,
        'engaged_users': engaged_users,
        'highly_engaged_users': highly_engaged_users,
        'users_with_goals': users_with_goals,
        'users_with_forecasts': users_with_forecasts,
        'engagement_rate': round(engagement_rate, 1),
        'goal_adoption_rate': round(goal_adoption_rate, 1),
        'forecast_adoption_rate': round(forecast_adoption_rate, 1)
    }