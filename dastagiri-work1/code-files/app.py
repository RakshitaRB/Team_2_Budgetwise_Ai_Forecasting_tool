import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from utils.database import load_data
import numpy as np
import uuid
import re
from create_admin import create_admin_accounts

from utils.auth import (
    verify_password, get_password_hash, create_access_token, 
    verify_token
)
from utils.database import (
    add_user, verify_user, add_transaction, 
    get_user_transactions, update_user_profile,
    add_financial_goal, get_user_goals, delete_financial_goal,
    get_transactions_time_series, get_category_time_series, get_total_spending_time_series,
    get_category_spending, get_total_savings,
    get_all_users_stats, get_platform_financial_insights, get_forecasting_analytics,
    get_system_health_metrics, get_user_activity_metrics
)
from utils.categorization import categorize_transaction, manual_categorization, CATEGORIES, detect_transaction_type
from utils.forecasting import generate_expense_forecast, generate_income_forecast, can_generate_forecast


def detect_transaction_type_fallback(description):
    """Simple fallback for transaction type detection"""
    if not description:
        return "expense"
    desc_lower = description.lower()
    
    income_indicators = ['salary', 'deposit', 'income', 'credited', 'refund']
    for indicator in income_indicators:
        if indicator in desc_lower:
            return "income"
    return "expense"

def categorize_transaction_fallback(description):
    """Simple fallback for categorization"""
    if not description:
        return "Other"
    desc_lower = description.lower()
    
    if any(word in desc_lower for word in ['grocery', 'food', 'supermarket']):
        return "Groceries"
    elif any(word in desc_lower for word in ['restaurant', 'cafe', 'coffee']):
        return "Dining"
    elif any(word in desc_lower for word in ['salary', 'paycheck']):
        return "Salary"
    else:
        return "Other"



st.set_page_config(
    page_title='Budget Forecasting Tool',
    page_icon='📊',
    layout='wide'
)


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

def main():
    st.title('Budget Forecasting Tool')
    st.markdown('Track your expenses, analyze spending patterns, and forecast your budget with AI-powered insights')
    st.markdown('---')
    
    if not st.session_state.logged_in:
        show_login_register()
    else:
        show_main_application()

def show_login_register():
    """Show login and registration forms"""
    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "👑 Admin Login"])
    
    with tab1:
        st.header("Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if email and password:
                    success, message = verify_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        # Load user role
                        data = load_data()
                        user_data = data['users'].get(email, {})
                        st.session_state.user_role = user_data.get('role', 'user')
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.header("Register")
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password", 
                                   help="Password must be at least 6 characters and less than 50 characters")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if username and email and password and confirm_password:
                    # Password validation
                    if len(password) < 6:
                        st.error("Password must be at least 6 characters long!")
                    elif len(password) > 50:
                        st.error("Password must be less than 50 characters!")
                    elif password != confirm_password:
                        st.error("Passwords don't match!")
                    else:
                        hashed_password = get_password_hash(password)
                        success, message = add_user(username, email, hashed_password, role='user')
                        if success:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields")
    
    with tab3:
        st.header("Admin Login")
        st.info("Administrator access only")
        with st.form("admin_login_form"):
            admin_email = st.text_input("Admin Email")
            admin_password = st.text_input("Admin Password", type="password")
            admin_login_button = st.form_submit_button("Admin Login")
            
            if admin_login_button:
                if admin_email and admin_password:
                    success, message = verify_user(admin_email, admin_password)
                    if success:
                        # Check if user is admin
                        data = load_data()
                        user_data = data['users'].get(admin_email, {})
                        if user_data.get('role') == 'admin':
                            st.session_state.logged_in = True
                            st.session_state.user_email = admin_email
                            st.session_state.user_role = 'admin'
                            st.success("Admin login successful!")
                            st.rerun()
                        else:
                            st.error("Access denied: Administrator privileges required")
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")

def show_main_application():
    """Show the main application after login"""
    st.sidebar.title(f" Dashboard")
    st.sidebar.write(f"👤 Logged in as: {st.session_state.user_email}")
    if st.session_state.user_role == 'admin':
        st.sidebar.write("👑 **Administrator**")
    
    # Navigation - UPDATED FOR ADMIN ROLES
    navigation_options = ["Dashboard", "Add Transaction", "View Transactions", "Budget Reports", "Financial Goals", "Budget Forecast"]

    # Add Admin option only for admin users
    if st.session_state.user_role == 'admin':
        navigation_options.append("👑 Admin Dashboard")

    navigation_options.append("👤 Profile")

    app_page = st.sidebar.selectbox("Navigate", navigation_options)
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.token = None
        st.session_state.user_role = None
        st.rerun()
    
    if app_page == "Dashboard":
        show_dashboard()
    elif app_page == "Add Transaction":
        add_transaction_page()
    elif app_page == "View Transactions":
        view_transactions_page()
    elif app_page == "Budget Reports":
        show_reports()
    elif app_page == "Financial Goals":
        show_financial_goals()
    elif app_page == "Budget Forecast":
        show_budget_forecast()
    elif app_page == "👑 Admin Dashboard":
        show_admin_dashboard()
    elif app_page == "👤 Profile":
        show_profile()


def show_admin_dashboard():
    """Admin Dashboard with comprehensive analytics"""

    if st.session_state.user_role != 'admin':
        st.error("⛔ Access Denied: Administrator privileges required")
        st.info("Please log in with an admin account to access this page.")
        return
    
    st.header("👑 Admin Dashboard")
    st.markdown("---")
    
    
    st.subheader("🔧 Admin Account Management")
    
    with st.expander("🚀 Create Default Admin Accounts", expanded=True):
        st.info("Create the default admin accounts for the system")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Default accounts that will be created:**")
            st.write("👑 **admin@budget.com** / Admin123!")
            st.write("👑 **test@admin.com** / Test123!")
            st.write("👑 **super@admin.com** / Super123!")
        
        with col2:
            if st.button("✨ Create Admin Accounts", type="primary", use_container_width=True):
                with st.spinner("Creating admin accounts..."):
                    try:
                        results = create_admin_accounts()
                        st.success("✅ Admin account creation completed!")
                        
                        
                        for result in results:
                            if "✅" in result:
                                st.success(result)
                            elif "❌" in result:
                                st.error(result)
                            else:
                                st.info(result)
                                
                        
                    except Exception as e:
                        st.error(f"❌ Error creating admin accounts: {str(e)}")
    
    st.markdown("---")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    
    try:
        with st.spinner("Loading admin analytics..."):
            users_stats = get_all_users_stats()
            financial_insights = get_platform_financial_insights()
            forecasting_analytics = get_forecasting_analytics()
            system_health = get_system_health_metrics()
            user_activity = get_user_activity_metrics()
        
        
        st.subheader("📊 Platform Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", users_stats['total_users'])
        
        with col2:
            st.metric("Active Users", users_stats['active_users'])
        
        with col3:
            st.metric("Total Transactions", financial_insights['total_transactions'])
        
        with col4:
            st.metric("Data Quality", f"{system_health['data_quality_score']:.1f}%")
        
        
        st.subheader("👥 User Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### User Registration Trend")
            if users_stats['monthly_registrations']:
                fig, ax = plt.subplots(figsize=(10, 4))
                months = list(users_stats['monthly_registrations'].keys())[-12:]
                counts = [users_stats['monthly_registrations'][m] for m in months]
                
                ax.bar(months, counts, color='skyblue', alpha=0.7)
                ax.set_title('Monthly User Registrations', fontweight='bold')
                ax.set_ylabel('New Users')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No registration data available")
        
        with col2:
            st.markdown("#### User Engagement")
            
            engagement_data = {
                'Metric': ['Total Users', 'Engaged Users', 'Highly Engaged', 'With Goals', 'With Forecasts'],
                'Count': [
                    user_activity['total_users'],
                    user_activity['engaged_users'],
                    user_activity['highly_engaged_users'],
                    user_activity['users_with_goals'],
                    user_activity['users_with_forecasts']
                ],
                'Rate (%)': [
                    100,
                    user_activity['engagement_rate'],
                    (user_activity['highly_engaged_users'] / user_activity['total_users'] * 100) if user_activity['total_users'] > 0 else 0,
                    user_activity['goal_adoption_rate'],
                    user_activity['forecast_adoption_rate']
                ]
            }
            
            engagement_df = pd.DataFrame(engagement_data)
            st.dataframe(engagement_df, hide_index=True, use_container_width=True)
        
        # Section 3: Financial Insights
        st.subheader("💰 Financial Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Popular Categories Chart
            st.markdown("#### Most Popular Categories")
            if financial_insights['popular_categories']:
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = list(financial_insights['popular_categories'].keys())[:8]
                counts = list(financial_insights['popular_categories'].values())[:8]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                bars = ax.barh(categories, counts, color=colors)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                           f'{count}', va='center', fontweight='bold')
                
                ax.set_xlabel('Number of Transactions')
                ax.set_title('Most Used Categories', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No category data available")
        
        with col2:
            # Transaction Statistics
            st.markdown("#### Transaction Overview")
            
            tx_data = {
                'Type': ['Expenses', 'Income', 'Total'],
                'Count': [
                    financial_insights['expense_count'],
                    financial_insights['income_count'],
                    financial_insights['total_transactions']
                ]
            }
            
            tx_df = pd.DataFrame(tx_data)
            st.dataframe(tx_df, hide_index=True, use_container_width=True)
            
            # Average transactions per user
            st.metric("Avg Transactions/User", f"{financial_insights['avg_transactions_per_user']:.1f}")
        
        # Section 4: Forecasting Analytics
        st.subheader("🔮 Forecasting Analytics")
        
        if forecasting_analytics['total_forecasts'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Forecast Usage
                st.markdown("#### Forecast Usage")
                if forecasting_analytics['forecast_types']:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    types = list(forecasting_analytics['forecast_types'].keys())
                    counts = list(forecasting_analytics['forecast_types'].values())
                    
                    colors = ['#ff9999', '#66b3ff', '#99ff99']
                    wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.1f%%', 
                                                    colors=colors[:len(types)])
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('Forecast Type Distribution', fontweight='bold')
                    st.pyplot(fig)
                else:
                    st.info("No forecast type data available")
            
            with col2:
                # Forecast Adoption
                st.markdown("#### Forecast Adoption")
                
                forecast_stats = {
                    'Metric': ['Users with Forecasts', 'Total Forecasts', 'Adoption Rate'],
                    'Value': [
                        forecasting_analytics['users_with_forecasts'],
                        forecasting_analytics['total_forecasts'],
                        f"{forecasting_analytics['forecast_usage_rate']:.1f}%"
                    ]
                }
                
                forecast_df = pd.DataFrame(forecast_stats)
                st.dataframe(forecast_df, hide_index=True, use_container_width=True)
                
                # Popular forecast categories
                if forecasting_analytics['popular_forecast_categories']:
                    st.markdown("**Popular Forecast Categories:**")
                    for category, count in list(forecasting_analytics['popular_forecast_categories'].items())[:5]:
                        st.write(f"- {category}: {count}")
        else:
            st.info("📊 No forecasting data available yet. Users need to generate forecasts first.")
        
        # Section 5: System Health
        st.subheader("⚙️ System Health")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Database Size", f"{system_health['database_size_kb']:.1f} KB")
        
        with col2:
            st.metric("Valid Transactions", f"{system_health['valid_transactions']}")
        
        with col3:
            st.metric("Goals Completion", f"{system_health['goal_completion_rate']:.1f}%")
        
        with col4:
            st.metric("Data with Dates", f"{system_health['transactions_with_dates']}")
        
        
        
        
    except Exception as e:
        st.error(f"❌ Error loading admin dashboard: {str(e)}")
        st.info("This might be because there's no data yet. Try creating some admin accounts and adding transactions first.")
    
    # Real-time data info
    st.markdown("---")
    st.caption(f"📡 Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("💡 All data is aggregated and anonymized for privacy protection")


def debug_forecast_data(email: str):
    """Debug function to check forecast data issues"""
    st.subheader("🔍 Forecast Data Debug")
    
    transactions = get_user_transactions(email)
    
    if not transactions:
        st.error("No transactions found!")
        return
    
    st.write(f"**Total transactions:** {len(transactions)}")
    
    # Show first few transactions with dates
    st.write("**First 5 transactions:**")
    debug_df = pd.DataFrame(transactions[:5])
    st.dataframe(debug_df[['date', 'description', 'amount', 'type', 'category']])
    
    # Check date parsing
    st.write("**Date Analysis:**")
    valid_dates = []
    date_issues = []
    
    for i, t in enumerate(transactions):
        try:
            date_str = t.get('date', '')
            parsed_date = pd.to_datetime(date_str, errors='coerce', format='mixed')
            
            if pd.isna(parsed_date):
                date_issues.append(f"Transaction {i}: Could not parse '{date_str}'")
            else:
                valid_dates.append(parsed_date)
                
        except Exception as e:
            date_issues.append(f"Transaction {i}: Error '{str(e)}'")
    
    st.write(f"**Valid dates found:** {len(valid_dates)}")
    st.write(f"**Date parsing issues:** {len(date_issues)}")
    
    if valid_dates:
        min_date = min(valid_dates)
        max_date = max(valid_dates)
        date_range = max_date - min_date
        months_of_data = date_range.days / 30.0
        
        st.write(f"**Date range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        st.write(f"**Days between:** {date_range.days}")
        st.write(f"**Months of data:** {months_of_data:.2f}")
    
    if date_issues:
        with st.expander("View Date Parsing Issues"):
            for issue in date_issues[:10]:  # Show first 10 issues
                st.write(issue)



def show_financial_goals():
    """Page for managing financial goals - WITH FIXED SAVINGS CALCULATION"""
    st.header(" Financial Goals")
    
    # Add refresh button at the top
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Goals", use_container_width=True):
            st.rerun()
    
    # Goal creation section
    st.subheader("Create New Goal")
    
    with st.form("goal_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            goal_name = st.text_input("Goal Name*", placeholder="e.g., Save for Vacation")
            target_amount = st.number_input("Target Amount *", min_value=0.01, value=1000.0, step=100.0)
            target_date = st.date_input("Target Date*", min_value=datetime.now().date())
        
        with col2:
            initial_savings = st.number_input("Current Savings ", min_value=0.0, value=0.0, step=100.0,
                                            help="How much you've already saved towards this goal")
            goal_description = st.text_area("Description", 
                                          placeholder="Describe your goal...",
                                          height=100)
        
        submitted = st.form_submit_button("Create Goal")
        
        if submitted:
            if not goal_name:
                st.error("Please enter a goal name")
                return
            
            goal_data = {
                "name": goal_name,
                "description": goal_description,
                "goal_type": "savings_goal",
                "target_amount": float(target_amount),
                "current_savings": float(initial_savings), 
                "target_date": target_date.isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            success, message = add_financial_goal(st.session_state.user_email, goal_data)
            if success:
                st.success("✅ Goal created successfully!")
                st.rerun()
            else:
                st.error(f"❌ {message}")
    
    
    st.subheader("Your Goals")
    goals = get_user_goals(st.session_state.user_email)
    
    if not goals:
        st.info("No goals yet. Create your first financial goal above!")
        return
    
    
    total_savings = get_total_savings(st.session_state.user_email)
    st.info(f" **Your Current Total Savings: {total_savings:,.2f}**")
    
    
    for i, goal in enumerate(goals):
        goal_id = goal.get('id')
        
        with st.container():
            st.subheader(f" {goal.get('name', 'Unnamed Goal')}")
            st.write(goal.get('description', 'No description'))
            
            target_amount = goal.get('target_amount', 0)
            target_date_str = goal.get('target_date', 'N/A')
            initial_savings = goal.get('current_savings', 0)
            current_towards_goal = goal.get('current_towards_goal', 0)
            
            days_remaining = 0
            daily_savings_needed = 0
            weekly_savings_needed = 0
            
            if target_date_str != 'N/A':
                try:
                    target_date = datetime.fromisoformat(target_date_str).date()
                    today = datetime.now().date()
                    
                    days_remaining = max(0, (target_date - today).days)
                    
                    remaining_amount = max(0, target_amount - current_towards_goal)
                    
                    if days_remaining > 0:
                        daily_savings_needed = remaining_amount / days_remaining
                        weekly_savings_needed = remaining_amount / (days_remaining / 7)
                    
                    target_date_display = target_date.strftime('%Y-%m-%d')
                except:
                    target_date_display = 'N/A'
            else:
                target_date_display = 'N/A'
            
            # Display goal metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Target Amount", f"{target_amount:,.2f}")
            
            with col2:
                remaining = max(0, target_amount - current_towards_goal)
                st.metric("Remaining to Save", f"{remaining:,.2f}")
            
            with col3:
                st.metric("Target Date", target_date_display)
            
            with col4:
                if days_remaining > 0:
                    st.metric("Days Remaining", f"{days_remaining}")
                else:
                    st.metric("Days Remaining", "0")
            
            
            if target_amount > 0:
                progress_percent = min((current_towards_goal / target_amount) * 100, 100)
                st.progress(progress_percent / 100)
                if progress_percent >= 100:
                    st.success(f"🎉 Goal Achieved! You have saved {current_towards_goal:,.2f} towards this goal")
                
            if days_remaining > 0 and (target_amount - current_towards_goal) > 0:
                st.subheader("Your Savings Plan")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Daily Savings Needed", f"{daily_savings_needed:,.2f}")
                
                with col2:
                    st.metric("Weekly Savings Needed", f"{weekly_savings_needed:,.2f}")
                
            col1, col2, col3 = st.columns([2, 1, 1])
            with col3:
                if st.button("Delete Goal", key=f"delete_{goal_id}", use_container_width=True):
                    success, message = delete_financial_goal(st.session_state.user_email, goal_id)
                    if success:
                        st.success("Goal deleted!")
                        st.rerun()
                    else:
                        st.error(f"Error: {message}")
            
            st.markdown("---")


def create_forecast_chart(historical_data, forecast_data, title="Spending Forecast"):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not historical_data.empty:
        ax.plot(historical_data['ds'], historical_data['y'], 'b-', 
                label='Historical', linewidth=2, marker='o')
    
    if forecast_data is not None and not forecast_data.empty:
        ax.plot(forecast_data['ds'], forecast_data['yhat'], 'r--', 
                label='Forecast', linewidth=2, marker='s')
        
        # Plot confidence interval
        ax.fill_between(forecast_data['ds'], 
                       forecast_data['yhat_lower'], 
                       forecast_data['yhat_upper'], 
                       alpha=0.2, color='red', label='Confidence Interval')
    
    # FIX: Format Y-axis to show full numbers instead of scientific notation
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Amount ', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_goal_progress_chart(goals):
    """Create a progress chart for financial goals"""
    if not goals:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    goal_names = []
    progress_values = []
    
    for goal in goals:
        if goal.get('status') == 'active':
            goal_names.append(goal.get('name', 'Goal')[:20] + '...')
            progress_values.append(goal.get('current_progress', 0))
    
    if not goal_names:
        return None
    
    colors = ['green' if p >= 100 else 'orange' if p >= 50 else 'red' for p in progress_values]
    
    bars = ax.barh(goal_names, progress_values, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, progress_values)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Progress (%)', fontsize=12)
    ax.set_title('Financial Goals Progress', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_income_expense_comparison(income, expenses):
    """Create a modern income vs expenses comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    categories = ['Income', 'Expenses', 'Net']
    values = [income, expenses, income - expenses]  
    colors = ['#2E86AB', '#A23B72', '#F18F01']  
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{value:,.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # FIX: Format Y-axis to show full numbers instead of scientific notation
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    ax.set_title('Income vs Expenses', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Amount ', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Customize the look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_savings_rate_gauge(savings_rate):
    """Create a savings rate gauge chart"""
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(projection='polar'))
    
    # Calculate angles
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    
    # Create the gauge background
    ax.fill_between(theta, 0, 1, color='lightgray', alpha=0.3)
    
    # Fill based on savings rate (scaled to 0-100%)
    fill_theta = np.linspace(0, np.pi * (savings_rate / 100), 50)
    ax.fill_between(fill_theta, 0, 1, color='seagreen', alpha=0.7)
    
    # Customize the gauge
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Add text
    ax.text(0, 0, f'{savings_rate:.1f}%', ha='center', va='center', 
            fontsize=24, fontweight='bold')
    ax.text(0, -1.5, 'Savings Rate', ha='center', va='center', 
            fontsize=12)
    
    return fig

def create_spending_by_category_chart(category_data):
    """Create a vertical bar chart for spending by category"""
    if category_data.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort categories by amount (descending for better visualization)
    category_data = category_data.sort_values(ascending=False)
    
    # Create vertical bar chart
    bars = ax.bar(range(len(category_data)), category_data.values, 
                  color=plt.cm.Set3(np.linspace(0, 1, len(category_data))),
                  alpha=0.7)
    
    # Add value labels on bars
    for i, (category, value) in enumerate(zip(category_data.index, category_data.values)):
        ax.text(i, value + max(category_data.values)*0.01, 
                f'{value:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # FIX: Format Y-axis to show full numbers instead of scientific notation
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    ax.set_xticks(range(len(category_data)))
    ax.set_xticklabels(category_data.index, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Amount', fontsize=11)
    ax.set_title('Spending by Category', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_simple_pie_chart(data, title, colors):
    """Create a simple pie chart with percentages and category labels outside"""
    if data.empty or data.sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart with labels outside
    wedges, texts, autotexts = ax.pie(
        data.values,
        labels=data.index,  # Show category names as labels
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10},
        pctdistance=0.85,  # Move percentage text inside
        labeldistance=1.05  # Move labels outside
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # Style the labels
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    plt.tight_layout()
    return fig

def show_dashboard():
    """Show dashboard with integrated financial charts"""
    st.header("📊 Financial Dashboard")
    
    # Add refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    
    transactions = get_user_transactions(st.session_state.user_email)
    
    if not transactions:
        st.info("No transactions yet. Add some transactions to see your dashboard!")
        return
    
    df = pd.DataFrame(transactions)
    
    # FIXED DATE PARSING: Use flexible approach
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    except Exception as e:
        st.error(f"Date parsing error: {e}")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove invalid dates
    df = df.dropna(subset=['date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Remove invalid amounts
    df = df.dropna(subset=['amount'])
    
    # Create display dataframe with absolute values for visualization
    display_df = df.copy()
    display_df['display_amount'] = display_df['amount'].abs()
    
    # Calculate key metrics
    total_income = df[df['type'] == 'income']['amount'].sum()
    total_expenses = df[df['type'] == 'expense']['amount'].sum()
    actual_expenses = abs(total_expenses)
    balance = total_income + total_expenses
    savings_rate = ((total_income - actual_expenses) / total_income * 100) if total_income > 0 else 0
    
    # Financial Overview Section
    st.subheader("💰 Financial Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Income", f"{total_income:,.2f}")
    
    with col2:
        st.metric("Total Expenses", f"{actual_expenses:,.2f}")
    
    with col3:
        st.metric("Balance", f"{balance:,.2f}")
    
    with col4:
        st.metric("Savings Rate", f"{savings_rate:.1f}%")
    
    # Income vs Expenses Chart
    st.subheader("📈 Income vs Expenses")
    income_expense_fig = create_income_expense_comparison(total_income, actual_expenses)
    st.pyplot(income_expense_fig)
    
    # MILESTONE 3: Goals Progress Section
    goals = get_user_goals(st.session_state.user_email)
    active_goals = [goal for goal in goals if goal.get('status') == 'active']
    
    if active_goals:
        st.subheader("Active Goals Progress")
        goal_chart = create_goal_progress_chart(active_goals)
        if goal_chart:
            st.pyplot(goal_chart)
        
        # Show goals summary
        col1, col2, col3 = st.columns(3)
        with col1:
            completed_goals = len([g for g in active_goals if g.get('current_progress', 0) >= 100])
            st.metric("Completed Goals", completed_goals)
        with col2:
            in_progress_goals = len([g for g in active_goals if 0 < g.get('current_progress', 0) < 100])
            st.metric("In Progress", in_progress_goals)
        with col3:
            st.metric("Total Active Goals", len(active_goals))
    
    # This Month Section
    st.subheader("📅 This Month")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Current month calculations
        current_month = datetime.now().strftime('%Y-%m')
        month_income_df = df[
            (df['type'] == 'income') & 
            (df['date'].dt.strftime('%Y-%m') == current_month)
        ]
        month_income = month_income_df['amount'].sum() if not month_income_df.empty else 0
        st.metric("Month Income", f"{month_income:,.2f}")
    
    with col2:
        month_expense_df = df[
            (df['type'] == 'expense') & 
            (df['date'].dt.strftime('%Y-%m') == current_month)
        ]
        month_expenses = abs(month_expense_df['amount'].sum()) if not month_expense_df.empty else 0
        st.metric("Month Expenses", f"{month_expenses:,.2f}")
    
    with col3:
        month_balance = month_income - month_expenses
        st.metric("Month Balance", f"{month_balance:,.2f}",
                 delta=f"+{month_balance:,.2f}" if month_balance > 0 else f"-{abs(month_balance):,.2f}",
                 delta_color="normal" if month_balance > 0 else "inverse")
    
    with col4:
        month_savings_rate = ((month_income - month_expenses) / month_income * 100) if month_income > 0 else 0
        st.metric("Month Savings Rate", f"{month_savings_rate:.1f}%")
    
    # Spending by Category
    st.subheader("🛍️ Spending by Category")
    
    expense_df = display_df[display_df['type'] == 'expense']
    if not expense_df.empty:
        category_totals = expense_df.groupby('category')['display_amount'].sum()
        category_chart = create_spending_by_category_chart(category_totals)
        if category_chart:
            st.pyplot(category_chart)
    else:
        st.info("No expenses to display")


def show_budget_forecast():
    """Page for budget forecasting and projections"""
    st.header(" Budget Forecast")

    can_forecast, message = can_generate_forecast(st.session_state.user_email)
    
    if not can_forecast:
        st.warning(f"📊 Forecasting Requirements: {message}")
        st.info("""
        **To generate reliable forecasts, you need:**
        - At least 10 transactions
        - At least 2 months of historical data  
        - Multiple data points for trend analysis
        """)
        
        transactions = get_user_transactions(st.session_state.user_email)
        if transactions:
            st.write(f"**Your current data:** {len(transactions)} transactions")
            
            try:
                df = pd.DataFrame(transactions)
                df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
                df = df.dropna(subset=['date'])
                
                if len(df) > 0:
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    date_range = max_date - min_date
                    months = date_range.days / 30.0
                    
                    st.write(f"**Date range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({months:.1f} months)")
            except:
                st.write("**Date range:** Could not calculate")
        
        return
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_type = st.selectbox(
            "Forecast Type",
            ["Total Expenses", "Total Income"]
        )
    
    
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            if forecast_type == "Total Expenses":
                forecast_result, error = generate_expense_forecast(st.session_state.user_email)
                title = "Total Expenses Forecast"
            elif forecast_type == "Total Income":
                forecast_result, error = generate_income_forecast(st.session_state.user_email)
                title = "Total Income Forecast"
            else:  
                forecast_result, error = generate_expense_forecast(st.session_state.user_email, selected_category)
                title = f"{selected_category} Expenses Forecast"
            
            if error:
                st.error(f"Forecasting error: {error}")
            else:
                st.success("Forecast generated successfully!")
                
                
                historical_data = forecast_result['historical']
                forecast_data = forecast_result['forecast']
                
                
                forecast_chart = create_forecast_chart(historical_data, forecast_data, title)
                st.pyplot(forecast_chart)
                
                
                st.subheader("Forecast Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    last_historical = historical_data['y'].iloc[-1] if not historical_data.empty else 0
                    st.metric("Last Historical Value", f"{last_historical:,.2f}")
                
                with col2:
                    avg_forecast = forecast_data['yhat'].mean()
                    st.metric("Average Forecast", f"{avg_forecast:,.2f}")
                
                with col3:
                    trend = "Increasing" if forecast_result.get('trend_slope', 0) > 0 else "Decreasing"
                    st.metric("Trend", trend)
                
                
                st.subheader("Detailed Forecast")
                display_forecast = forecast_data.copy()
                display_forecast['ds'] = display_forecast['ds'].dt.strftime('%Y-%m-%d')
                display_forecast = display_forecast.rename(columns={
                    'ds': 'Date',
                    'yhat': 'Forecast',
                    'yhat_lower': 'Lower Bound',
                    'yhat_upper': 'Upper Bound'
                })
                st.dataframe(display_forecast)
    
    
    st.subheader("Historical Data Preview")
    ts_data = get_transactions_time_series(st.session_state.user_email, 'M')
    
    if not ts_data.empty:
        historical_chart_data = ts_data[ts_data['type'] == 'expense'].groupby('date')['amount_positive'].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(historical_chart_data['date'], historical_chart_data['amount_positive'], 'b-o', linewidth=2)
        ax.set_title('Historical Spending Trend', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amount ')
        
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(f"**Data Summary:** {len(historical_chart_data)} months of historical data available")
    else:
        st.info("No historical data available for forecasting")


def debug_date_parsing():
    """Debug function to check date parsing issues"""
    st.header("🔍 Date Parsing Debug")
    
    transactions = get_user_transactions(st.session_state.user_email)
    if not transactions:
        st.error("No transactions found!")
        return
    
    df = pd.DataFrame(transactions)
    
    st.subheader("Raw Date Samples")
    st.write("First 10 raw dates from database:")
    st.write(df['date'].head(10))
    
    st.subheader("Date Parsing Tests")
    
    try:
        df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
        valid_dates = df[df['parsed_date'].notna()]
        st.write(f"✅ Pandas parsing: {len(valid_dates)}/{len(df)} dates parsed successfully")
        st.write(f"Date range: {valid_dates['parsed_date'].min()} to {valid_dates['parsed_date'].max()}")
    except Exception as e:
        st.error(f"❌ Pandas parsing failed: {e}")
    
    st.subheader("Manual Date Inspection")
    date_samples = df['date'].head(20).tolist()
    for i, date_str in enumerate(date_samples):
        st.write(f"Row {i}: '{date_str}'")
    
    st.subheader("Common Date Issues")
    problematic_dates = []
    for date_str in df['date'].unique()[:50]:  
        if not isinstance(date_str, str):
            problematic_dates.append(f"Non-string date: {date_str}")
            continue
        
        
        if len(str(date_str).strip()) < 8:  
            problematic_dates.append(f"Too short: '{date_str}'")
        elif '/' in str(date_str) and len(str(date_str).split('/')) != 3:
            problematic_dates.append(f"Invalid slash format: '{date_str}'")
        elif '-' in str(date_str) and len(str(date_str).split('-')) != 3:
            problematic_dates.append(f"Invalid dash format: '{date_str}'")
    
    if problematic_dates:
        st.error("Found problematic dates:")
        for issue in problematic_dates[:10]:  
            st.write(f"  - {issue}")


def process_uploaded_csv(uploaded_file):
    """Process uploaded CSV file and return transactions with AUTO-CORRECTED categories and types"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Show preview
        st.subheader("CSV Preview (Original)")
        st.dataframe(df.head(10))
        
        # Validate required columns
        required_columns = ['description', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("Required columns: 'description', 'amount'")
            st.info("Optional columns: 'category', 'date', 'notes', 'type'")
            return None
        
        st.subheader("🔄 Auto-Correcting Categories and Types...")
        
        
        original_df = df.copy()
        
        
        corrections_made = 0
        for index, row in df.iterrows():
            description = str(row['description']).strip()
            if not description:
                continue
            
            
            original_type = None
            if 'type' in df.columns and not pd.isna(row['type']):
                original_type = str(row['type']).strip().lower()
            
            corrected_type = detect_transaction_type(description)
            

            if original_type != corrected_type:
                df.at[index, 'type'] = corrected_type
                corrections_made += 1
                st.write(f"✅ Row {index + 1}: Type corrected '{original_type}' → '{corrected_type}'")
            
            
            original_category = None
            if 'category' in df.columns and not pd.isna(row['category']):
                original_category = str(row['category']).strip()
            
            corrected_category = categorize_transaction(description)
            
            
            if original_category != corrected_category:
                df.at[index, 'category'] = corrected_category
                corrections_made += 1
                st.write(f"✅ Row {index + 1}: Category corrected '{original_category}' → '{corrected_category}'")
        
        
        st.success(f" Auto-corrected {corrections_made} entries based on descriptions!")
        
        
        if corrections_made > 0:
            st.subheader("📊 Correction Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Correction**")
                display_original = original_df[['description', 'type', 'category']].head(10).copy()
                st.dataframe(display_original)
            
            with col2:
                st.write("**After Correction**")    
                display_corrected = df[['description', 'type', 'category']].head(10).copy()
                st.dataframe(display_corrected)
        
        
        transactions = []
        success_count = 0
        error_count = 0
        
        st.subheader("🔄 Creating Transactions...")
        
        for index, row in df.iterrows():
            try:
                description = str(row['description']).strip()
                if not description:
                    st.warning(f"Row {index + 1}: Empty description, skipping")
                    error_count += 1
                    continue
                
                
                try:
                    raw_amount = float(row['amount'])
                except (ValueError, TypeError):
                    st.warning(f"Row {index + 1}: Invalid amount '{row['amount']}', skipping")
                    error_count += 1
                    continue
                
                
                transaction_type = str(row['type']).strip().lower() if 'type' in df.columns and not pd.isna(row['type']) else 'expense'
                
                
                if transaction_type not in ['income', 'expense']:
                    # Final fallback detection
                    transaction_type = detect_transaction_type(description)
                
                
                category = str(row['category']).strip() if 'category' in df.columns and not pd.isna(row['category']) else 'Other'
                
                
                transaction_date = None
                if 'date' in df.columns and not pd.isna(row['date']):
                    date_str = str(row['date']).strip()
                    
                    date_formats = [
                        '%Y-%m-%d',           
                        '%m/%d/%Y',           
                        '%d/%m/%Y',
                        '%Y-%m-%d %H:%M:%S',  
                        '%m/%d/%y',
                        '%d-%m-%Y',
                        '%Y/%m/%d',
                        '%d-%b-%Y',
                        '%d %b %Y',
                        '%b %d, %Y',
                    ]
                    
                    parsed_successfully = False
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, date_format)
                            transaction_date = parsed_date.strftime('%Y-%m-%d')
                            parsed_successfully = True
                            break
                        except ValueError:
                            continue
                    
                    
                    if not parsed_successfully:
                        try:
                            parsed_date = pd.to_datetime(date_str)
                            if not pd.isna(parsed_date):
                                transaction_date = parsed_date.strftime('%Y-%m-%d')
                                parsed_successfully = True
                        except:
                            pass
                    if not parsed_successfully:
                        transaction_date = datetime.now().strftime('%Y-%m-%d')
                        st.warning(f"Row {index + 1}: Using today's date for unparsable date '{date_str}'")
                else:
                    transaction_date = datetime.now().strftime('%Y-%m-%d')
                
               
                if transaction_type == 'expense':
                    amount = -abs(raw_amount)  
                else:
                    amount = abs(raw_amount)
                
                notes = ""
                if 'notes' in df.columns and not pd.isna(row.get('notes')):
                    notes = str(row['notes']).strip()
                
                transaction_data = {
                    "description": description,
                    "amount": amount,  
                    "type": transaction_type,
                    "category": category,
                    "notes": notes,
                    "date": transaction_date,
                    "id": str(uuid.uuid4())[:8]  
                }
                
                transactions.append(transaction_data)
                success_count += 1
                
                
                st.write(f"✅ Row {index + 1}: {transaction_date} - {transaction_type} - {category} - {abs(raw_amount):.2f}")
                
            except Exception as e:
                st.error(f"❌ Row {index + 1}: Unexpected error - {str(e)}")
                error_count += 1
                continue
        
        st.subheader("📊 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Successfully Processed", success_count)
        with col3:
            st.metric("Errors", error_count)
        with col4:
            st.metric("Corrections Made", corrections_made)
        
        if transactions:
            st.subheader("✅ Final Processed Transactions")
            final_df = pd.DataFrame(transactions)
            
            
            display_df = final_df.copy()
            display_df['display_amount'] = display_df['amount'].abs()
            
            st.dataframe(display_df[['date', 'description', 'type', 'category', 'display_amount']])
            

            st.subheader("📈 Transaction Breakdown (After Correction)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                income_df = display_df[display_df['type'] == 'income']
                if not income_df.empty:
                    income_by_category = income_df.groupby('category')['display_amount'].sum()
                    st.write("**Income by Category:**")
                    for category, amount in income_by_category.items():
                        st.write(f"  - {category}: {amount:,.2f}")
                else:
                    st.write("**Income by Category:** No income transactions")
            
            with col2:
                expense_df = display_df[display_df['type'] == 'expense']
                if not expense_df.empty:
                    expense_by_category = expense_df.groupby('category')['display_amount'].sum()
                    st.write("**Expenses by Category:**")
                    for category, amount in expense_by_category.items():
                        st.write(f"  - {category}: {amount:,.2f}")
                else:
                    st.write("**Expenses by Category:** No expense transactions")
            
            if 'date' in final_df.columns:
                dates = pd.to_datetime(final_df['date'])
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                st.info(f"📅 Date Range: {date_range}")
            
            income_count = sum(1 for t in transactions if t['type'] == 'income')
            expense_count = sum(1 for t in transactions if t['type'] == 'expense')
            total_income = sum(t['amount'] for t in transactions if t['type'] == 'income')
            total_expenses = sum(abs(t['amount']) for t in transactions if t['type'] == 'expense')
            
            st.success(f"✅ Ready to import {success_count} transactions!")
            st.info(f"📈 Breakdown: {income_count} income ({total_income:.2f}), {expense_count} expenses ({total_expenses:.2f})")
        
        return transactions, success_count, error_count
        
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        return None, 0, 0

def add_transaction_page():
    """Page to add new transactions with CSV upload option"""
    st.header("Add Transactions")
    
    input_method = st.radio(
        "Choose how to add transactions:",
        ["Add Transaction", " Upload a CSV File"],
        horizontal=True
    )
    
    if input_method == "Add Transaction":
        add_single_transaction()
    else:
        upload_csv_transactions()

def add_single_transaction():
    """Form for adding single transaction"""
    if 'auto_type' not in st.session_state:
        st.session_state.auto_type = "expense"
    if 'auto_category' not in st.session_state:
        st.session_state.auto_category = "Other"
    if 'last_description' not in st.session_state:
        st.session_state.last_description = ""
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    with st.form("transaction_form"):
        st.subheader("Add New Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            description = st.text_input("Description*", placeholder="e.g., Walmart grocery shopping", 
                                      key="description_input")
            
            
            if (description and description != st.session_state.last_description and 
                not st.session_state.form_submitted):
                st.session_state.last_description = description
                try:
                    st.session_state.auto_type = detect_transaction_type(description)
                    st.session_state.auto_category = categorize_transaction(description)
                except Exception as e:
                    st.warning(f"Auto-detection warning: {e}")
                    st.session_state.auto_type = detect_transaction_type_fallback(description)
                    st.session_state.auto_category = categorize_transaction_fallback(description)
            
            amount = st.number_input("Amount*", min_value=0.01, step=0.01, format="%.2f", key="amount_input")
        
        with col2:
            st.markdown("### Auto-detected")
            type_color = "green" if st.session_state.auto_type == "income" else "red"
            st.markdown(f"**Type:** :{type_color}[{st.session_state.auto_type.title()}]")
            
            category_emojis = {
                'Salary': '💰', 'Freelance': '💼', 'Business': '🏢', 'Investments': '📈',
                'Rental': '🏠', 'Groceries': '🛒', 'Dining': '🍽️', 'Transport': '🚗', 
                'Entertainment': '🎬', 'Shopping': '🛍️', 'Bills': '📄', 'Healthcare': '🏥', 
                'Other': '📦'
            }
            emoji = category_emojis.get(st.session_state.auto_category, '📦')
            st.markdown(f"**Category:** {emoji} **{st.session_state.auto_category}**")
            
            st.markdown("---")
            
        
        notes = st.text_area("Notes (optional)", key="notes_input",
                           placeholder="Add any additional notes about this transaction...")
        
        # SUBMIT BUTTON ALIGNED TO LEFT
        submitted = st.form_submit_button("Add Transaction")
        
        if submitted:
            st.session_state.form_submitted = True
            
            # Validation
            if not description or not description.strip():
                st.error("❌ Please enter a description")
                st.session_state.form_submitted = False
                return
                
            if amount <= 0:
                st.error("❌ Please enter a valid amount greater than 0")
                st.session_state.form_submitted = False
                return
            
            # AUTO-USE THE DETECTED VALUES (no manual selection)
            final_type = st.session_state.auto_type
            final_category = st.session_state.auto_category
            
            transaction_data = {
                "description": description.strip(),
                "amount": amount if final_type == 'income' else -amount,
                "type": final_type,
                "category": final_category,
                "notes": notes.strip(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "id": str(uuid.uuid4())[:8]
            }
            
            success, message = add_transaction(st.session_state.user_email, transaction_data)
            if success:
                st.success("✅ Transaction added successfully!")
                st.info(f"**Details:** {final_type.title()} - {final_category} - {amount:.2f}")
                
                # Store the success message in session state to persist after rerun
                st.session_state.success_message = f"✅ Transaction added: {final_type.title()} - {final_category} - {amount:.2f}"
                
                # Clear the form by resetting session state
                st.session_state.last_description = ""
                st.session_state.auto_type = "expense"
                st.session_state.auto_category = "Other"
                st.session_state.form_submitted = False
                
                # Force a rerun to clear the form
                st.rerun()
                    
            else:
                st.error(f"❌ {message}")
                st.session_state.form_submitted = False
    
    # Show success message after form reset (outside the form)
    if st.session_state.get('success_message'):
        st.success(st.session_state.success_message)
        # Clear the success message after showing it
        st.session_state.success_message = None

def upload_csv_transactions():
    """Upload and process CSV file with transactions"""
    st.subheader("📤 Upload CSV File")
    
    st.info("""
    **🤖 Smart CSV Processing:**
    - **Required columns**: `description`, `amount`
    - **Optional columns**: `category`, `date`, `notes`, `type`
    - **AI will automatically detect**: Transaction type (Income/Expense) and Category
    - **Amount handling**: Positive for income, negative for expenses (automatically handled)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing your CSV file..."):
            result = process_uploaded_csv(uploaded_file)
        
        if result is not None:
            transactions, success_count, error_count = result
            
            if transactions and st.button("🚀 Import All Transactions", type="primary"):
                import_progress = st.progress(0)
                status_text = st.empty()
                
                total_transactions = len(transactions)
                imported_count = 0
                failed_count = 0
                failed_transactions = []
                
                for i, transaction in enumerate(transactions):
                    status_text.text(f"Importing transaction {i+1} of {total_transactions}...")
                    import_progress.progress((i + 1) / total_transactions)
                    
                    success, message = add_transaction(st.session_state.user_email, transaction)
                    if success:
                        imported_count += 1
                    else:
                        failed_count += 1
                        failed_transactions.append((transaction['description'], message))
                
                status_text.text("")
                import_progress.empty()
                
                # Show final results
                if imported_count > 0:
                    st.balloons()
                    st.success(f"🎉 Successfully imported {imported_count} transactions!")
                    
                    # Show quick summary
                    income_transactions = [t for t in transactions if t['type'] == 'income']
                    expense_transactions = [t for t in transactions if t['type'] == 'expense']
                    
                    total_income = sum(t['amount'] for t in income_transactions)
                    total_expenses = sum(abs(t['amount']) for t in expense_transactions)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Income Transactions", len(income_transactions))
                    with col2:
                        st.metric("Expense Transactions", len(expense_transactions))
                    with col3:
                        st.metric("Total Income", f"{total_income:,.2f}")
                    with col4:
                        st.metric("Total Expenses", f"{total_expenses:,.2f}")
                    
                    # Navigation suggestion
                    st.info("💡 **Check your Dashboard to see the updated financial overview!**")
                
                if failed_count > 0:
                    st.error(f"❌ {failed_count} transactions failed to import")
                    with st.expander("View failed transactions"):
                        for desc, error in failed_transactions:
                            st.write(f"- {desc}: {error}")

def view_transactions_page():
    """Page to view and manage transactions"""
    st.header("Your Transactions")
    
    transactions = get_user_transactions(st.session_state.user_email)
    
    if not transactions:
        st.info("No transactions found.")
        return
    
    # Convert to DataFrame for display
    df = pd.DataFrame(transactions)
    
    # FIXED DATE PARSING: Use flexible date parsing
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    except Exception as e:
        st.error(f"Date parsing error: {e}")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove any invalid dates
    df = df.dropna(subset=['date'])
    
    # Create display version with absolute values and better formatting
    display_df = df.copy()
    display_df['display_amount'] = display_df['amount'].abs()
    
    # Add emojis for categories
    category_emojis = {
        'Salary': '💰', 'Freelance': '💼', 'Business': '🏢', 'Investments': '📈', 'Rental': '🏠',
        'Groceries': '🛒', 'Dining': '🍽️', 'Transport': '🚗', 'Entertainment': '🎬',
        'Shopping': '🛍️', 'Bills': '📄', 'Healthcare': '🏥', 'Other': '📦'
    }
    
    display_df['category_with_emoji'] = display_df['category'].apply(
        lambda x: f"{category_emojis.get(x, '📦')} {x}"
    )
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.selectbox("Filter by Date", ["All", "Last 7 days", "Last 30 days", "Last 90 days"])
    
    with col2:
        type_filter = st.selectbox("Filter by Type", ["All", "income", "expense"])
    
    with col3:
        category_filter = st.selectbox("Filter by Category", ["All"] + list(display_df['category'].unique()))
    
    # Apply filters
    filtered_df = display_df.copy()
    
    if date_filter != "All":
        if date_filter == "Last 7 days":
            cutoff_date = datetime.now() - timedelta(days=7)
        elif date_filter == "Last 30 days":
            cutoff_date = datetime.now() - timedelta(days=30)
        else:  # Last 90 days
            cutoff_date = datetime.now() - timedelta(days=90)
        filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
    
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df['type'] == type_filter]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    # Display transactions with better formatting
    st.subheader(f"Transactions ({len(filtered_df)} found)")
    
    # Create a styled dataframe
    display_columns = ['date', 'description', 'display_amount', 'type', 'category_with_emoji', 'notes']
    styled_df = filtered_df[display_columns].rename(columns={
        'display_amount': 'amount',
        'category_with_emoji': 'category'
    })
    
    st.dataframe(styled_df, width='stretch')
    
    # Summary statistics
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_income = filtered_df[filtered_df['type'] == 'income']['display_amount'].sum()
        st.metric("Total Income", f"{total_income:.2f}")
    
    with col2:
        total_expenses = filtered_df[filtered_df['type'] == 'expense']['display_amount'].sum()
        st.metric("Total Expenses", f"{total_expenses:.2f}")
    
    with col3:
        balance = total_income - total_expenses
        st.metric("Balance", f"{balance:.2f}")
    
    # Manual categorization section
    st.subheader("🔄 Manual Category Update")
    
    if transactions:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transaction_options = [f"{t['id']}: {t['description']} ({t['category']})" for t in transactions]
            selected_transaction = st.selectbox("Select Transaction", transaction_options)
        
        with col2:
            new_category = st.selectbox("New Category", list(CATEGORIES.keys()))
        
        with col3:
            if st.button("🔄 Update Category"):
                if selected_transaction:
                    # Extract the transaction ID from the selected option
                    selected_id = selected_transaction.split(":")[0].strip()
                    
                    success, message = manual_categorization(selected_id, new_category, st.session_state.user_email)
                    if success:
                        st.success("✅ Category updated successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Update failed: {message}")
    else:
        st.info("No transactions available for manual categorization")

def show_reports():
    """Generate and display financial reports"""
    st.header("Budget Reports")
    
    transactions = get_user_transactions(st.session_state.user_email)
    
    if not transactions:
        st.info("No transactions available for reports.")
        return
    
    df = pd.DataFrame(transactions)
    
    # FIXED DATE PARSING: Use more flexible date parsing
    try:
        # First try to parse dates with flexible approach
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
        
        # Check if any dates failed to parse
        if df['date'].isna().any():
            st.warning("Some dates could not be parsed properly. Using fallback parsing.")
            # Try alternative parsing method
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
    except Exception as e:
        st.error(f"Date parsing error: {e}")
        # Final fallback - use string dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove any rows with invalid dates
    original_count = len(df)
    df = df.dropna(subset=['date'])
    if len(df) < original_count:
        st.warning(f"Removed {original_count - len(df)} transactions with invalid dates")
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Remove invalid amounts
    df = df.dropna(subset=['amount'])
    
    # Create display dataframe with absolute values
    display_df = df.copy()
    display_df['display_amount'] = display_df['amount'].abs()
    
    # Income vs Expense Report
    st.subheader("Income vs Expenses Analysis")
    
    # Calculate metrics
    total_income = display_df[display_df['type'] == 'income']['display_amount'].sum()
    total_expenses = display_df[display_df['type'] == 'expense']['display_amount'].sum()
    balance = total_income - total_expenses
    
    # Use the new chart function
    comparison_fig = create_income_expense_comparison(total_income, total_expenses)
    st.pyplot(comparison_fig)
    
    # Category analysis - TWO SEPARATE PIE CHARTS
    st.subheader("Category Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # EXPENSE CATEGORIES PIE CHART
        st.markdown("### Expense Categories")
        expense_df = display_df[display_df['type'] == 'expense']
        if not expense_df.empty:
            expense_by_category = expense_df.groupby('category')['display_amount'].sum()
            
            # Create simple pie chart for expense categories
            if not expense_by_category.empty and expense_by_category.sum() > 0:
                # Define valid matplotlib colors for expense categories
                expense_colors = {
                    'Groceries': '#2E8B57',        # Sea Green
                    'Dining': '#FF6B6B',           # Coral Red
                    'Transport': '#4682B4',        # Steel Blue
                    'Entertainment': '#FFD700',    # Gold
                    'Shopping': '#FF69B4',         # Hot Pink
                    'Bills': '#9370DB',            # Medium Purple
                    'Healthcare': '#20B2AA',       # Light Sea Green
                    'Other': '#A9A9A9'             # Dark Gray
                }
                
                # Get colors for each category, default to gray if not found
                colors = [expense_colors.get(cat, '#A9A9A9') for cat in expense_by_category.index]
                
                # Create simple pie chart
                expense_pie = create_simple_pie_chart(
                    expense_by_category, 
                    'Where Your Money Goes', 
                    colors
                )
                st.pyplot(expense_pie)
            else:
                st.info("No expense data available")
        else:
            st.info("No expenses to display")
    
    with col2:
        # INCOME CATEGORIES PIE CHART
        st.markdown("### Income Sources")
        income_df = display_df[display_df['type'] == 'income']
        if not income_df.empty:
            income_by_category = income_df.groupby('category')['display_amount'].sum()
            
            # Create simple pie chart for income sources
            if not income_by_category.empty and income_by_category.sum() > 0:
                # Define valid matplotlib colors for income categories
                income_colors = {
                    'Salary': '#228B22',           # Forest Green
                    'Freelance': '#32CD32',        # Lime Green
                    'Business': '#00FF7F',         # Spring Green
                    'Investments': '#006400',      # Dark Green
                    'Rental': '#98FB98',           # Pale Green
                    'Other': '#90EE90'             # Light Green
                }
                
                # Get colors for each category
                colors = [income_colors.get(cat, '#90EE90') for cat in income_by_category.index]
                
                income_pie = create_simple_pie_chart(
                    income_by_category, 
                    'Where Your Money Comes From', 
                    colors
                )
                st.pyplot(income_pie)
            else:
                st.info("No income data available")
        else:
            st.info("No income to display")
    
    st.subheader("Download Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📊 Download Transactions as CSV",
        data=csv,
        file_name="budget_transactions.csv",
        mime="text/csv"
    )

def show_profile():
    """User profile management"""
    st.header("👤 User Profile")
    st.info(f"Logged in as: {st.session_state.user_email}")
    
    # Password change functionality
    st.subheader("🔒 Change Password")
    
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        change_button = st.form_submit_button("Change Password")
        
        if change_button:
            if current_password and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("New passwords don't match!")
                else:
                    success, message = verify_user(st.session_state.user_email, current_password)
                    if success:
                        st.success("Password changed successfully!")
                    else:
                        st.error("Current password is incorrect!")
            else:
                st.error("Please fill in all fields")

if __name__ == '__main__':
    main()