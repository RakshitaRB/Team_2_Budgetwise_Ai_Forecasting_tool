# admin_dashboard.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
import os
import uuid
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdminDashboard:
    def __init__(self, data_file='data/users.json'):
        self.data_file = data_file
        self.admin_email = "admin@forecastapp.com"
        
    def load_data(self) -> Dict:
        """Load all data from JSON file"""
        if not os.path.exists('data'):
            os.makedirs('data')
        
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump({'users': {}, 'transactions': {}, 'goals': {}, 'forecasts': {}}, f)
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            # Ensure all required keys exist
            required_keys = ['users', 'transactions', 'goals', 'forecasts']
            for key in required_keys:
                if key not in data:
                    data[key] = {}
                    
            return data
        except (json.JSONDecodeError, FileNotFoundError):
            return {'users': {}, 'transactions': {}, 'goals': {}, 'forecasts': {}}

    def save_data(self, data: Dict):
        """Save all data to JSON file"""
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=4)

    def is_admin(self, email: str) -> bool:
        """Check if user is admin"""
        data = self.load_data()
        
        if email not in data['users']:
            return False
        
        user = data['users'][email]
        return user.get('role') == 'admin'

    def create_admin_user(self) -> Tuple[bool, str]:
        """Create initial admin user if none exists"""
        data = self.load_data()
        
        # Check if any admin exists
        admin_exists = any(user_data.get('role') == 'admin' for user_data in data['users'].values())
        
        if not admin_exists:
            from utils.auth import hash_password  # Import from your auth module
            
            admin_password = "admin123"  # Change this in production!
            
            data['users'][self.admin_email] = {
                'username': "System Admin",
                'email': self.admin_email,
                'hashed_password': hash_password(admin_password),
                'role': 'admin',
                'created_at': datetime.now().isoformat()
            }
            
            # Initialize admin data structures
            data['transactions'][self.admin_email] = []
            data['goals'][self.admin_email] = []
            data['forecasts'][self.admin_email] = []
            
            self.save_data(data)
            
            logger.info("Admin user created successfully")
            logger.info(f"Admin email: {self.admin_email}")
            logger.info(f"Admin password: {admin_password}")
            return True, "Admin user created"
        else:
            logger.info("Admin user already exists")
            return True, "Admin user already exists"

    def get_platform_statistics(self) -> Dict:
        """Get comprehensive platform statistics"""
        data = self.load_data()
        
        stats = {
            'total_users': len(data['users']),
            'total_transactions': sum(len(transactions) for transactions in data['transactions'].values()),
            'total_goals': sum(len(goals) for goals in data['goals'].values()),
            'total_forecasts': sum(len(forecasts) for forecasts in data['forecasts'].values()),
            'users': [],
            'recent_registrations': [],
            'admin_count': 0,
            'regular_user_count': 0
        }
        
        # User details for admin view
        for email, user_data in data['users'].items():
            user_stats = {
                'email': email,
                'username': user_data.get('username', 'N/A'),
                'role': user_data.get('role', 'user'),
                'created_at': user_data.get('created_at', ''),
                'transaction_count': len(data['transactions'].get(email, [])),
                'goal_count': len(data['goals'].get(email, [])),
                'forecast_count': len(data['forecasts'].get(email, [])),
                'last_active': self.get_last_active_date(email, data)
            }
            stats['users'].append(user_stats)
            
            # Count admin vs regular users
            if user_data.get('role') == 'admin':
                stats['admin_count'] += 1
            else:
                stats['regular_user_count'] += 1
        
        # Sort users by registration date
        stats['users'].sort(key=lambda x: x['created_at'], reverse=True)
        stats['recent_registrations'] = stats['users'][:5]
        
        return stats

    def get_last_active_date(self, email: str, data: Dict) -> str:
        """Get the last active date for a user"""
        last_activity = None
        
        # Check transactions
        transactions = data['transactions'].get(email, [])
        for transaction in transactions:
            date_str = transaction.get('date', '')
            if date_str:
                try:
                    transaction_date = datetime.fromisoformat(date_str)
                    if not last_activity or transaction_date > last_activity:
                        last_activity = transaction_date
                except:
                    continue
        
        # Check goals
        goals = data['goals'].get(email, [])
        for goal in goals:
            created_at = goal.get('created_at', '')
            if created_at:
                try:
                    goal_date = datetime.fromisoformat(created_at)
                    if not last_activity or goal_date > last_activity:
                        last_activity = goal_date
                except:
                    continue
        
        # Check forecasts
        forecasts = data['forecasts'].get(email, [])
        for forecast in forecasts:
            created_at = forecast.get('created_at', '')
            if created_at:
                try:
                    forecast_date = datetime.fromisoformat(created_at)
                    if not last_activity or forecast_date > last_activity:
                        last_activity = forecast_date
                except:
                    continue
        
        return last_activity.isoformat() if last_activity else 'Never'

    def get_financial_metrics(self) -> Dict:
        """Get financial metrics across all users"""
        data = self.load_data()
        transactions_data = data['transactions']
        
        total_income = 0
        total_expenses = 0
        total_transactions = 0
        category_breakdown = {}
        
        for email, transactions in transactions_data.items():
            for transaction in transactions:
                amount = float(transaction.get('amount', 0))
                transaction_type = transaction.get('type', '')
                category = transaction.get('category', 'Other')
                
                if transaction_type == 'income':
                    total_income += amount
                elif transaction_type == 'expense':
                    total_expenses += abs(amount)
                
                total_transactions += 1
                
                # Track category breakdown
                if category not in category_breakdown:
                    category_breakdown[category] = 0
                category_breakdown[category] += abs(amount)
        
        # Get top categories
        top_categories = dict(sorted(category_breakdown.items(), 
                                   key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'total_transactions': total_transactions,
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_savings': total_income - total_expenses,
            'top_categories': top_categories,
            'avg_transaction_value': (total_income + total_expenses) / max(1, total_transactions)
        }

    def get_user_activity_timeline(self, days: int = 30) -> Dict:
        """Get user activity timeline for charts"""
        data = self.load_data()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        timeline = {}
        current_date = start_date
        
        # Initialize timeline with zeros
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            timeline[date_str] = {
                'registrations': 0,
                'transactions': 0,
                'goals_created': 0,
                'forecasts_generated': 0
            }
            current_date += timedelta(days=1)
        
        # Count registrations
        for user_data in data['users'].values():
            created_at = user_data.get('created_at', '')
            if created_at:
                try:
                    user_date = datetime.fromisoformat(created_at).strftime('%Y-%m-%d')
                    if user_date in timeline:
                        timeline[user_date]['registrations'] += 1
                except:
                    continue
        
        # Count transactions
        for email, transactions in data['transactions'].items():
            for transaction in transactions:
                date_str = transaction.get('date', '')
                if date_str and date_str in timeline:
                    timeline[date_str]['transactions'] += 1
        
        # Count goals created
        for email, goals in data['goals'].items():
            for goal in goals:
                created_at = goal.get('created_at', '')
                if created_at:
                    try:
                        goal_date = datetime.fromisoformat(created_at).strftime('%Y-%m-%d')
                        if goal_date in timeline:
                            timeline[goal_date]['goals_created'] += 1
                    except:
                        continue
        
        # Count forecasts generated
        for email, forecasts in data['forecasts'].items():
            for forecast in forecasts:
                created_at = forecast.get('created_at', '')
                if created_at:
                    try:
                        forecast_date = datetime.fromisoformat(created_at).strftime('%Y-%m-%d')
                        if forecast_date in timeline:
                            timeline[forecast_date]['forecasts_generated'] += 1
                    except:
                        continue
        
        return timeline

    def generate_user_growth_chart(self) -> str:
        """Generate user growth over time chart"""
        try:
            data = self.load_data()
            users = data['users']
            
            # Extract registration dates and sort
            reg_dates = []
            for user_data in users.values():
                created_at = user_data.get('created_at', '')
                if created_at:
                    try:
                        reg_dates.append(datetime.fromisoformat(created_at))
                    except:
                        continue
            
            if not reg_dates:
                return None
            
            reg_dates.sort()
            
            # Calculate cumulative user count over time
            dates = []
            cumulative_users = []
            current_count = 0
            
            for date in reg_dates:
                current_count += 1
                dates.append(date)
                cumulative_users.append(current_count)
            
            # Create chart
            plt.figure(figsize=(10, 6))
            plt.plot(dates, cumulative_users, marker='o', linewidth=2, markersize=4, color='#4CAF50')
            plt.title('User Growth Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Total Users')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64 for HTML embedding
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_url}"
            
        except Exception as e:
            logger.error(f"Error generating user growth chart: {e}")
            return None

    def generate_platform_metrics_chart(self, stats: Dict) -> str:
        """Generate platform metrics bar chart"""
        try:
            metrics = ['Users', 'Transactions', 'Goals', 'Forecasts']
            counts = [
                stats['total_users'],
                stats['total_transactions'],
                stats['total_goals'],
                stats['total_forecasts']
            ]
            
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, counts, color=colors, alpha=0.8)
            plt.title('Platform Overview', fontsize=14, fontweight='bold')
            plt.ylabel('Count')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_url}"
            
        except Exception as e:
            logger.error(f"Error generating platform metrics chart: {e}")
            return None

    def generate_activity_breakdown_chart(self, stats: Dict) -> str:
        """Generate user activity breakdown pie chart"""
        try:
            # Calculate average transactions per user
            avg_transactions = stats['total_transactions'] / max(1, stats['total_users'])
            avg_goals = stats['total_goals'] / max(1, stats['total_users'])
            avg_forecasts = stats['total_forecasts'] / max(1, stats['total_users'])
            
            labels = ['Avg Transactions/User', 'Avg Goals/User', 'Avg Forecasts/User']
            values = [avg_transactions, avg_goals, avg_forecasts]
            colors = ['#FF6384', '#36A2EB', '#FFCE56']
            
            plt.figure(figsize=(8, 6))
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f', startangle=90)
            plt.title('User Activity Breakdown', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_url}"
            
        except Exception as e:
            logger.error(f"Error generating activity breakdown chart: {e}")
            return None

    def generate_forecast_usage_chart(self, stats: Dict) -> str:
        """Generate forecast usage patterns"""
        try:
            # Analyze forecast generation patterns
            users = stats['users']
            forecast_counts = [user['forecast_count'] for user in users]
            
            # Categorize users by forecast activity
            categories = ['No Forecasts', '1-5 Forecasts', '6+ Forecasts']
            counts = [
                len([x for x in forecast_counts if x == 0]),
                len([x for x in forecast_counts if 1 <= x <= 5]),
                len([x for x in forecast_counts if x > 5])
            ]
            
            colors = ['#E57373', '#64B5F6', '#81C784']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, counts, color=colors, alpha=0.8)
            plt.title('Forecast Usage Patterns', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Users')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_url}"
            
        except Exception as e:
            logger.error(f"Error generating forecast usage chart: {e}")
            return None

    def generate_financial_overview_chart(self) -> str:
        """Generate financial overview chart"""
        try:
            financial_metrics = self.get_financial_metrics()
            
            categories = ['Total Income', 'Total Expenses', 'Net Savings']
            amounts = [
                financial_metrics['total_income'],
                financial_metrics['total_expenses'],
                financial_metrics['net_savings']
            ]
            colors = ['#4CAF50', '#F44336', '#2196F3']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, amounts, color=colors, alpha=0.8)
            plt.title('Financial Overview Across Platform', fontsize=14, fontweight='bold')
            plt.ylabel('Amount ($)')
            
            # Add value labels
            for bar, amount in zip(bars, amounts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(amounts)*0.01,
                        f'${amount:,.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_url}"
            
        except Exception as e:
            logger.error(f"Error generating financial overview chart: {e}")
            return None

    def get_dashboard_data(self, email: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Get complete admin dashboard data"""
        if not self.is_admin(email):
            return None, "Access denied: Admin privileges required"
        
        try:
            # Get platform statistics
            stats = self.get_platform_statistics()
            financial_metrics = self.get_financial_metrics()
            activity_timeline = self.get_user_activity_timeline(30)
            
            # Generate all charts
            user_growth_chart = self.generate_user_growth_chart()
            platform_metrics_chart = self.generate_platform_metrics_chart(stats)
            activity_breakdown_chart = self.generate_activity_breakdown_chart(stats)
            forecast_usage_chart = self.generate_forecast_usage_chart(stats)
            financial_overview_chart = self.generate_financial_overview_chart()
            
            dashboard_data = {
                'platform_stats': stats,
                'financial_metrics': financial_metrics,
                'activity_timeline': activity_timeline,
                'charts': {
                    'user_growth': user_growth_chart,
                    'platform_metrics': platform_metrics_chart,
                    'activity_breakdown': activity_breakdown_chart,
                    'forecast_usage': forecast_usage_chart,
                    'financial_overview': financial_overview_chart
                },
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'admin_email': email
            }
            
            return dashboard_data, None
            
        except Exception as e:
            logger.error(f"Error generating admin dashboard: {e}")
            return None, f"Dashboard error: {str(e)}"

    def get_top_active_users(self, limit: int = 10) -> List[Dict]:
        """Get top active users based on various metrics"""
        data = self.load_data()
        
        user_activity = []
        
        for email, user_data in data['users'].items():
            transactions_count = len(data['transactions'].get(email, []))
            goals_count = len(data['goals'].get(email, []))
            forecasts_count = len(data['forecasts'].get(email, []))
            
            # Calculate activity score
            activity_score = (transactions_count * 0.5 + 
                             goals_count * 0.3 + 
                             forecasts_count * 0.2)
            
            user_activity.append({
                'email': email,
                'username': user_data.get('username', 'N/A'),
                'role': user_data.get('role', 'user'),
                'transactions_count': transactions_count,
                'goals_count': goals_count,
                'forecasts_count': forecasts_count,
                'activity_score': round(activity_score, 2),
                'last_active': self.get_last_active_date(email, data)
            })
        
        # Sort by activity score and return top users
        user_activity.sort(key=lambda x: x['activity_score'], reverse=True)
        return user_activity[:limit]

# Global instance
admin_dashboard = AdminDashboard()

# Initialize admin user when module is imported
admin_dashboard.create_admin_user()