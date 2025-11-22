# create_admin.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.auth import get_password_hash
from utils.database import add_user, load_data, save_data

def create_admin_accounts():
    """Create default admin accounts for the system"""
    admin_accounts = [
        {"username": "Main Administrator", "email": "admin@budget.com", "password": "Admin123!"},
        {"username": "Test Admin", "email": "test@admin.com", "password": "Test123!"},
        {"username": "Super Admin", "email": "super@admin.com", "password": "Super123!"},
    ]
    
    results = []
    for admin in admin_accounts:
        try:
            # Check if user already exists
            data = load_data()
            if admin['email'] in data.get('users', {}):
                results.append(f"Admin {admin['email']} already exists")
                continue
            
            # Create admin user
            hashed_password = get_password_hash(admin['password'])
            success, message = add_user(admin['username'], admin['email'], hashed_password, 'admin')
            
            if success:
                results.append(f"✅ Admin {admin['email']} created successfully")
            else:
                results.append(f"❌ Failed to create {admin['email']}: {message}")
                
        except Exception as e:
            results.append(f"❌ Error creating {admin['email']}: {str(e)}")
    
    return results

if __name__ == "__main__":
    print("Creating admin accounts...")
    results = create_admin_accounts()
    for result in results:
        print(result)