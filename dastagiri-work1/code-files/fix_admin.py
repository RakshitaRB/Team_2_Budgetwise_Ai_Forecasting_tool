# fix_admin.py - Run this once to create your first admin account
import json
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.auth import get_password_hash

def create_first_admin():
    """Create the first admin account directly in the database"""
    print("🛠️ Creating first admin account...")
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load existing data or create new structure
    try:
        with open('data/users.json', 'r') as f:
            data = json.load(f)
        print("📁 Loaded existing database")
    except (FileNotFoundError, json.JSONDecodeError):
        print("📁 Creating new database structure")
        data = {
            'users': {},
            'transactions': {},
            'goals': {},
            'forecasts': {}
        }
    
    # Admin account details
    admin_accounts = [
        {
            "username": "System Administrator",
            "email": "admin@budget.com",
            "password": "Admin123!"
        },
        {
            "username": "Test Administrator", 
            "email": "test@admin.com",
            "password": "Test123!"
        }
    ]
    
    created_count = 0
    for admin in admin_accounts:
        email = admin['email']
        
        # Check if admin already exists
        if email in data.get('users', {}):
            print(f"ℹ️ Admin account already exists: {email}")
            continue
        
        # Create admin account
        hashed_password = get_password_hash(admin['password'])
        
        data['users'][email] = {
            'username': admin['username'],
            'email': email,
            'hashed_password': hashed_password,
            'role': 'admin',
            'created_at': '2024-01-15T10:30:00'
        }
        
        # Initialize user data structures
        data['transactions'][email] = []
        data['goals'][email] = []
        data['forecasts'][email] = []
        
        created_count += 1
        print(f"✅ Created admin: {email} / {admin['password']}")
    
    # Save back to file
    with open('data/users.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n🎉 Successfully created {created_count} admin accounts!")
    print("\n🔑 You can now login with:")
    print("   - admin@budget.com / Admin123!")
    print("   - test@admin.com / Test123!")
    print("\n🚀 Run: streamlit run app.py")

if __name__ == "__main__":
    create_first_admin()