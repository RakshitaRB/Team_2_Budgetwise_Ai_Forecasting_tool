# AI-Based Expense Forecasting Tool - Quick Start Guide

## 📋 Prerequisites
- Python 3.8+ (already installed)
- Virtual environment created (`venv/`)
- Dependencies installed (`pip install -r requirements.txt`)

---

## 🚀 Quick Start (All-in-One)

### Option 1: Batch File (Windows)
Simply double-click or run in CMD:
```batch
RUN.bat
```
This will automatically start:
- Backend API (port 5000)
- Main Streamlit App (port 8501)
- Admin Dashboard (port 8502)

---

## 🔧 Manual Setup (Step-by-Step)

### Step 1: Open PowerShell and navigate to project
```powershell
cd C:\Users\vasan\Downloads\AI-Based-Expense-Forecasting-Tool-main1
```

### Step 2: Activate Virtual Environment
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
. .\venv\Scripts\Activate.ps1
```

### Step 3: Install Requirements (if not already done)
```powershell
pip install -r .\AI-Based-Expense-Forecasting-Tool-main\requirements.txt
```

### Step 4: Initialize Database
```powershell
python .\AI-Based-Expense-Forecasting-Tool-main\backend\init_db_py.py
```

---

## 🎯 Running Services

### Terminal 1: Start Backend API
```powershell
python .\AI-Based-Expense-Forecasting-Tool-main\backend\app.py
```
- Runs on: **http://localhost:5000**
- Health check: `http://localhost:5000/health`

### Terminal 2: Start Main Streamlit App
```powershell
streamlit run .\AI-Based-Expense-Forecasting-Tool-main\frontend\streamlit_app.py --server.port 8501
```
- Runs on: **http://localhost:8501**
- Features: Transaction management, reports, forecasting, goals

### Terminal 3: Start Admin Dashboard
```powershell
streamlit run .\AI-Based-Expense-Forecasting-Tool-main\frontend\admin_dashboard.py --server.port 8502
```
- Runs on: **http://localhost:8502**
- Features: System analytics, user management, transactions, categories

---

## 🌐 Access the Applications

| Service | URL | Purpose |
|---------|-----|---------|
| Main App | http://localhost:8501 | User expense management |
| Admin Panel | http://localhost:8502 | System administration |
| Backend API | http://localhost:5000 | REST API endpoints |

---

## 🔐 Default Credentials

**Admin Account:**
- Email: `admin@budgetwise.com`
- Password: `admin123`

**To create a new user:**
1. Go to http://localhost:8501
2. Click "Register"
3. Enter email and password
4. Click "Submit"

---

## 📝 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user

### Transactions
- `GET /transactions` - Get all transactions
- `POST /transactions` - Add single transaction
- `POST /transactions/bulk` - Upload CSV of transactions
- `PUT /transactions/<id>/category` - Update transaction category

### Reports
- `GET /reports/overview` - Get total income, expense, balance
- `GET /reports/category?days=30` - Get category breakdown
- `GET /reports/monthly?months=6` - Get monthly summary

### Forecasting
- `POST /forecast` - Generate expense forecast
- `GET /forecast/categories` - Get available forecast categories
- `POST /forecast/compare` - Compare forecasts across categories

### Goals
- `GET /goals` - Get all user goals
- `POST /goals` - Create new goal
- `DELETE /goals/<id>` - Delete goal
- `POST /goals/<id>/savings` - Add savings to goal

### Admin
- `GET /admin/users` - Get all users (admin only)
- `GET /admin/transactions` - Get all transactions (admin only)
- `GET /admin/analytics` - Get system analytics (admin only)
- `GET /admin/categories` - Get category statistics (admin only)

---

## 🐛 Troubleshooting

### Backend not starting?
```powershell
# Check if port 5000 is in use
Get-NetTCPConnection -LocalPort 5000
```

### Streamlit connection error?
- Ensure backend is running first
- Check if `API_BASE` in `streamlit_app.py` is set to `http://localhost:5000`

### Database errors?
```powershell
# Reinitialize database
python .\AI-Based-Expense-Forecasting-Tool-main\backend\init_db_py.py
```

### Clear Python cache
```powershell
Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Force -Recurse
```

---

## 📂 Project Structure

```
AI-Based-Expense-Forecasting-Tool-main1/
├── venv/                          # Virtual environment
├── AI-Based-Expense-Forecasting-Tool-main/
│   ├── backend/
│   │   ├── app.py                # Flask backend server
│   │   ├── auth.py               # Authentication logic
│   │   ├── db.py                 # Database functions
│   │   ├── forecast_engine.py    # AI forecasting
│   │   ├── categorizer.py        # Transaction categorizer
│   │   ├── goals.py              # Goal management
│   │   ├── init_db_py.py         # Database initialization
│   │   ├── init_db.sql           # Database schema
│   │   └── test.py               # Verification script
│   ├── frontend/
│   │   ├── streamlit_app.py      # Main user app
│   │   └── admin_dashboard.py    # Admin panel
│   ├── data/                      # SQLite database
│   └── requirements.txt           # Python dependencies
├── RUN.bat                        # All-in-one launcher
└── QUICKSTART.md                  # This file
```

---

## 🔄 Development Workflow

1. **Make code changes** to `backend/` or `frontend/`
2. **Backend**: Restart Flask server (Ctrl+C, run again)
3. **Frontend**: Streamlit auto-reloads on file changes
4. **Test**: Visit http://localhost:8501 or http://localhost:8502

---

## 📊 Features Overview

### User App (Port 8501)
- ✅ User authentication (register/login)
- ✅ Add individual transactions
- ✅ Bulk import transactions (CSV)
- ✅ View transaction history
- ✅ Category breakdown reports
- ✅ Monthly spending summary
- ✅ AI-powered expense forecasting
- ✅ Create and track financial goals
- ✅ Goal coaching with AI recommendations
- ✅ Override transaction categories

### Admin Dashboard (Port 8502)
- ✅ System analytics (users, transactions, goals)
- ✅ User management and statistics
- ✅ All transactions across system
- ✅ Category management and bulk updates
- ✅ System health checks
- ✅ Data consistency verification

### Backend API (Port 5000)
- ✅ RESTful endpoints for all features
- ✅ JWT authentication
- ✅ Database management
- ✅ Advanced forecasting with Prophet
- ✅ Category auto-detection (NLP)
- ✅ Goal progress tracking

---

## 💾 Database

- **Type**: SQLite
- **Location**: `data/expense.db`
- **Schema**: Automatically created on first run
- **Tables**: users, transactions, financial_goals, admin_users (if applicable)

To reset database:
```powershell
Remove-Item .\AI-Based-Expense-Forecasting-Tool-main\data\expense.db
python .\AI-Based-Expense-Forecasting-Tool-main\backend\init_db_py.py
```

---

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` in activated venv |
| "Port already in use" | Change port in command or kill process using port |
| "Connection refused" | Ensure backend is running first |
| "Login fails" | Check credentials or create new account |
| "Streamlit not updating" | Press R key in Streamlit, or restart |
| "Database locked" | Close all instances and delete `.db` file |

---

## 📞 Support

For issues, check:
- Backend logs at terminal
- Streamlit logs in browser console (F12)
- Database structure: `PRAGMA table_info(table_name);`

---

**Happy budgeting! 💰**
