@echo off
REM ============================================================================
REM AI-Based Expense Forecasting Tool - All-in-One Launcher (Batch)
REM ============================================================================

cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║  AI-Based Expense Forecasting Tool - Launcher                 ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Activate venv
echo 🔧 Setting up Python environment...
call .\venv\Scripts\activate.bat

REM Start Backend
echo 🚀 Starting Backend API (port 5000)...
start "Backend API" python .\AI-Based-Expense-Forecasting-Tool-main\backend\app.py
timeout /t 3 /nobreak

REM Start Main Streamlit App
echo 📊 Starting Main Streamlit App (port 8501)...
start "Expense Forecaster" python -m streamlit run .\AI-Based-Expense-Forecasting-Tool-main\frontend\streamlit_app.py --server.port 8501
timeout /t 3 /nobreak

REM Start Admin Dashboard
echo 👨‍💼 Starting Admin Dashboard (port 8502)...
start "Admin Dashboard" python -m streamlit run .\AI-Based-Expense-Forecasting-Tool-main\frontend\admin_dashboard.py --server.port 8502
timeout /t 2 /nobreak

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║  ✓ All services started!                                       ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo 📱 Access the applications:
echo    🏠 Main App:       http://localhost:8501
echo    👨‍💼 Admin Panel:    http://localhost:8502
echo    🔗 Backend API:    http://localhost:5000
echo.
echo ⚠️  Default Credentials:
echo    Admin Email:      admin@budgetwise.com
echo    Admin Password:   admin123
echo.
echo 💡 Services running in background windows. Close them to stop.
echo.
pause
