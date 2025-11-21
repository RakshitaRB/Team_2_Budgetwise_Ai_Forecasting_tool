# ============================================================================
# AI-Based Expense Forecasting Tool - All-in-One Launcher
# ============================================================================
# This script starts:
#   1. Backend Flask API (port 5000)
#   2. Main Streamlit App (port 8501)
#   3. Admin Dashboard (port 8502)
# ============================================================================

Clear-Host
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  AI-Based Expense Forecasting Tool - Launcher                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# Step 1: Activate venv
Write-Host "🔧 Setting up Python environment..." -ForegroundColor Yellow
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force | Out-Null
. .\venv\Scripts\Activate.ps1

# Step 2: Start Backend
Write-Host "🚀 Starting Backend API (port 5000)..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath (Get-Command python).Source `
  -ArgumentList ".\AI-Based-Expense-Forecasting-Tool-main\backend\app.py" `
  -PassThru -WindowStyle Minimized
Write-Host "   Backend PID: $($backendProcess.Id)" -ForegroundColor Green

# Step 3: Wait for backend to be ready
Write-Host "⏳ Waiting for Backend to be ready..." -ForegroundColor Yellow
$attempts = 0
$backendReady = $false
while ($attempts -lt 15 -and -not $backendReady) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "   ✓ Backend is ready!" -ForegroundColor Green
        }
    } catch {
        Start-Sleep -Seconds 1
        $attempts++
    }
}

if (-not $backendReady) {
    Write-Host "   ✗ Backend failed to start!" -ForegroundColor Red
    exit 1
}

# Step 4: Start Main Streamlit App
Write-Host "📊 Starting Main Streamlit App (port 8501)..." -ForegroundColor Green
$streamlitMainProcess = Start-Process -FilePath (Get-Command python).Source `
  -ArgumentList "-m", "streamlit", "run", ".\AI-Based-Expense-Forecasting-Tool-main\frontend\streamlit_app.py", "--server.port", "8501" `
  -PassThru -WindowStyle Minimized
Write-Host "   Streamlit Main PID: $($streamlitMainProcess.Id)" -ForegroundColor Green

# Step 5: Start Admin Dashboard
Write-Host "👨‍💼 Starting Admin Dashboard (port 8502)..." -ForegroundColor Green
$streamlitAdminProcess = Start-Process -FilePath (Get-Command python).Source `
  -ArgumentList "-m", "streamlit", "run", ".\AI-Based-Expense-Forecasting-Tool-main\frontend\admin_dashboard.py", "--server.port", "8502" `
  -PassThru -WindowStyle Minimized
Write-Host "   Streamlit Admin PID: $($streamlitAdminProcess.Id)" -ForegroundColor Green

# Step 6: Wait for Streamlit apps to be ready
Write-Host "⏳ Waiting for Streamlit apps to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

$mainReady = $false
$adminReady = $false
$attempts = 0

while ($attempts -lt 15 -and (-not $mainReady -or -not $adminReady)) {
    if (-not $mainReady) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $mainReady = $true
                Write-Host "   ✓ Main App is ready!" -ForegroundColor Green
            }
        } catch { }
    }
    
    if (-not $adminReady) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8502" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $adminReady = $true
                Write-Host "   ✓ Admin Dashboard is ready!" -ForegroundColor Green
            }
        } catch { }
    }
    
    if (-not ($mainReady -and $adminReady)) {
        Start-Sleep -Seconds 1
        $attempts++
    }
}

# Step 7: Display startup summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  ✅ All services started successfully!                          ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "📱 Access the applications:" -ForegroundColor Yellow
Write-Host "   🏠 Main App:       http://localhost:8501" -ForegroundColor Cyan
Write-Host "   👨‍💼 Admin Panel:    http://localhost:8502" -ForegroundColor Cyan
Write-Host "   🔗 Backend API:    http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Running Processes:" -ForegroundColor Yellow
Write-Host "   Backend:          PID $($backendProcess.Id)" -ForegroundColor Green
Write-Host "   Main Streamlit:   PID $($streamlitMainProcess.Id)" -ForegroundColor Green
Write-Host "   Admin Streamlit:  PID $($streamlitAdminProcess.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  Default Credentials:" -ForegroundColor Yellow
Write-Host "   Admin Email:      admin@budgetwise.com" -ForegroundColor Cyan
Write-Host "   Admin Password:   admin123" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 To stop all services, close this window or press Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Keep script running
Write-Host "🔄 Monitoring services..." -ForegroundColor Yellow
while ($true) {
    Start-Sleep -Seconds 5
    
    # Check if any process has exited
    if ($backendProcess.HasExited) {
        Write-Host "⚠️  Backend process exited!" -ForegroundColor Red
    }
    if ($streamlitMainProcess.HasExited) {
        Write-Host "⚠️  Main Streamlit process exited!" -ForegroundColor Red
    }
    if ($streamlitAdminProcess.HasExited) {
        Write-Host "⚠️  Admin Streamlit process exited!" -ForegroundColor Red
    }
}
