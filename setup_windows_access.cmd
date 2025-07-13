@echo off
echo ========================================
echo SutazAI Windows Access Setup
echo ========================================
echo.
echo This script sets up port forwarding so you can access
echo SutazAI from your Windows browser at http://localhost:5555
echo.
echo Run this in Windows Command Prompt as Administrator:
echo.
echo 1. Remove any existing port proxy:
netsh interface portproxy delete v4tov4 listenport=5555
echo.
echo 2. Add new port forwarding:
netsh interface portproxy add v4tov4 listenport=5555 listenaddress=0.0.0.0 connectport=5555 connectaddress=172.17.0.3
echo.
echo 3. Check if it was added successfully:
netsh interface portproxy show all
echo.
echo ========================================
echo After running this, open your browser and go to:
echo http://localhost:5555
echo ========================================
pause