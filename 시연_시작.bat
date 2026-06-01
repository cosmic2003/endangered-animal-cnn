@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════╗
echo ║     멸종위기 동물 분류 서비스 시연 시작    ║
echo ╚══════════════════════════════════════════╝
echo.

:: 현재 핫스팟/LAN IP 출력
echo [내 IP 주소 목록]
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do echo   http://%%b:8000
)
echo.
echo 위 주소 중 핫스팟 IP(192.168.x.x)를 QR코드로 만들어 공유하세요.
echo.
echo [서버 시작 중...]
cd /d "%~dp0backend\app"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
