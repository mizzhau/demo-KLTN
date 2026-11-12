@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Hay cai Python 3.10 tro len va tick "Add python.exe to PATH".
    pause
    exit /b 1
)

python -c "import streamlit, ultralytics, pi_heif, pandas" >nul 2>nul
if errorlevel 1 (
    echo Dang cai thu vien can thiet lan dau, vui long doi...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai thu vien that bai. Xem thong bao loi o tren.
        pause
        exit /b 1
    )
)

echo Dang khoi dong demo tai http://localhost:8501 ...
rem Mo trinh duyet sau 6 giay, khi server da san sang
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
python -m streamlit run app.py --server.port 8501 --server.headless true
pause
