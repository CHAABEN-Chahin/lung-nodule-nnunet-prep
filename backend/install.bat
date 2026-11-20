@echo off
echo === Nodulis Backend - Installing Dependencies ===
echo.

echo [1/4] Installing numpy first (build dependency for pyradiomics)...
pip install numpy
echo.

echo [2/4] Installing main requirements...
pip install -r requirements.txt
echo.

echo [3/4] Installing pyradiomics (requires numpy already present)...
pip install pyradiomics
echo.

echo [4/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo === Installation Complete ===
echo Run: python app.py
pause
