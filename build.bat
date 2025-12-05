@echo off
REM ASRInput Build Script for Windows
REM Creates standalone executable using PyInstaller

echo.
echo ============================================================
echo   ASRInput Build Script
echo ============================================================
echo.

REM Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv is not installed or not in PATH
    echo Please install uv: https://docs.astral.sh/uv/
    exit /b 1
)

REM Sync dependencies
echo [1/4] Syncing dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to sync dependencies
    exit /b 1
)

REM Install PyInstaller
echo [2/4] Installing PyInstaller...
uv add --dev pyinstaller
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install PyInstaller
    exit /b 1
)

REM Get customtkinter path
echo [3/4] Getting customtkinter path...
for /f "delims=" %%i in ('uv run python -c "import customtkinter; print(customtkinter.__path__[0])"') do set CTK_PATH=%%i
echo    Found: %CTK_PATH%

REM Build executable
echo [4/4] Building executable...
uv run pyinstaller --onefile --windowed --name asrinput --add-data "%CTK_PATH%;customtkinter" main.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed
    exit /b 1
)

echo.
echo ============================================================
echo   Build Successful!
echo ============================================================
echo   Output: dist\asrinput.exe
echo.
echo   Note: First run may take time to download Whisper model
echo ============================================================
echo.

pause
