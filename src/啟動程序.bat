@echo off
:: 獲取當前腳本所在目錄
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM 設定 Python 預期安裝目錄
set "PYTHON_INSTALL_DIR=%LOCALAPPDATA%\Programs\Python\Python312"

REM 檢查是否安裝了 Python 3.12.x（大於 3.12 且小於 3.13）
if not exist "%PYTHON_INSTALL_DIR%\python.exe" (
    echo Python 3.12.x not installed, starting installation...

    :: 檢查安裝檔是否存在，適用於任何 3.12.x 版本的 Python 安裝檔案
    if not exist "%SCRIPT_DIR%\setup\component\python-3.12.8-amd64.exe" (
        echo Python installer not found in setup\component\
        pause
        exit /b
    )

    :: 執行安裝並顯示安裝界面
    echo Starting Python 3.12.8 installation...
    "%SCRIPT_DIR%\setup\component\python-3.12.8-amd64.exe"

    :: 驗證安裝
    if not exist "%PYTHON_INSTALL_DIR%\python.exe" (
        echo Failed to install Python 3.12.x. Please check the installer.
        pause
        exit /b
    ) else (
        echo Python 3.12.x successfully installed at %PYTHON_INSTALL_DIR%.
    )
) else (
    echo Python 3.12.x is already installed at %PYTHON_INSTALL_DIR%.
)

REM 呼叫虛擬環境
if not exist "%SCRIPT_DIR%\NCUT_HOSPITAL" (
    echo NCUT_HOSPITAL not found, creating...
    "%PYTHON_INSTALL_DIR%\python.exe" -m venv "%SCRIPT_DIR%\NCUT_HOSPITAL"
    if exist "%SCRIPT_DIR%\NCUT_HOSPITAL\Scripts\activate.bat" (
        call "%SCRIPT_DIR%\NCUT_HOSPITAL\Scripts\activate.bat"
        echo Installing requirements...
        "%PYTHON_INSTALL_DIR%\python.exe" PIPInstaller.py
    ) else (
        echo Failed to create virtual environment. Please check Python installation.
        pause
        exit /b
    )
) else (
    echo Activating existing virtual environment...
    call "%SCRIPT_DIR%\NCUT_HOSPITAL\Scripts\activate.bat"
)

echo *************************************************
echo Python 3.12.x and virtual environment activated
echo *************************************************

:: 切換到 Django 項目目錄
cd /d "%SCRIPT_DIR%"
"%PYTHON_INSTALL_DIR%\\python.exe" Main.py
pause