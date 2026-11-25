@echo off
title Potatostics Standalone Executable Builder
cd /d "%~dp0.."
echo ===================================================
echo Building Potatostics Standalone Windows Executable
echo ===================================================
if not defined TEMP set "TEMP=%LOCALAPPDATA%\Temp"
python -m PyInstaller potatostics.spec --noconfirm --clean --workpath "%TEMP%\pyi_build"
echo.
echo ===================================================
echo Build Complete! Check the 'dist\Potatostics' folder.
echo ===================================================
pause
