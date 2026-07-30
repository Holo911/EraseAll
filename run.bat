@echo off
cd /d "%~dp0"
title EraseAll

REM All the real logic lives in start.ps1 (port scan, prerequisite checks,
REM browser timing). -ExecutionPolicy Bypass so it also runs on a machine that
REM has never had PowerShell scripts enabled.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"

REM Always pause: without this the window vanishes on any error and the user
REM never gets to read what went wrong.
echo.
pause
