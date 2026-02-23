@echo off
title Dota Ability Draft Assistant

:: ---- WINDOW SIZE ----
:: mode con: cols=WIDTH lines=HEIGHT
mode con: cols=140 lines=90

:: optional bigger font (works on most systems)
powershell -command "$Host.UI.RawUI.WindowTitle='Dota Ability Draft Assistant'"

cd /d Z:\DotaAD\dota_ad

"C:\Users\brett\AppData\Local\Programs\Python\Python310\python.exe" live_gui.py

echo.
echo Press any key to close...
pause >nul
