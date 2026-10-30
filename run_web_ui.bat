@echo off
title Deepfake & Face-Swap Cyber Forensics Lab
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
echo Starting Deepfake Forensic Web Dashboard...
streamlit run web_app.py
pause
