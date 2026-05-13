@echo off
chcp 65001 > nul
cd /d "%~dp0"
streamlit run ma_app.py
