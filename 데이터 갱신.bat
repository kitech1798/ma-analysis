@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo === 이동평균선 분석 — 데이터 증분 갱신 ===
python download_data.py
echo.
pause
