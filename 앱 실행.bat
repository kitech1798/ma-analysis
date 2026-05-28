@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo === 이동평균선 분석 (포트 8501) ===
echo 브라우저에서 http://localhost:8501 접속
echo 종료: Ctrl+C
echo.
streamlit run ma_app.py --server.port 8501
pause
