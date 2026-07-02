@echo off
rem Double-click this file to start the Personal PDF Chatbot.
rem Close this window (or press Ctrl+C) to stop the app.
cd /d "%~dp0"

if "%GROQ_API_KEY%"=="" (
  echo GROQ_API_KEY is not set for this window.
  echo Set it once with:   setx GROQ_API_KEY "your-key-here"
  echo ...then double-click this file again.
  pause
  exit /b 1
)

rem Silence the harmless HuggingFace symlink warning on Windows.
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Starting the PDF Chatbot at http://localhost:8501 ...
.venv\Scripts\python -m streamlit run app.py
pause
