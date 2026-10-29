@echo off
REM Usage: run_project.bat setup | test | full | dashboard
cd /d "%~dp0"
if "%1"=="setup" (
  python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt && copy /n .env.example .env
  goto :eof
)
call .venv\Scripts\activate
if "%1"=="test" (
  python src\generate_data.py --rows 100000 && python src\pipeline.py && python src\benchmark.py --sizes 100000 && streamlit run dashboard\dashboard.py
  goto :eof
)
if "%1"=="full" (
  python src\generate_data.py --rows 20000000 && python src\pipeline.py && python src\benchmark.py --sizes 100000,500000,1000000,5000000,10000000,20000000
  goto :eof
)
if "%1"=="dashboard" ( streamlit run dashboard\dashboard.py & goto :eof )
echo Usage: run_project.bat setup ^| test ^| full ^| dashboard
