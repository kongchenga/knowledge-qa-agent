@echo off
chcp 65001 >nul
title Knowledge QA Agent

echo ============================================
echo  Knowledge QA Agent v2.0.0
echo ============================================
echo.

if "%LLM_API_KEY%"=="" (
    if exist .env (
        echo Loading environment from .env
    ) else (
        echo [WARNING] LLM_API_KEY is not set
        echo Create a .env file from .env.example or set the environment variable
        echo.
    )
)

echo Starting server...
echo.

python run.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server exited with error code %ERRORLEVEL%
    pause
)
