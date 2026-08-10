@echo off
setlocal

set "PYTHON312=C:\Users\Acer\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PYTHON312%" (
    echo Python 3.12 was not found at %PYTHON312%
    exit /b 1
)

"%PYTHON312%" "%~dp0lunar_lander_dqn.py" %*