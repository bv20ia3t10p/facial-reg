@echo off
REM Script to clean up cache files, build artifacts, and other temporary files

echo Cleaning up project files...

REM Clean Python cache files
echo Removing Python cache files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1
del /s /q *.pyo >nul 2>&1
del /s /q *.pyd >nul 2>&1
del /s /q .coverage >nul 2>&1
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.eggs) do @if exist "%%d" rd /s /q "%%d"

REM Clean Docker artifacts (but preserve docker-compose.yml)
echo Cleaning Docker artifacts...
if exist .docker rd /s /q .docker
if exist .env del /q .env

REM Clean build artifacts
echo Removing build artifacts...
if exist build rd /s /q build
if exist dist rd /s /q dist

REM Clean temporary runtime files 
echo Removing temporary runtime files...
if exist logs rd /s /q logs
if exist output rd /s /q output
if exist runs rd /s /q runs
if exist .cached_extractions* rd /s /q .cached_extractions*
del /s /q *.log >nul 2>&1
del /s /q *.tfevents.* >nul 2>&1

REM Optional: Remove shared Docker volume data (uncomment if needed)
REM echo Removing shared Docker volume data...
REM if exist shared rd /s /q shared

REM Optional: Clean model files (uncomment if needed)
REM echo Removing saved models...
REM if exist models rd /s /q models

echo Cleanup complete! 