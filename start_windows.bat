@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Все, что ставится, остается внутри этой папки: .tools (uv, Python, кэш пакетов) и .venv-app.
set "TOOLS=%~dp0.tools"
set "UV=%TOOLS%\uv\uv.exe"
set "UV_CACHE_DIR=%TOOLS%\cache"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
rem Python берется только свой, из .tools: установленный на компьютере не используется и не меняется.
set "UV_PYTHON_PREFERENCE=only-managed"
set "VENV=%~dp0.venv-app"

echo.
echo  Прототип: ИИ-поддержка врача - опухоли головного мозга на МРТ
echo  ------------------------------------------------------------
echo.

if not exist "%UV%" (
  echo [1/3] Скачиваю установщик пакетов uv...
  if not exist "%TOOLS%\uv" mkdir "%TOOLS%\uv"
  rem curl.exe есть в Windows 10 и 11; если он не справился - загрузка через PowerShell
  curl.exe -fsSL -o "%TOOLS%\uv.zip" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" || powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '%TOOLS%\uv.zip'"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%TOOLS%\uv.zip' '%TOOLS%\uv'"
  del "%TOOLS%\uv.zip" 2>nul
  if not exist "%UV%" goto fail
)

if not exist "%VENV%\ready.txt" (
  echo [2/3] Первый запуск: ставлю Python и пакеты, это 5-10 минут...
  "%UV%" venv --clear --python 3.10 "%VENV%"
  if errorlevel 1 goto fail
  "%UV%" pip install --python "%VENV%\Scripts\python.exe" torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto fail
  "%UV%" pip install --python "%VENV%\Scripts\python.exe" -r requirements-app.txt
  if errorlevel 1 goto fail
  echo ok> "%VENV%\ready.txt"
)

echo [3/3] Запускаю прототип, браузер откроется сам...
"%VENV%\Scripts\python.exe" app\launch.py
if errorlevel 1 goto fail
goto end

:fail
echo.
echo Не получилось запустить - причина в сообщениях выше. При первом запуске нужен интернет.
pause

:end
endlocal
