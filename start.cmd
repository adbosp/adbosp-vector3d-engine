@echo off
rem Vector3D Engine launcher (Windows).
rem   start.cmd                      menu
rem   start.cmd ui                   web interface
rem   start.cmd triposg anh.png      TripoSG: real AI image -> 3D (NVIDIA GPU)
rem   start.cmd photo anh.png        photo mode (Depth Anything, CPU ok)
rem   start.cmd triposg-lora anh.png TripoSG + LoRA trained on the engine's geometry
rem   start.cmd compare             benchmark released vs LoRA weights on the 6 objects
rem   start.cmd omni anhnguoi.png    Omni (nguoi, thu nghiem): TripoSG than + ban tay can canh (GPU + WSL)
rem   start.cmd benchmark | audit | test
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem --- engine venv: create on first run ---
if not exist ".venv\Scripts\python.exe" (
  echo [setup] Tao moi truong .venv lan dau...
  py -3 -m venv .venv || python -m venv .venv || (echo Khong tim thay Python 3.10+ & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -e .[dev] || (echo Cai dat that bai & pause & exit /b 1)
)
set "PY=%~dp0.venv\Scripts\python.exe"

rem --- TripoSG: default install from docs/LAB.md unless already set ---
if not defined V3D_TRIPOSG_DIR if exist "C:\ai3d\TripoSG\triposg" set "V3D_TRIPOSG_DIR=C:\ai3d\TripoSG"
if not defined V3D_TRIPOSG_PYTHON if exist "C:\ai3d\triposg-env\Scripts\python.exe" set "V3D_TRIPOSG_PYTHON=C:\ai3d\triposg-env\Scripts\python.exe"

set "CMD=%~1"
if "%CMD%"=="" goto menu
shift
goto run

:menu
echo.
echo  ===== Vector3D Engine =====
echo   1. Giao dien web (Generate, xem 3D)
echo   2. TripoSG: anh -^> model 3D bang AI (can GPU NVIDIA)
echo   3. Photo mode: anh -^> model 3D (Depth Anything, chay CPU)
echo   4. Benchmark TripoSG tren 6 vat the
echo   5. Kiem tra phan cung (GPU, VRAM, CUDA)
echo   6. Chay test
echo   7. TripoSG + LoRA (ban train lai): anh -^> model 3D
echo   8. So sanh TripoSG goc va ban LoRA tren 6 vat the
echo   9. Omni nguoi (thu nghiem): anh nguoi -^> model 3D, ban tay dung can canh
echo   0. Thoat
echo.
set /p "CHOICE=Chon: "
if "%CHOICE%"=="1" set "CMD=ui"
if "%CHOICE%"=="2" set "CMD=triposg"
if "%CHOICE%"=="3" set "CMD=photo"
if "%CHOICE%"=="4" set "CMD=benchmark"
if "%CHOICE%"=="5" set "CMD=audit"
if "%CHOICE%"=="6" set "CMD=test"
if "%CHOICE%"=="7" set "CMD=triposg-lora"
if "%CHOICE%"=="8" set "CMD=compare"
if "%CHOICE%"=="9" set "CMD=omni"
if "%CHOICE%"=="0" exit /b 0
if "%CMD%"=="" goto menu
set "IMG="
if /i "%CMD%"=="triposg" set /p "IMG=Duong dan anh (keo tha file vao day): "
if /i "%CMD%"=="photo" set /p "IMG=Duong dan anh (keo tha file vao day): "
if /i "%CMD%"=="triposg-lora" set /p "IMG=Duong dan anh (keo tha file vao day): "
if /i "%CMD%"=="omni" set /p "IMG=Duong dan anh nguoi (keo tha file vao day): "
set "PAUSE_AT_END=1"
goto dispatch

:run
set "IMG=%~1"

:dispatch
if defined IMG set "IMG=!IMG:"=!"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%t"
if /i "%CMD%"=="ui" (
  "%PY%" -m vector3d ui
) else if /i "%CMD%"=="triposg" (
  if not defined IMG (echo Can duong dan anh. & goto end)
  if not defined V3D_TRIPOSG_DIR (echo Chua cai TripoSG: xem docs\LAB.md & goto end)
  set "OUT=runs\triposg_!STAMP!"
  "%PY%" -m vector3d lab generate --image "!IMG!" -o "!OUT!" --profile balanced --download
  if exist "!OUT!\gates\previews.png" start "" "!OUT!\gates\previews.png"
  echo Ket qua: %~dp0!OUT!
) else if /i "%CMD%"=="triposg-lora" (
  if not defined IMG (echo Can duong dan anh. & goto end)
  if not defined V3D_TRIPOSG_DIR (echo Chua cai TripoSG: xem docs\LAB.md & goto end)
  if not exist "bench\ft_run\lora\adapter_config.json" (echo Chua co LoRA: chay lab finetune truoc, xem docs\LAB.md & goto end)
  set "OUT=runs\triposg_lora_!STAMP!"
  "%PY%" -m vector3d lab generate --image "!IMG!" -o "!OUT!" --profile balanced --lora bench\ft_run\lora
  if exist "!OUT!\gates\previews.png" start "" "!OUT!\gates\previews.png"
  echo Ket qua: %~dp0!OUT!
) else if /i "%CMD%"=="omni" (
  if not defined IMG (echo Can duong dan anh. & goto end)
  if not defined V3D_TRIPOSG_DIR (echo Chua cai TripoSG: xem docs\LAB.md & goto end)
  echo Omni la che do THU NGHIEM: than tot, ban tay can canh chua on dinh. Mat khoang 4-5 phut.
  set "OUT=runs\omni_!STAMP!"
  "%PY%" -m vector3d generate --front "!IMG!" --mode omni -o "!OUT!"
  echo Ket qua: %~dp0!OUT!  ^(mo model.glb bang giao dien web hoac 3D Viewer^)
) else if /i "%CMD%"=="compare" (
  if not exist "bench\ft_run\lora\adapter_config.json" (echo Chua co LoRA: chay lab finetune truoc & goto end)
  if not exist "bench\data\manifest.json" "%PY%" -m vector3d lab dataset -o bench\data
  if not exist "bench\triposg_base_rerun\summary.json" "%PY%" -m vector3d lab benchmark --data bench\data -o bench\triposg_base_rerun
  if not exist "bench\triposg_lora\summary.json" "%PY%" -m vector3d lab benchmark --data bench\data -o bench\triposg_lora --lora bench\ft_run\lora
  "%PY%" -m vector3d lab compare bench\triposg_base_rerun bench\triposg_lora
) else if /i "%CMD%"=="photo" (
  if not defined IMG (echo Can duong dan anh. & goto end)
  set "OUT=out\photo_!STAMP!"
  "%PY%" -m vector3d generate --front "!IMG!" --mode photo -o "!OUT!"
  echo Ket qua: %~dp0!OUT!
) else if /i "%CMD%"=="benchmark" (
  if not exist "audit.json" "%PY%" -m vector3d lab audit -o audit.json >nul
  if not exist "bench\data\manifest.json" "%PY%" -m vector3d lab dataset -o bench\data
  "%PY%" -m vector3d lab benchmark --data bench\data -o bench\triposg --profile balanced
  if not exist "bench\exp001\summary.json" "%PY%" -m vector3d lab experiment --data bench\data -o bench\exp001
  "%PY%" -m vector3d lab report --experiment bench\exp001 --benchmark bench\triposg --audit audit.json -o bench\report.md
  if exist "bench\report.md" start "" "bench\report.md"
) else if /i "%CMD%"=="audit" (
  if defined V3D_TRIPOSG_PYTHON ("%PY%" -m vector3d lab audit --python "%V3D_TRIPOSG_PYTHON%" -o audit.json) else ("%PY%" -m vector3d lab audit -o audit.json)
) else if /i "%CMD%"=="test" (
  "%PY%" -m pytest -q
) else (
  echo Lenh khong hop le: %CMD%
  echo Dung: start.cmd [ui ^| triposg anh.png ^| triposg-lora anh.png ^| omni anhnguoi.png ^| photo anh.png ^| compare ^| benchmark ^| audit ^| test]
)

:end
if defined PAUSE_AT_END pause
endlocal
