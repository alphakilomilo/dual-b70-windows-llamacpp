@echo off
rem Qwen3.8-27B llama-server launcher for 2x Intel Arc Pro B70 (tensor parallel, DFlash2).
rem Reads server flags from ..\config\llm-server.conf. Edit the paths below.
setlocal enabledelayedexpansion

set "LLAMA_BUILD=%~dp0..\llama.cpp\build\bin"
set "ONEDNN_BIN=C:\oneDNN-sycl\install\bin"
set "CONF_FILE=%~dp0..\config\llm-server.conf"
rem P-cores of an i7-14700K (logical CPUs 0-15). Adjust for your CPU.
set "AFFINITY=FFFF"

set "NoDefaultCurrentDirectoryInExePath="
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
if errorlevel 1 (
    echo ERROR: failed to initialize the oneAPI environment.
    exit /b 1
)
set "PATH=%ONEDNN_BIN%;%PATH%"

rem --- stock llama.cpp / oneAPI settings --------------------------------------
set "ONEAPI_DEVICE_SELECTOR=level_zero:0,1"
set "ZES_ENABLE_SYSMAN=1"
rem large single KV allocations at 200K context
set "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1"
set "GGML_SYCL_ENABLE_MKL_FA=0"
rem device-to-device copies through Level Zero (+5% decode measured)
set "GGML_SYCL_DEV2DEV_MEMCPY=1"
set "GGML_SYCL_USE_LEVEL_ZERO_API=1"

rem --- switches added by patches\llama.cpp-d81d0d8fb-dual-b70.patch ------------
rem one SYCL context per GPU over one Level Zero context: works around a oneAPI 2026.1 leak
rem (~0.5 MB per token with tensor parallel, see README). Keep this on.
set "GGML_SYCL_PER_DEVICE_CONTEXT=1"
rem tensor-parallel all-reduce: read the peer's pinned host buffer directly, join through
rem host-memory flags instead of a host wait, symmetric bf16 so both replicas stay identical
set "GGML_SYCL_TP_DIRECT=1"
set "GGML_SYCL_TP_GPUFLAG=1"
set "GGML_SYCL_TP_BF16_SYM=1"
rem DFlash2 needs the full output head on each GPU under tensor parallel
set "LLAMA_TEST_DFLASH_TP_MIRROR_HEAD=1"
rem XMX split-K flash attention for 1-8 query tokens (decode / speculative verify)
set "GGML_SYCL_FA_XMX=1"
rem oneDNN SDPA for prefill, without a host wait after each call
set "GGML_SYCL_FA_ONEDNN=1"
set "GGML_SYCL_FA_ONEDNN_SYNC=0"
rem shared activation quantization for the paired Q8 FFN up/gate
set "GGML_SYCL_Q8_GLU_FUSION=1"
set "GGML_SYCL_Q8_GLU_SPLIT_EPILOGUE=1"
rem do not feed the drafter prompt rows outside its 2048-token sliding window
set "LLAMA_DFLASH_PREFILL_TAIL=1"

if not exist "%LLAMA_BUILD%\llama-server.exe" (
    echo ERROR: llama-server.exe not found at "%LLAMA_BUILD%".
    exit /b 1
)

if not exist "%~dp0..\logs" mkdir "%~dp0..\logs"
set "LLAMA_ARG_LOG_FILE=%~dp0..\logs\llm-server.log"

rem keep both GPUs out of runtime D3 while idle (optional, see keepalive\)
if exist "%~dp0..\keepalive\gpu-keepalive.exe" (
    start "" /b /low "%~dp0..\keepalive\gpu-keepalive.exe" 10 llama-server.exe
)

set "ARGS="
for /f "usebackq eol=# delims=" %%L in ("%CONF_FILE%") do (
    if not "%%L"=="" set "ARGS=!ARGS! %%L"
)

echo %LLAMA_BUILD%\llama-server.exe !ARGS!
start "" /b /wait /high /affinity %AFFINITY% "%LLAMA_BUILD%\llama-server.exe" !ARGS!
