@echo off
rem oneAPI 2026.1 SYCL runtime: launching one kernel alternately on two GPUs of one SYCL context leaks host memory.
rem alt.exe <same|alt|alt_sync|alt_names|alt_sepctx|alt_threads> <launches>; shared_ze.exe <launches> tests the workaround.
set "NoDefaultCurrentDirectoryInExePath="
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "L0_SDK=C:\Program Files\LevelZeroSDK\1.33.1"
call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" > nul
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force > nul
cd /d "%~dp0"
icx -fsycl -O2 alt.cpp -o alt.exe psapi.lib || exit /b 1
icx -fsycl -O2 shared_ze.cpp -o shared_ze.exe psapi.lib -I"%L0_SDK%\include" "%L0_SDK%\lib\ze_loader.lib" || exit /b 1
set "ONEAPI_DEVICE_SELECTOR=level_zero:0,1"
alt.exe same 1000000
alt.exe alt 1000000
alt.exe alt_names 1000000
alt.exe alt_sepctx 1000000
shared_ze.exe 300000
rem the runtime's own cache trace shows a fast-cache insert on every device switch:
set "SYCL_CACHE_TRACE=7"
alt.exe alt 6
