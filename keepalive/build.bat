@echo off
rem Builds gpu-keepalive.exe: a 256-byte fill per GPU every N seconds so an idle headless GPU does not drop to
rem runtime D3 (Windows then evicts its VRAM and the next request stalls while it is paged back in).
rem Usage: gpu-keepalive.exe <seconds> <process-name-to-watch>   (exits when that process exits)
set "NoDefaultCurrentDirectoryInExePath="
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" > nul
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force > nul
cd /d "%~dp0"
icx -fsycl -O2 gpu-keepalive.cpp -o gpu-keepalive.exe && echo BUILD_OK
