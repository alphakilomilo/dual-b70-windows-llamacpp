@echo off
rem Builds patched llama.cpp (SYCL, Level Zero) for 2x Intel Arc Pro B70 on Windows.
rem Edit the four paths below, then run from any directory.
setlocal

set "LLAMA_SRC=%~dp0..\llama.cpp"
set "ONEDNN_DIR=C:\oneDNN-sycl\install"
set "L0_SDK=C:\Program Files\LevelZeroSDK\1.33.1"
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"

set "NoDefaultCurrentDirectoryInExePath="
call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" > nul
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force > nul
if errorlevel 1 exit /b 1

cd /d "%LLAMA_SRC%"
cmake -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_CXX_FLAGS="/EHsc" ^
  -DCMAKE_PREFIX_PATH="%ONEDNN_DIR%" ^
  -DGGML_SYCL=ON ^
  -DGGML_SYCL_TARGET=INTEL ^
  -DGGML_SYCL_DNN=ON ^
  -DGGML_SYCL_F16=ON ^
  -DGGML_SYCL_GRAPH=ON ^
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON ^
  -DLEVEL_ZERO_INCLUDE_DIR="%L0_SDK%\include" ^
  -DLEVEL_ZERO_DEV_INCLUDE_DIR="%L0_SDK%\include" ^
  -DZE_LOADER_LIB="%L0_SDK%\lib\ze_loader.lib"
if errorlevel 1 exit /b 1
cmake --build build --config Release -j 8 --target llama-server
