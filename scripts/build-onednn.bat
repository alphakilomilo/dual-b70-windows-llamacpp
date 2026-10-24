@echo off
rem Builds oneDNN with the SYCL runtime for llama.cpp's GGML_SYCL_DNN=ON (tested with oneDNN main at 6e2d02e2b).
rem   git clone https://github.com/uxlfoundation/oneDNN oneDNN-sycl && cd oneDNN-sycl && <this script>
rem Set ONEDNN_INSTALL to where it should be installed (scripts\build-llama.bat and the launcher point there).
setlocal
set "ONEDNN_INSTALL=C:\oneDNN-sycl\install"

set "NoDefaultCurrentDirectoryInExePath="
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if errorlevel 1 exit /b 1
cmake -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_CXX_FLAGS="/EHsc" ^
  -DCMAKE_INSTALL_PREFIX="%ONEDNN_INSTALL%" ^
  -DDNNL_GPU_RUNTIME=SYCL ^
  -DDNNL_CPU_RUNTIME=SYCL ^
  -DDNNL_GPU_VENDOR=INTEL ^
  -DDNNL_BUILD_TESTS=OFF ^
  -DDNNL_BUILD_EXAMPLES=OFF
if errorlevel 1 exit /b 1
cmake --build build --config Release -j 8
if errorlevel 1 exit /b 1
cmake --install build --config Release
