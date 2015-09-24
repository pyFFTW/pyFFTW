@ECHO OFF
"%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Bin\SetEnv.cmd"
call powershell setup_fftw_dlls.ps1 || EXIT 1
