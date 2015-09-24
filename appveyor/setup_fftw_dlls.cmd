@ECHO OFF
call "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Setup\WindowsSdkVer.exe" -q -version:%WINDOWS_SDK_VERSION%
call "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Bin\SetEnv.cmd"

ECHO Downloading DLL files
IF %PYTHON_ARCH% == 64 (
    call appveyor DownloadFile "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll64.zip"
    SET MACHINE=X64
    SET FFTW_DLL_FILENAME=fftw-3.3.4-dll64.zip
) ELSE (
    call appveyor DownloadFile "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll32.zip"
    SET MACHINE=X86
    SET FFTW_DLL_FILENAME=fftw-3.3.4-dll32.zip
)
call ls pyfftw
ECHO Extracting DLLs from %FFTW_DLL_FILENAME%
call 7z.exe e %FFTW_DLL_FILENAME% -opyfftw *.dll
call 7z.exe e %FFTW_DLL_FILENAME% -opyfftw *.def
ECHO Generating def files
call lib /machine:%MACHINE% /def:pyfftw\libfftw3-3.def
call lib /machine:%MACHINE% /def:pyfftw\libfftw3f-3.def
call lib /machine:%MACHINE% /def:pyfftw\libfftw3l-3.def

