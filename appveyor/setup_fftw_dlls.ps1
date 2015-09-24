

function SetupFFTWDLLs(){
    Write-Host "Downloading and configuring the FFTW DLLs."

    if ($env:PYTHON_ARCH -eq "32"){
        $source_file = "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll32.zip"
    }
    else {
        $source_file = "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll64.zip"
    }

    $zip_destination = "fftw-3.3.4-dll.zip"
    Invoke-WebRequest $source_file -OutFile $zip_destination

    iex "7z.exe e $zip_destination -opyfftw *.dll"    
    iex "7z.exe e $zip_destination -opyfftw *.def"

    Invoke-Expression -command & "$env:WIN_SDK_ROOT\$env:WINDOWS_SDK_VERSION\Bin\lib.exe /def:pyfftw\libfftw3-3.def"
    Invoke-Expression -command & "$env:WIN_SDK_ROOT\$env:WINDOWS_SDK_VERSION\Bin\lib.exe /def:pyfftw\libfftw3f-3.def"
    Invoke-Expression -command & "$env:WIN_SDK_ROOT\$env:WINDOWS_SDK_VERSION\Bin\lib.exe /def:pyfftw\libfftw3l-3.def"

}

SetupFFTWDLLs
