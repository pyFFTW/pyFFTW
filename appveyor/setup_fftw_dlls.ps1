

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

    iex "dir"
    Write-Host $env:WIN_SDK_ROOT
    Write-Host $env:WINDOWS_SDK_VERSION
    iex "dir $env:WIN_SDK_ROOT\$env:WINDOWS_SDK_VERSION"

    iex "7z.exe e $zip_destination -opyfftw *.dll"    
    iex "7z.exe e $zip_destination -opyfftw *.def"

    iex "lib /def:pyfftw\libfftw3-3.def"
    iex "lib /def:pyfftw\libfftw3f-3.def"
    iex "lib /def:pyfftw\libfftw3l-3.def"

}

SetupFFTWDLLs
