

function Expand-ZIPFile($file, $destination)
{
    $shell = new-object -com shell.application
    $zip = $shell.NameSpace($file)
    foreach($item in $zip.items())
    {
        $shell.Namespace($destination).copyhere($item)
    }
}

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

    Expand-ZIPFile -File $zip_destination -Destination "c:\temp\fftw"
    Copy-Item c:\temp\fftw\*.dll pyfftw
    Copy-Item c:\temp\fftw\*.def pyfftw

    iex "lib /def:pyfftw\libfftw3-3.def"
    iex "lib /def:pyfftw\libfftw3f-3.def"
    iex "lib /def:pyfftw\libfftw3l-3.def"

}

SetupFFTWDLLs
