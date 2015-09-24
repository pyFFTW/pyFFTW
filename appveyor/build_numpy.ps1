#contents of build_numpy.ps1

function BuildNumpy(){
 
        Write-Host "Checking for Numpy wheel and building if necessary."
        Write-Host "pip wheel --find-links=wheelhouse --wheel-dir=wheelhouse numpy"
        iex "cmd /E:ON /V:ON /C .\\appveyor\\run_with_env.cmd pip wheel --find-links=wheelhouse --wheel-dir=wheelhouse numpy"
}

BuildNumpy
