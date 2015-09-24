#contents of build_scipy.ps1

function BuildScipy(){
 
        Write-Host "Checking for Scipy wheel and building if necessary."
        Write-Host "pip wheel --find-links=wheelhouse --wheel-dir=wheelhouse scipy"
        iex "cmd /E:ON /V:ON /C .\\appveyor\\run_with_env.cmd pip wheel --find-links=wheelhouse --wheel-dir=wheelhouse scipy"
}

BuildScipy
