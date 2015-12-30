
function deploy_to_bintray() {
    iex "activate build_env"
    #iex "conda install numpy"
    $version_string = (iex "python -m pyfftw.version") | Out-String
    $version_list = $version_string.Split("`r`n")
    $short_version = [string]$version_list[0]
    $version = [string]$version_list[2]

    if ($env:PYTHON_ARCH -eq "32") {
        $platform_suffix = "x86"
    } else {
        $platform_suffix = "amd64"
    }

    $python_version = $env:PYTHON_VERSION -replace '\.',''
    $filename = "pyFFTW-$version-cp$python_version-none-win_$platform_suffix.whl"
    $filepath = ".\dist\$filename"
    $username_password = "$env:bintray_username:$env:bintray_api_key"

    Write-Host "$filepath https://api.bintray.com/content/hgomersall/generic/PyFFTW-development-builds/$short_version/$filename"
    iex "curl.exe -T $filepath -u$username_password https://api.bintray.com/content/hgomersall/generic/PyFFTW-development-builds/$short_version/$filename"
}

function deploy_to_pypi () {
    iex "python setup.py bdist_wheel upload"
}

function main () {
    if($env:appveyor_repo_tag -eq 'True') {
        deploy_to_pypi
    }

    deploy_to_bintray
}

main
