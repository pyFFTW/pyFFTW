
function deploy_to_bintray() {
    $version_string = (iex "python -c `"import pyfftw; print(pyfftw.__version__)`"") | Out-String
    $version = $version_string -replace "`t|`n|`r",""

    if ($env:PYTHON_ARCH -eq "32") {
        $platform_suffix = "win32"
    } else {
        $platform_suffix = "win_amd64"
    }

    $python_version = $env:PYTHON_VERSION -replace '\.',''
    $filename = "pyFFTW-$version-cp$python_version-cp${python_version}m-$platform_suffix.whl"
    $filepath = ".\dist\$filename"
    $username_password = "${env:bintray_username}:${env:bintray_api_key}"

    Write-Host "Uploading: $filepath"
    Write-Host "to https://api.bintray.com/content/hgomersall/generic/PyFFTW-development-builds/$version/$filename"
    iex "curl.exe -s -T $filepath -u$username_password -H `"X-Bintray-Package:PyFFTW-development-builds`" -H `"X-Bintray-Version:$version`" -H `"X-Bintray-Publish: 1`" -H `"X-Bintray-Override: 1`" https://api.bintray.com/content/hgomersall/generic/$filename"
}

function deploy_to_pypi () {
    Write-Host "Uploading to PyPI..."
    iex "python setup.py bdist_wheel upload"
}

function main () {
    if($env:appveyor_repo_tag -eq 'True') {
        deploy_to_pypi
    }

    deploy_to_bintray
}

main
