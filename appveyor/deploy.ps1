
function deploy_to_bintray() {
    $version_string = (iex "python -c `"import pyfftw; print(pyfftw.__version__)`"") | Out-String
    $version = $version_string -replace "`t|`n|`r",""
    $short_version = [string]$version.Split("+")[0]

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
    Write-Host "to https://api.bintray.com/content/hgomersall/generic/PyFFTW-development-builds/$short_version/$filename"
    iex "curl.exe -s -T $filepath -u$username_password -H `"X-Bintray-Package:PyFFTW-development-builds`" -H `"X-Bintray-Version:$short_version`" -H `"X-Bintray-Publish: 1`" -H `"X-Bintray-Override: 1`" https://api.bintray.com/content/hgomersall/generic/$filename"
}

function main () {
    deploy_to_bintray
}

main
