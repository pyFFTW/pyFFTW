
function deploy_to_bintray() {
    $version_string = (iex "python -m pyfftw.version") | Out-String
    $version_list = $version_string.Split("\n")
    $short_version = $version_list[0]
    $version = $version_list[1]

    if ($env:PYTHON_ARCH -eq "32") {
        $platform_suffix = "x86"
    } else {
        $platform_suffix = "x86_64"
    }

    $filename = "dist/$version-py$env:PYTHON_VERSION-win-$platform_suffix.whl"

    iex "curl -T $filename -u$env:bintray_username:$env:bintray_api_key https://api.bintray.com/content/hgomersall/generic/PyFFTW-development-builds/$short_version/$filename"
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
