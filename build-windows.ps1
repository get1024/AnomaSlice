[CmdletBinding()]
param(
    [string]$Python,
    [switch]$InstallDependencies,
    [switch]$SkipZip
)

$ErrorActionPreference = 'Stop'
$projectDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ($env:OS -ne 'Windows_NT') {
    throw 'Build this Windows application on Windows.'
}

if (-not $Python) {
    $candidates = @((Join-Path $projectDirectory '.venv\Scripts\python.exe'))
    if ($env:CONDA_PREFIX) {
        $candidates += Join-Path $env:CONDA_PREFIX 'envs\AnomaSlice\python.exe'
        $candidates += Join-Path $env:CONDA_PREFIX 'python.exe'
    }
    if ($env:CONDA_PYTHON_EXE) {
        $candidates += Join-Path (Split-Path $env:CONDA_PYTHON_EXE -Parent) 'envs\AnomaSlice\python.exe'
    }
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates += $pythonCommand.Source
    }
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python was not found. Pass -Python C:\path\to\python.exe (Python 3.12, 64-bit).'
}
$Python = (Resolve-Path -LiteralPath $Python).Path

# PyInstaller may replace its own build/output trees. Resolve and verify the
# fixed targets stay below this checkout before enabling --noconfirm.
$distDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectDirectory 'dist'))
$buildDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectDirectory 'build\windows'))
$applicationDirectory = Join-Path $distDirectory 'AnomaSlice'
$workspacePrefix = $projectDirectory.TrimEnd('\') + '\'
foreach ($target in @($distDirectory, (Join-Path $projectDirectory 'build'), $buildDirectory, $applicationDirectory)) {
    if (-not $target.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a build outside this checkout: $target"
    }
    if (Test-Path -LiteralPath $target) {
        $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
        if (-not $resolvedTarget.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing a redirected build directory: $resolvedTarget"
        }
        if ((Get-Item -LiteralPath $target).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Refusing a linked build directory: $target"
        }
    }
}

Push-Location -LiteralPath $projectDirectory
try {
    if ($InstallDependencies) {
        & $Python -m pip install -r (Join-Path $projectDirectory 'requirements-build.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Installing build dependencies failed.' }
    }
    & $Python -c 'import struct, sys; import PyInstaller, PySide6.QtWidgets, cv2, numpy; assert sys.version_info >= (3, 12), "Use Python 3.12 or newer"; assert struct.calcsize("P") == 8, "Use 64-bit Python"; print("Building with", sys.executable, "PyInstaller", PyInstaller.__version__)'
    if ($LASTEXITCODE -ne 0) {
        throw 'Build dependencies are unavailable. Re-run with -InstallDependencies.'
    }
    & $Python -m PyInstaller --noconfirm --clean --distpath $distDirectory --workpath $buildDirectory (Join-Path $projectDirectory 'AnomaSlice.spec')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

    $applicationExe = Join-Path $applicationDirectory 'AnomaSlice.exe'
    if (-not (Test-Path -LiteralPath $applicationExe -PathType Leaf)) {
        throw "Expected executable was not produced: $applicationExe"
    }
    # Verify the frozen runtime without relying on the developer's conda PATH.
    # --smoke-test opens the main window offscreen and exits its event loop.
    $smokeStart = New-Object System.Diagnostics.ProcessStartInfo
    $smokeStart.FileName = $applicationExe
    $smokeStart.Arguments = '--smoke-test'
    $smokeStart.WorkingDirectory = $applicationDirectory
    $smokeStart.UseShellExecute = $false
    $smokeStart.CreateNoWindow = $true
    $smokeStart.EnvironmentVariables['QT_QPA_PLATFORM'] = 'offscreen'
    $smokeStart.EnvironmentVariables['PATH'] = "$env:SystemRoot\System32;$env:SystemRoot"
    $smokeStart.EnvironmentVariables.Remove('PYTHONPATH')
    $smokeStart.EnvironmentVariables.Remove('PYTHONHOME')
    $smokeProcess = [System.Diagnostics.Process]::Start($smokeStart)
    if (-not $smokeProcess.WaitForExit(30000)) {
        $smokeProcess.Kill()
        throw 'The frozen application startup check timed out after 30 seconds.'
    }
    if ($smokeProcess.ExitCode -ne 0) {
        throw "The frozen application startup check failed (exit $($smokeProcess.ExitCode))."
    }
    Write-Host 'Frozen application startup check passed (isolated PATH).'
    $bytes = (Get-ChildItem -LiteralPath $applicationDirectory -Recurse -File | Measure-Object -Property Length -Sum).Sum
    Write-Host ('Portable application: {0} ({1:N1} MiB)' -f $applicationExe, ($bytes / 1MB))
    if (-not $SkipZip) {
        $archive = Join-Path $distDirectory 'AnomaSlice-Windows-x64.zip'
        Compress-Archive -LiteralPath $applicationDirectory -DestinationPath $archive -CompressionLevel Optimal -Force
        Write-Host "Distribution ZIP: $archive"
    }
    Write-Host 'Keep AnomaSlice.exe and its _internal folder together. Python is not required on the target computer.'
}
finally {
    Pop-Location
}
