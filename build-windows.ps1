[CmdletBinding()]
param(
    [string]$Python,
    [switch]$InstallDependencies
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
$applicationExe = Join-Path $distDirectory 'AnomaSlice.exe'
$workspacePrefix = $projectDirectory.TrimEnd('\') + '\'
foreach ($target in @($distDirectory, (Join-Path $projectDirectory 'build'), $buildDirectory, $applicationExe)) {
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
    & $Python -c 'import struct, sys; import PyInstaller, PySide6.QtWidgets; assert sys.version_info >= (3, 12), "Use Python 3.12 or newer"; assert struct.calcsize("P") == 8, "Use 64-bit Python"; print("Building with", sys.executable, "PyInstaller", PyInstaller.__version__)'
    if ($LASTEXITCODE -ne 0) {
        throw 'Build dependencies are unavailable. Re-run with -InstallDependencies.'
    }
    & $Python -m PyInstaller --noconfirm --clean --distpath $distDirectory --workpath $buildDirectory (Join-Path $projectDirectory 'AnomaSlice.spec')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

    if (-not (Test-Path -LiteralPath $applicationExe -PathType Leaf)) {
        throw "Expected executable was not produced: $applicationExe"
    }
    # Copy just the EXE into an empty directory before launching it. This checks
    # the same distribution the recipient receives, with no adjacent source or
    # runtime files and no reliance on the developer's conda PATH.
    $smokeDirectory = Join-Path $buildDirectory ('standalone-smoke-' + [System.Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $smokeDirectory | Out-Null
    $smokeExe = Join-Path $smokeDirectory 'AnomaSlice.exe'
    Copy-Item -LiteralPath $applicationExe -Destination $smokeExe
    $smokeStart = New-Object System.Diagnostics.ProcessStartInfo
    $smokeStart.FileName = $smokeExe
    $smokeStart.Arguments = '--smoke-test'
    $smokeStart.WorkingDirectory = $smokeDirectory
    $smokeStart.UseShellExecute = $false
    $smokeStart.CreateNoWindow = $true
    $smokeStart.EnvironmentVariables['QT_QPA_PLATFORM'] = 'offscreen'
    $smokeStart.EnvironmentVariables['PATH'] = "$env:SystemRoot\System32;$env:SystemRoot"
    $smokeStart.EnvironmentVariables.Remove('PYTHONPATH')
    $smokeStart.EnvironmentVariables.Remove('PYTHONHOME')
    $smokeTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $smokeProcess = [System.Diagnostics.Process]::Start($smokeStart)
    if (-not $smokeProcess.WaitForExit(30000)) {
        $smokeProcess.Kill()
        throw 'The frozen application startup check timed out after 30 seconds.'
    }
    if ($smokeProcess.ExitCode -ne 0) {
        throw "The frozen application startup check failed (exit $($smokeProcess.ExitCode))."
    }
    $smokeTimer.Stop()
    Remove-Item -LiteralPath $smokeExe
    Remove-Item -LiteralPath $smokeDirectory
    Write-Host 'Single EXE startup check passed (empty directory and isolated PATH).'
    $bytes = (Get-Item -LiteralPath $applicationExe).Length
    $report = [ordered]@{
        executable = $applicationExe
        singleFile = $true
        applicationMiB = [Math]::Round($bytes / 1MB, 2)
        standaloneCopyExit = $smokeProcess.ExitCode
        isolatedPath = $true
        smokeSeconds = [Math]::Round($smokeTimer.Elapsed.TotalSeconds, 2)
        sha256 = (Get-FileHash -LiteralPath $applicationExe -Algorithm SHA256).Hash
    }
    $report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $buildDirectory 'single-exe-report.json') -Encoding UTF8
    Write-Host ('Single-file application: {0} ({1:N1} MiB)' -f $applicationExe, ($bytes / 1MB))
    Write-Host 'Distribute AnomaSlice.exe directly. No adjacent runtime folder or Python installation is required.'
}
finally {
    Pop-Location
}
