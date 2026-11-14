<#
.SYNOPSIS
  Plain setup for Windows PowerShell, without questions. The guided installer is install.ps1 / install.sh in the
  repository root; it calls this script when Git for Windows (bash) is not installed.

.DESCRIPTION
  Finds a supported Python (3.10 - 3.12) on its own, creates .venv in the repository, installs the pinned
  packages, and checks the result. Safe to run again.

.PARAMETER Notebooks
  Also install JupyterLab and the notebook tools.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Notebooks
#>
param([switch]$Notebooks)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Test-SupportedPython([string[]]$Cmd) {
    try {
        $exe = $Cmd[0]; $rest = @($Cmd | Select-Object -Skip 1)
        & $exe @rest -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$candidates = @(@('py', '-3.12'), @('py', '-3.11'), @('py', '-3.10'), @('python'), @('python3'))
$python = $null
foreach ($c in $candidates) {
    if ((Get-Command $c[0] -ErrorAction SilentlyContinue) -and (Test-SupportedPython $c)) { $python = $c; break }
}
if (-not $python) {
    Write-Host @"
No supported Python found. memtune needs Python 3.10, 3.11 or 3.12 (3.13+ is not supported by the pinned numpy).
Install one, then run this script again:
  winget install Python.Python.3.12
  or download it from https://www.python.org/downloads/windows/  (tick "Add python.exe to PATH")
"@
    exit 1
}
$exe = $python[0]; $pre = @($python | Select-Object -Skip 1)
Write-Host ("Using " + (& $exe @pre --version))

if (-not (Test-Path .venv\Scripts\python.exe)) {
    & $exe @pre -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'

$req = if ($Notebooks) { 'requirements-notebooks.txt' } else { 'requirements-lock.txt' }
Write-Host "Installing $req (about two minutes) ..."
& $venvPython -m pip install -q --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $venvPython -m pip install -q -r $req
if ($LASTEXITCODE -ne 0) { throw "Installing $req failed" }
& $venvPython -m pip install -q -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "Installing memtune failed" }

Write-Host ""
& $venvPython -m memtune demo --build-only
if ($LASTEXITCODE -ne 0) { throw "Computing the demo data failed" }
Write-Host ""
& $venvPython -m memtune doctor
if ($LASTEXITCODE -ne 0) { throw "The installation check failed" }

Write-Host @"

Installation finished. Activate the environment in every new terminal:   .venv\Scripts\Activate.ps1
Then try:
  python -m memtune reproduce        recompute the paper's results and compare them with the reference
  python -m memtune demo             open the interactive demo
  python -m memtune recommend FILE   recommend mem for your own problems (docs\USAGE.md)
  python -m pytest                   run all tests
"@
