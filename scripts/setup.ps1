<#
.SYNOPSIS
  One-command setup for Windows (PowerShell).

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
& $venvPython -m memtune info
Write-Host ""
Write-Host "Smoke test: recomputing the paper's results ..."
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("memtune_" + [guid]::NewGuid().ToString('N'))
$result = & $venvPython -m memtune evaluate --out $tmp
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }
$result | Select-Object -Skip 3 -First 6

Write-Host @"

Setup finished. Next time, activate the environment first:   .venv\Scripts\Activate.ps1
Then try:   python -m memtune evaluate      python -m pytest      python -m memtune --help
"@
