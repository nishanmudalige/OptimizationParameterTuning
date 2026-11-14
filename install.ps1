<#
.SYNOPSIS
  memtune installer for Windows PowerShell.

.DESCRIPTION
  Runs install.sh (the installer for every system: system report, questions, installation) with the bash that
  comes with Git for Windows. All arguments are passed on, for example:

      powershell -ExecutionPolicy Bypass -File install.ps1
      powershell -ExecutionPolicy Bypass -File install.ps1 --yes
      powershell -ExecutionPolicy Bypass -File install.ps1 --detect
      powershell -ExecutionPolicy Bypass -File install.ps1 --yes --with notebooks

  If Git for Windows is not installed, the plain PowerShell setup (scripts\setup.ps1) is used instead: it creates
  .venv with Python 3.10 - 3.12 and installs the pinned packages, without questions.

  "bash" on the PATH is not used on purpose: on machines with WSL it is the Linux bash, which would install a
  Linux environment.
#>
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$candidates = @(
    (Join-Path $env:ProgramFiles 'Git\bin\bash.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Git\bin\bash.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Git\bin\bash.exe')
)
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) { $candidates += (Join-Path (Split-Path (Split-Path $git.Source)) 'bin\bash.exe') }
$bash = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if ($bash) {
    & $bash 'install.sh' @args
    exit $LASTEXITCODE
}

Write-Host "Git for Windows (its bash) was not found; using the plain PowerShell setup."
Write-Host "For the guided installer, install Git for Windows (https://git-scm.com/download/win) and run this again."
Write-Host ""
$notebooks = ($args -join ' ') -match 'notebooks|--full'
if ($notebooks) { & (Join-Path $PSScriptRoot 'scripts\setup.ps1') -Notebooks }
else { & (Join-Path $PSScriptRoot 'scripts\setup.ps1') }
exit $LASTEXITCODE
