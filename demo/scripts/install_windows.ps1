# Install VocaLocate for the current Windows user.
#
#   powershell -ExecutionPolicy Bypass -File install_windows.ps1 -Wheels <dir with the two .whl> [-Models <dir>] [-Autostart]
#
# Creates %LOCALAPPDATA%\VocaLocate\venv (reusing the base Python's torch via --system-site-packages),
# installs the vocalocate-pipeline and vocalocate-demo wheels, optionally imports trained models
# (*.pt + fusion.json, e.g. a run's lake\models\vis), adds a Start-menu shortcut and, with -Autostart,
# starts the background server at sign-in. Re-running it upgrades in place.
param(
    [Parameter(Mandatory = $true)][string]$Wheels,
    [string]$Models = "",
    [string]$Python = "",
    [switch]$Autostart
)
$ErrorActionPreference = "Stop"
$homeDir = Join-Path $env:LOCALAPPDATA "VocaLocate"
$venv = Join-Path $homeDir "venv"
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }

New-Item -ItemType Directory -Force $homeDir | Out-Null
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    & $Python -m venv --system-site-packages $venv
}
$py = "$venv\Scripts\python.exe"
$pipeline = Get-ChildItem $Wheels -Filter "vocalocate_pipeline-*.whl" | Select-Object -First 1
$demo = Get-ChildItem $Wheels -Filter "vocalocate_demo-*.whl" | Select-Object -First 1
& $py -m pip install -q --disable-pip-version-check --force-reinstall --no-deps $pipeline.FullName $demo.FullName
& $py -m pip install -q --disable-pip-version-check $pipeline.FullName "$($demo.FullName)[text]"

if ($Models) {
    $dest = Join-Path $homeDir "lake\models\installed"
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $Models "*.pt") $dest -Force
    $fusion = Join-Path $Models "fusion.json"
    if (Test-Path $fusion) { Copy-Item $fusion $dest -Force }
}

& $py -m vocalocate_demo.desktop --install-shortcut
if ($Autostart) {
    & $py -c "from vocalocate_demo import desktop as d; h=d.home(); p=d.enable_autostart(h); s=d.load_settings(h); s['autostart']=True; d.save_settings(h,s); print('autostart:', p)"
}
Write-Output "VocaLocate installed in $venv"
