# folio installer for Windows.
#
# Usage:
#   powershell -c "irm https://raw.githubusercontent.com/kmoskwiak/folio/main/install.ps1 | iex"
#
# Env knobs:
#   $env:FOLIO_INSTALL   target directory (default: $HOME\.folio\bin)
#   $env:FOLIO_VERSION   release tag to install (default: latest)
#   ($env:BRAIN_INSTALL / $env:BRAIN_VERSION are still honoured as fallbacks.)

$ErrorActionPreference = "Stop"

$Repo       = "kmoskwiak/folio"
$BinName    = "folio.exe"
$InstallDir = if ($env:FOLIO_INSTALL) { $env:FOLIO_INSTALL } elseif ($env:BRAIN_INSTALL) { $env:BRAIN_INSTALL } else { Join-Path $HOME ".folio\bin" }
$Version    = if ($env:FOLIO_VERSION) { $env:FOLIO_VERSION } elseif ($env:BRAIN_VERSION) { $env:BRAIN_VERSION } else { "latest" }

# ---------- detect arch ----------
switch ($env:PROCESSOR_ARCHITECTURE) {
  "AMD64" { $Arch = "x64" }
  "ARM64" {
    # Today's release matrix only ships an x64 Windows binary.
    # Windows on ARM64 runs x64 binaries via emulation, so this still works.
    $Arch = "x64"
    Write-Warning "Windows ARM64 detected — installing x64 binary (runs via emulation)."
  }
  default { throw "unsupported architecture: $env:PROCESSOR_ARCHITECTURE" }
}

$Asset = "folio-windows-$Arch.exe"
$Url = if ($Version -eq "latest") {
  "https://github.com/$Repo/releases/latest/download/$Asset"
} else {
  "https://github.com/$Repo/releases/download/$Version/$Asset"
}

# ---------- download ----------
if (-not (Test-Path $InstallDir)) {
  New-Item -ItemType Directory -Path $InstallDir | Out-Null
}
$Target = Join-Path $InstallDir $BinName
$Tmp    = "$Target.tmp"

Write-Host "→ downloading $Asset ($Version)" -ForegroundColor White
Write-Host "  url: $Url"
try {
  Invoke-WebRequest -Uri $Url -OutFile $Tmp -UseBasicParsing
} catch {
  throw "download failed. Check that release '$Version' exists at https://github.com/$Repo/releases"
}

if (Test-Path $Target) { Remove-Item $Target -Force }
Move-Item $Tmp $Target

# ---------- verify ----------
Write-Host "→ verifying" -ForegroundColor White
$versionLine = & $Target --version 2>$null | Select-Object -First 1
if (-not $versionLine) { throw "the binary was downloaded to $Target but failed to run." }

# ---------- summary ----------
Write-Host ""
Write-Host "✔ installed: $versionLine" -ForegroundColor Green
Write-Host "  location: $Target"
Write-Host ""

# ---------- PATH check ----------
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$onPath   = ($UserPath -split ';' | ForEach-Object { $_.TrimEnd('\') }) -contains $InstallDir.TrimEnd('\')
if (-not $onPath) {
  Write-Warning "$InstallDir is not on your PATH."
  Write-Host "  Add it permanently for the current user:" -ForegroundColor Yellow
  Write-Host "    [Environment]::SetEnvironmentVariable('Path', `"$UserPath;$InstallDir`", 'User')" -ForegroundColor Yellow
  Write-Host "  Or for this session only:" -ForegroundColor Yellow
  Write-Host "    `$env:Path += `";$InstallDir`"" -ForegroundColor Yellow
  Write-Host ""
}

Write-Host "Get started:" -ForegroundColor White
Write-Host "  folio --help                         # see all flags"
Write-Host "  folio                                # serve on :3000"
Write-Host "  folio --vault-dir `"$HOME\my-notes`"    # custom vault"
Write-Host ""
Write-Host "Then open: http://localhost:3000" -ForegroundColor Blue
