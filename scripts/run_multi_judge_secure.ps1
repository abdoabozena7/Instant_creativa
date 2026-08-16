param(
    [ValidateRange(1, 8)]
    [int]$Concurrency = 3
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$temporarySecret = $false

if (-not $env:GEMINI_API_KEY -and -not $env:GOOGLE_API_KEY) {
    $env:GEMINI_API_KEY = Read-Host "Gemini API key (masked; process-only)" -MaskInput
    $temporarySecret = $true
}

try {
    & $python (Join-Path $PSScriptRoot "run_multi_judge_evaluation.py") `
        --execute `
        --concurrency $Concurrency
    exit $LASTEXITCODE
}
finally {
    if ($temporarySecret) {
        Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
    }
}
