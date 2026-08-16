param(
    [string]$CurrentPdf = "",
    [string]$FullPdf = "",
    [switch]$SkipFrontend,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$previousLocation = Get-Location

try {
    Set-Location $repoRoot
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m venv .venv
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv .venv
        }
        else {
            throw "Python 3 was not found on PATH."
        }
    }

    & $venvPython -m pip install -r requirements.txt

    $chunksPath = Join-Path $repoRoot "data\parsed\chunks.jsonl"
    if (-not (Test-Path -LiteralPath $chunksPath)) {
        if (-not $CurrentPdf -or -not $FullPdf) {
            throw (
                "The tracked chunk snapshot is missing. Re-run with " +
                "-CurrentPdf <current-2026.pdf> -FullPdf <full-2015.pdf>."
            )
        }
        & $venvPython scripts\build_corpus.py --current $CurrentPdf --full $FullPdf
    }

    $embeddingsPath = Join-Path $repoRoot "data\index\chunk_embeddings.npy"
    $manifestPath = Join-Path $repoRoot "data\index\index_manifest.json"
    if (-not (Test-Path -LiteralPath $embeddingsPath) -or -not (Test-Path -LiteralPath $manifestPath)) {
        Write-Host "Dense index missing; rebuilding it with the configured Ollama embedding model."
        & $venvPython scripts\build_retrieval_index.py
    }

    & $venvPython scripts\verify_runtime_artifacts.py

    if (-not $SkipFrontend) {
        Push-Location (Join-Path $repoRoot "frontend")
        try {
            npm ci
            npm run build
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipTests) {
        & $venvPython -m pytest -q
    }

    Write-Host "Bootstrap complete. Start the app with:"
    Write-Host ".\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
}
finally {
    Set-Location $previousLocation
}
