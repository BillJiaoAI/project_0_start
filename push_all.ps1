# push_all.ps1
# Push all changes to GitHub. Try git push first; on failure, fall back to
# the GitHub Git Data API (build full tree -> commit -> update ref).
# Usage: powershell -ExecutionPolicy Bypass -File push_all.ps1

$ErrorActionPreference = "Stop"

$RepoPath = "D:\software2_ln\project_0_start"
$Owner    = "BillJiaoAI"
$Repo     = "project_0_start"
$Branch   = "master"
$CommitMsg = "add MLIR/matmul/vectoradd/scripts sources and profiling notes"

Set-Location $RepoPath
$env:PATH = "C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI;" + $env:PATH

# ---------- 1. Stage all changes ----------
Write-Host "`n=== [1/4] git add -A ===" -ForegroundColor Cyan
git add -A
$staged = git status --porcelain
if (-not $staged) {
    Write-Host "Working tree clean, nothing to commit."
    exit 0
}
$staged | ForEach-Object { Write-Host "  $_" }

# ---------- 2. Local commit ----------
Write-Host "`n=== [2/4] git commit ===" -ForegroundColor Cyan
git commit -m $CommitMsg | Out-Host
$localCommit = git rev-parse HEAD
$localTree   = git rev-parse "HEAD^{tree}"
Write-Host "local commit: $localCommit"
Write-Host "local tree:   $localTree"

# ---------- 3. Try git push ----------
Write-Host "`n=== [3/4] git push (attempt) ===" -ForegroundColor Cyan
$pushOk = $false
try {
    $out = git push origin $Branch 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pushOk = $true
        $out | Out-Host
    } else {
        Write-Host "git push exit code $LASTEXITCODE, falling back to API..."
    }
} catch {
    Write-Host "git push error: $_, falling back to API..."
}

if ($pushOk) {
    Write-Host "`ngit push succeeded." -ForegroundColor Green
    exit 0
}

# ---------- 4. GitHub API fallback ----------
Write-Host "`n=== [4/4] GitHub API push ===" -ForegroundColor Cyan

$remoteHead = gh api "repos/$Owner/$Repo/commits/$Branch" --jq .sha
$remoteTree = gh api "repos/$Owner/$Repo/commits/$Branch" --jq .commit.tree.sha
Write-Host "remote HEAD: $remoteHead"
Write-Host "remote tree: $remoteTree"

if ($localTree -eq $remoteTree) {
    Write-Host "Local tree matches remote, nothing to push."
    exit 0
}

# ---- 4a. Parse local tree: mode type sha path ----
Write-Host "`n--- parsing local git ls-tree ---"
$entries = git ls-tree -r HEAD | ForEach-Object {
    if ($_ -match '^(\d+)\s+(\w+)\s+(\w+)\t(.+)$') {
        [PSCustomObject]@{ Mode=$Matches[1]; Type=$Matches[2]; Sha=$Matches[3]; Path=$Matches[4] }
    }
}
Write-Host "total objects: $($entries.Count)"

# ---- 4b. Create all blobs (idempotent if already exists) ----
Write-Host "`n--- creating blobs ---"
$blobCount = 0
foreach ($e in $entries) {
    if ($e.Type -eq 'blob') {
        $diskPath = Join-Path $RepoPath ($e.Path -replace '/', '\')
        $bytes = [System.IO.File]::ReadAllBytes($diskPath)
        $b64 = [Convert]::ToBase64String($bytes)
        $body = @{ content = $b64; encoding = "base64" } | ConvertTo-Json -Compress
        $null = $body | gh api "repos/$Owner/$Repo/git/blobs" --method POST --input -
        $blobCount++
        if ($blobCount % 10 -eq 0) { Write-Host "  blobs: $blobCount / $($entries.Count)" }
    }
}
Write-Host "blobs created: $blobCount"

# ---- 4c. Build trees bottom-up ----
Write-Host "`n--- building trees ---"
$dirMap = @{}
foreach ($e in $entries) {
    $dir  = Split-Path $e.Path -Parent
    $name = Split-Path $e.Path -Leaf
    if (-not $dirMap.ContainsKey($dir)) { $dirMap[$dir] = @() }
    $dirMap[$dir] += [PSCustomObject]@{ Name=$name; Mode=$e.Mode; Type=$e.Type; Sha=$e.Sha }
}

$dirsByDepth = $dirMap.Keys | Sort-Object { ($_.ToCharArray() | Where-Object { $_ -eq '\' }).Count } -Descending

$treeShaOfDir = @{}

foreach ($dir in $dirsByDepth) {
    $treeEntries = @()
    foreach ($child in $dirMap[$dir]) {
        $treeEntries += @{ path = $child.Name; mode = $child.Mode; type = $child.Type; sha = $child.Sha }
    }
    $body = @{ tree = $treeEntries } | ConvertTo-Json -Depth 10 -Compress
    $created = ($body | gh api "repos/$Owner/$Repo/git/trees" --method POST --input -) | ConvertFrom-Json
    $treeShaOfDir[$dir] = $created.sha
    Write-Host "  tree[$dir] = $($created.sha)  ($($treeEntries.Count) entries)"
}

$topTree = $treeShaOfDir['']
Write-Host "`ntop tree: $topTree"
Write-Host "expected: $localTree"
if ($topTree -ne $localTree) {
    Write-Warning "Top tree does not match local tree! Something may be wrong."
}

# ---- 4d. Create commit ----
Write-Host "`n--- creating commit ---"
$commitBody = @{
    message = $CommitMsg
    tree    = $topTree
    parents = @($remoteHead)
} | ConvertTo-Json -Depth 10 -Compress
$newCommit = ($commitBody | gh api "repos/$Owner/$Repo/git/commits" --method POST --input -) | ConvertFrom-Json
Write-Host "new commit: $($newCommit.sha)"

# ---- 4e. Update ref (force) ----
Write-Host "`n--- updating refs/heads/$Branch ---"
$refBody = @{ sha = $newCommit.sha; force = $true } | ConvertTo-Json -Compress
$ref = ($refBody | gh api "repos/$Owner/$Repo/git/refs/heads/$Branch" --method PATCH --input -) | ConvertFrom-Json
Write-Host "$Branch -> $($ref.object.sha)"

Write-Host "`n=== API push complete ===" -ForegroundColor Green
