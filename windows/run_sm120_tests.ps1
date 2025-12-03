$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root "buildlogs"
if (!(Test-Path $logs)) {
    New-Item $logs -ItemType Directory | Out-Null
}

$tests = @(
    @{ Name = "enum_atoms"; Path = "tests\bin_tmem_atom_enum.exe"; Memcheck = $false },
    @{ Name = "race_alias"; Path = "tests\bin_tmem_race_alias_tests.exe"; Memcheck = $false },
    @{ Name = "layout_laws"; Path = "tests\bin_layout_laws_tests.exe"; Memcheck = $false },
    @{ Name = "watchdog"; Path = "tests\bin_watchdog_tests.exe"; Memcheck = $false },
    @{ Name = "mem_pattern"; Path = "tests\bin_mem_pattern_tests.exe"; Memcheck = $false },
    @{ Name = "copy_ops"; Path = "tests\bin_sm120_copy_ops_test.exe"; Memcheck = $true }
)

$san = $null
if ($env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH "bin\compute-sanitizer.exe"
    if (Test-Path $candidate) {
        $san = $candidate
    }
}

$results = @()
$overall = 0
Write-Host "[run] starting tests..."
Push-Location $root
try {
    foreach ($test in $tests) {
        $exe = Join-Path $root $test.Path
        if (!(Test-Path $exe)) {
            throw "Missing test binary: $($test.Path). Build tests first."
        }

        Write-Host ("[run] {0}" -f $test.Name)
        if ($test.Memcheck -and $san) {
            $logFile = Join-Path $logs ("{0}_memcheck.txt" -f $test.Name)
            & $san --tool memcheck --log-file $logFile $exe
        }
        else {
            & $exe
        }
        $code = $LASTEXITCODE
        $status = if ($code -eq 0) { "PASS" } else { "FAIL" }
        if ($status -eq "PASS") {
            $overall += 1
        }
        $results += [pscustomobject]@{
            test = $test.Name
            exit_code = $code
            status = $status
        }
        if ($code -ne 0) {
            Write-Host ("[run] {0} failed with code {1}" -f $test.Name, $code)
        }
    }
}
finally {
    Pop-Location
}

$summary = Join-Path $logs "sm120_copy_ops_test.csv"
"test,exit_code,status" | Out-File -FilePath $summary -Encoding ascii -Force
foreach ($row in $results) {
    "$($row.test),$($row.exit_code),$($row.status)" | Out-File -FilePath $summary -Append -Encoding ascii
}

Write-Host ("[run] completed {0}/{1} PASS" -f $overall, $results.Count)
Get-Content $summary | Write-Host

$failed = $results | Where-Object { $_.status -ne "PASS" }
if ($failed.Count -gt 0) {
    exit 1
}
exit 0
