# Real test, run once: stage C1 then stage C2, nothing changed between them (RUN_ONCE.txt checks the code hashes).
# Run from cctv-accident-ITS-system\ after: .venv-ml\Scripts\python.exe aggregate_runs.py --real-lock
$ErrorActionPreference = "Continue"
$py = ".venv-ml\Scripts\python.exe"
$R = "results\real_test"
New-Item -ItemType Directory -Force "$R\logs" | Out-Null
function Step($name, [string[]]$a) {
    "=== $name : $py $($a -join ' ')" | Tee-Object -FilePath "$R\logs\$name.log"
    & $py @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$R\logs\$name.log" -Append
    if ($LASTEXITCODE -ne 0) { "STEP FAILED: $name (exit $LASTEXITCODE)"; exit 1 }
    "STEP DONE: $name"
}
Step "C1_score_C_v2_seed0"   @("analyse_predictions.py", "--run", "C_v2_seed0", "--score-windows", "real", "--folds", "0,1,2", "--out-dir", "$R\scores\C_v2_seed0")
Step "C1_score_C_cctv_seed0" @("analyse_predictions.py", "--run", "C_cctv_seed0", "--score-windows", "real", "--folds", "0,1,2", "--out-dir", "$R\scores\C_cctv_seed0")
Step "C1_eval"               @("aggregate_runs.py", "--real-eval", "C1")
Step "C2_score_C_cctv_seed1" @("analyse_predictions.py", "--run", "C_cctv_seed1", "--score-windows", "real", "--folds", "0,1,2", "--out-dir", "$R\scores\C_cctv_seed1")
Step "C2_score_C_cctv_seed2" @("analyse_predictions.py", "--run", "C_cctv_seed2", "--score-windows", "real", "--folds", "0,1,2", "--out-dir", "$R\scores\C_cctv_seed2")
Step "C2_eval"               @("aggregate_runs.py", "--real-eval", "C2")
"ALL STAGES DONE"
