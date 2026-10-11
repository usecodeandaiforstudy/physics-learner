# 开源 Stage 1 准备脚本（2026-10-02）
# **用途**：把仓库整理成可以 git init 并公开的状态。
# **请先审阅本脚本再执行**——它会做三件有"副作用"的事：
#   ① 把 `md/`（含第三方书籍笔记 → **版权风险**）移到仓库**之外**
#   ② 把根目录残渣（日志/临时脚本）移入 `_archive/`
#   ③ `git init` + 首次提交（**不改动任何数据文件的内容**，只做移动与记录）
#
# 运行前建议先看 dry-run：
#   pwsh -File tools\stage1_opensource.ps1 -DryRun
# 确认无误后：
#   pwsh -File tools\stage1_opensource.ps1

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "仓库根：$root" -ForegroundColor Cyan
if ($DryRun) { Write-Host "**DRY-RUN：只打印将执行的操作**" -ForegroundColor Yellow }

function Do($desc, $block) {
    Write-Host "  - $desc" -ForegroundColor Gray
    if (-not $DryRun) { & $block }
}

# ---------------------------------------------------------------- ① md/ 移出
# 第三方书籍笔记（Elements of Causal Inference / Information Geometry /
# Statistical Physics of Fields / Spiking Neuron Models / Path Integrals …）
# **不得随代码公开**。移出到仓库同级目录，保留不删。
$mdOut = Join-Path (Split-Path -Parent $root) "physics-learn-notes"
if (Test-Path (Join-Path $root "md")) {
    Write-Host "① 移出 md/ （第三方笔记，版权风险）→ $mdOut" -ForegroundColor Yellow
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $mdOut | Out-Null
        # 保留自写文档：information mechanics derivations / delimitation 等
        $keep = @("information mechanics derivations.md",
                  "information mechanics delimitation.md",
                  "physical_information_theory_notes.md")
        Get-ChildItem (Join-Path $root "md") -File | ForEach-Object {
            if ($keep -contains $_.Name) {
                Write-Host "    [保留] $($_.Name)（自写推导）" -ForegroundColor Green
            } else {
                Move-Item $_.FullName (Join-Path $mdOut $_.Name) -Force
            }
        }
        Write-Host "    其余 $((Get-ChildItem $mdOut -File).Count) 个文件已移出" -ForegroundColor Green
    }
} else {
    Write-Host "① md/ 不存在（已处理过？）" -ForegroundColor DarkGray
}

# ------------------------------------------------------------- ② 残渣归档
Write-Host "② 根目录残渣 → _archive/" -ForegroundColor Yellow
$archive = Join-Path $root "_archive"
Do "创建 _archive/" { New-Item -ItemType Directory -Force -Path $archive | Out-Null }
$junkPatterns = @("*.log", "io_test.*", "grid_check.*", "slow_check.*",
                  "single_sys_test.*", "encoder_test_result.txt",
                  "new_encoders_test.txt", "*_report.json",
                  "v20_runs.csv", "v27_runs.csv", "v28_runs.csv")
foreach ($pat in $junkPatterns) {
    Get-ChildItem -Path $root -Filter $pat -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            Do "归档 $($_.Name)" { Move-Item $_.FullName $archive -Force }
        }
}

# ------------------------------------------------------------- ③ git 初始化
Write-Host "③ git init + 首次提交" -ForegroundColor Yellow
if (Test-Path (Join-Path $root ".git")) {
    Write-Host "  .git 已存在，跳过 init（仅提示可提交）" -ForegroundColor DarkGray
} else {
    Do "git init -b main" { git init -b main | Out-Null }
}
Do "git add -A" { git add -A }
Do "首次提交" {
    git commit -m "Initial public snapshot: information-mechanics cross-domain design system

- core: Fisher metric / MI / IB / information Lagrangian (definition layer)
- b_network: unified front-end (knowledge encoder + requirement parser)
- a_network: GINE + GFN/SAC + variational solver (extremal action layer)
- c_network: domain decoders (MDN + CFM with exact log-density)
- sim: external verification adapters + truth self-checks
- db1-db5: five databases (AI-generated, human spot-checked; see README red lines)
- pipeline: requirement->beta calibration, decode contract, end-to-end loop
- tests: core invariant suite (12 checks)
- HANDOVER.md: credibility index incl. all negative results" | Out-Null
}

Write-Host "`n完成。后续建议：" -ForegroundColor Cyan
Write-Host "  1) git log --stat 检查首次提交内容（确认 md/ 未入版本库）"
Write-Host "  2) 若要发布：补 CITATION.cff 与 CI（pytest + run_system.py --list）"
Write-Host "  3) 数据红线已在 README.md 与 LICENSE 中声明，勿删"
if ($DryRun) { Write-Host "`n（DRY-RUN 结束：未做任何改动）" -ForegroundColor Yellow }
