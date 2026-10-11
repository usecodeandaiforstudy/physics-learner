# -*- coding: utf-8 -*-
"""**隐私/密钥扫描器**（2026-10-02，第 214 条）——命中即**非零退出** ✗。

## 为什么有它
发布副本曾把 `a_network/multi_seed_verify.log`（含 **Windows 用户名**
`C:\\Users\\<name>\\…`，94 处 ✗）带进发布目录 ⟹ 事后才发现 ✗。
本脚本把这件事变成**常设检查** ✓：发布前/发布后各跑一次，命中即报错 ✓

## 查什么（预登记，不含"见机行事" ✓）
  1. **Windows 用户目录**：`C:\\Users\\<名字>`（排除 `C:\\Users\\Public` 等通用名 ✓）
  2. **macOS / Linux 家目录**：`/Users/<名字>`、`/home/<名字>`
  3. **云厂商/服务密钥**：`sk-…`(OpenAI 等)、`AKIA…`(AWS)、`ghp_…`(GitHub)、
     `AIza…`(Google)、`xox…`(Slack)、`-----BEGIN … PRIVATE KEY-----`
  4. **JWT**：`eyJ….….…`
  5. **邮箱**、**中国大陆手机号**、**微信/QQ 号**
  6. **凭据类文件名**：`.env`、`*.pem`、`*.key`、`id_rsa`、`credentials`、
     `.netrc`、`cookies`
  7. **本机/环境路径痕迹**：`AppData\\Local`、`WindowsApps`、`Program Files\\WindowsApps`

## 例外（白名单）
  · 本脚本自身、以及**文档中"用于说明扫描规则"的引用** ✓（`ALLOW` 显式列出 ✓）
  · 二进制文件（`.pt`/`.png`/…）默认跳过（避免误报 ✓；`--binaries` 可强制扫 ✓）

用法：
    python3.13 tools/privacy_scan.py                 # 扫当前项目
    python3.13 tools/privacy_scan.py <目录>           # 扫指定目录（如发布副本）
    python3.13 tools/privacy_scan.py <目录> --binaries
退出码：**0 = 干净** ✓；**1 = 命中敏感内容** ✗；**2 = 用法错误**
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# ---- 预登记规则 ✓ ----
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Windows 用户目录",
     re.compile(r"[Cc]:[\\/]{1,2}Users[\\/]{1,2}(?!Public\b|Default\b|All Users\b)"
                r"[A-Za-z0-9._-]{2,}")),
    ("macOS/Linux 家目录",
     re.compile(r"/(?:Users|home)/(?!runner\b|user\b|shared\b)"
                r"[A-Za-z0-9._-]{2,}")),
    ("OpenAI 类密钥", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}")),
    ("Slack Token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("私钥文件头", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.")),
    ("邮箱", re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")),
    # **注意误报**：巨型负浮点数（如 `-13058933959.42`）会撞手机号正则 ✗
    #   ⟹ 前后排除数字/小数点/负号 ✓
    ("中国大陆手机号", re.compile(r"(?<![\d.\-])1[3-9]\d{9}(?![\d.])")),
    ("微信/QQ 号", re.compile(r"(?:微信|QQ|qq)\s*[:：]\s*\d{5,}")),
    ("环境路径痕迹",
     re.compile(r"AppData[\\/]{1,2}Local|Program Files[\\/]{1,2}WindowsApps")),
]
BAD_NAMES = re.compile(
    r"^\.env|\.env$|\.pem$|\.key$|\.pfx$|\.p12$|^id_rsa|^id_ed25519|"
    r"^credentials|\.netrc$|^cookies|\.kdbx$|\.ovpn$")
# ---- 白名单：文档里"说明扫描规则 / 记录历史路径"的引用 ✓ ----
#   **中英双语**：`X.md` 与 `X_cn.md` 都需放行 ✓（同一内容的两版 ✓）
ALLOW: dict[str, tuple[str, ...]] = {
    "RELEASE_CHECKLIST.md": ("Windows 用户目录", "环境路径痕迹", "邮箱"),
    "RELEASE_CHECKLIST_cn.md": ("Windows 用户目录", "环境路径痕迹", "邮箱"),
    "HANDOVER.md": ("Windows 用户目录", "环境路径痕迹", "邮箱"),
    "HANDOVER_cn.md": ("Windows 用户目录", "环境路径痕迹", "邮箱"),
    "README.md": ("Windows 用户目录",),
    "README_cn.md": ("Windows 用户目录",),
    "CLAIMS_AND_BOUNDARIES.md": ("Windows 用户目录",),
    "CLAIMS_AND_BOUNDARIES_cn.md": ("Windows 用户目录",),
    "tools/privacy_scan.py": tuple(n for n, _ in RULES),
    "tools/make_release_copy.py": ("Windows 用户目录", "邮箱"),
}
TEXT_SUFFIX = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml",
               ".toml", ".ini", ".cfg", ".ps1", ".sh", ".log", ".tex", ".cff"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".trae", ".pytest_cache"}


def scan(root: Path, binaries: bool = False) -> list[tuple[str, int, str, str]]:
    hits: list[tuple[str, int, str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() or any(d in SKIP_DIRS for d in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if BAD_NAMES.search(p.name):
            hits.append((rel, 0, "凭据类文件名", p.name))
            continue
        if p.suffix.lower() not in TEXT_SUFFIX and not binaries:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        allow = ALLOW.get(rel) or ALLOW.get(p.name) or ()
        for name, pat in RULES:
            if name in allow:
                continue
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append((rel, line, name, m.group(0)[:60]))
    return hits


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]).resolve() if args else Path(__file__).resolve().parent.parent
    binaries = "--binaries" in sys.argv
    print("=" * 74)
    print("隐私/密钥扫描：%s" % root)
    print("=" * 74)
    if not root.exists():
        print("**目录不存在** ✗")
        return 2
    hits = scan(root, binaries)
    if not hits:
        print("**干净** ✓ —— 预登记的 %d 条规则全部 0 命中 ✓" % len(RULES))
        return 0
    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for rel, line, name, snip in hits:
        by_file.setdefault(rel, []).append((line, name, snip))
    print("**命中 %d 处，涉及 %d 个文件** ✗\n" % (len(hits), len(by_file)))
    for rel, items in sorted(by_file.items()):
        print("  %s（%d 处）" % (rel, len(items)))
        for line, name, snip in items[:4]:
            print("     L%-5d [%s] %s" % (line, name, snip))
        if len(items) > 4:
            print("     … 另有 %d 处" % (len(items) - 4))
    print("\n**处置建议**：① 从发布内容中排除该文件 ✓（`*.log` 等构建残留一律不发 ✓）；")
    print("            ② 或把其中的路径改为相对路径/占位符 ✓；"
          "③ 确认非敏感后再入 `ALLOW` ✓")
    return 1


if __name__ == "__main__":
    sys.exit(main())
