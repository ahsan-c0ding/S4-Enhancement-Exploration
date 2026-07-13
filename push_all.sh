#!/usr/bin/env bash
# Run this INSIDE the repo you want to push to (e.g. ~/s4-enhancement/riscv).
# It ignores Windows/junk files, stages everything necessary, commits, and pushes.
# It does NOT delete anything from your working tree ("no clean").
set -e
cd "$(git rev-parse --show-toplevel)"
echo "[*] repo root: $(pwd)   branch: $(git rev-parse --abbrev-ref HEAD)"

# 1) keep Windows/junk & build artifacts out of the commit (files stay on disk)
cat >> .gitignore <<'EOF'
*Zone.Identifier
*:Zone.Identifier
*_just old*
* just old*
build/
*.elf
/tmp/
qemu_compile.s
full_benchmark.s
__pycache__/
EOF
sort -u .gitignore -o .gitignore

# 2) give the cheat sheet a stable name if you dropped in the versioned file
[ -f S4-Optimization-v7.xlsx ] && mv -f S4-Optimization-v7.xlsx S4-Optimization.xlsx
[ -f S4-Optimization-v6.xlsx ] && rm -f S4-Optimization-v6.xlsx
[ -f S4-Optimization-v5.xlsx ] && rm -f S4-Optimization-v5.xlsx

# 3) stage everything necessary
git add -A
echo "[*] staged changes:"; git status --short

# 4) commit + push  (review the status above; Ctrl-C now to abort)
git commit -m "Optimized RVV scan (m4 + resident constants), Remez asm exp, 10/10 validation;
plug-and-play README reproducing every S4-Optimization.xlsx value; reproducibility
scripts (run_task2 fix, rederive_static_vec, generate_refs_from_data) + final workbook"
git push origin HEAD
echo "[*] done."
