#!/bin/bash
echo " INITIATING FINAL M4 VEER-ISS BENCHMARK"
mkdir -p build/logs build/exe

echo "[*] Concatenating source files into full_benchmark.s..."
# Safely concatenating with newlines in between to prevent fused text
cat main.s > build/exe/full_benchmark.s
echo "" >> build/exe/full_benchmark.s
cat nn.s >> build/exe/full_benchmark.s
echo "" >> build/exe/full_benchmark.s
cat math.s >> build/exe/full_benchmark.s
echo "" >> build/exe/full_benchmark.s
cat data_0.s >> build/exe/full_benchmark.s

echo "[*] Compiling the concatenated file..."
riscv32-unknown-elf-gcc -march=rv32gcv -mabi=ilp32f -T veer/link.ld -o build/exe/galaxy_0_veer.exe build/exe/full_benchmark.s -nostartfiles -lm

echo "[*] Simulating Sample 0 in VeeR-iSS... (THIS WILL TAKE ~25 MINUTES. LET IT COOK.)"
time whisper --configfile veer/whisper.json --profile build/logs/galaxy_0.txt build/exe/galaxy_0_veer.exe > build/logs/galaxy_0_stdout.txt 2>&1
    
INST_COUNT=$(grep -oP 'Retired \K\d+' build/logs/galaxy_0_stdout.txt | head -n 1)
echo " Sample 0 finished! Total Instructions: $INST_COUNT"

echo "=================================================="
echo " BENCHMARK COMPLETE!"
echo "Generating M4 Family Breakdown Table..."
python3 parse_veer_profile.py
