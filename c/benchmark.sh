#!/bin/bash
levels=("-O0" "-O1" "-O2" "-O3" "-Ofast")
echo "Optimization Level | Inference Time (seconds)"
echo "-------------------|------------------------"

for opt in "${levels[@]}"; do
    gcc $opt -Wall -Wextra -o galaxy_bench main.c nn.c math.c
    start=$(date +%s.%N)
    ./galaxy_bench > /dev/null 2>&1
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc)
    printf "%-18s | %s\n" "$opt" "$runtime"
done
rm galaxy_bench
