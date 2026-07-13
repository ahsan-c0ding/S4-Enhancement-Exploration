#!/bin/bash
# validate.sh <nn.c> <math.c>   (run from main_conv/c)
# Gate 1: per-layer MSE/MAE on sample 0 vs references (absolute correctness).
# Gate 2: final prediction on samples 0-4 must match the convolution baseline.
NN=${1:-nn.c}; MATH=${2:-math.c}
BASE=/tmp/baseline_preds.txt
# (re)generate baseline prediction cache once, from the untouched conv+old-math
if [ ! -s "$BASE" ]; then
  gcc -O2 main.c nn_count_conv.c math.c -include string.h -o /tmp/_pbase -lm 2>/dev/null
  for f in ../test_data/sample_*_img.bin; do n=$(basename $f _img.bin);
    echo "$n : $(timeout 20 /tmp/_pbase $f 2>/dev/null | grep 'Final Prediction' | sed 's/Final Prediction: //')"; done > "$BASE"
fi
echo "=== [$NN + $MATH] ==="
echo "--- Gate 1: per-layer, sample 0 (vs references) ---"
gcc -O2 test.c $NN $MATH -include string.h -o /tmp/_val -lm 2>/dev/null
/tmp/_val ../test_data/sample_0 2>/dev/null | grep -E 'PASS|FAIL|STATUS' | grep -v Processing
echo "--- Gate 2: predictions vs baseline (samples 0-4) ---"
gcc -O2 main.c $NN $MATH -include string.h -o /tmp/_pred -lm 2>/dev/null
ok=1
for f in ../test_data/sample_*_img.bin; do
  n=$(basename $f _img.bin)
  p=$(timeout 20 /tmp/_pred $f 2>/dev/null | grep 'Final Prediction' | sed 's/Final Prediction: //')
  b=$(grep "^$n " "$BASE" | sed 's/.*: //')
  if [ "$p" = "$b" ]; then echo "  $n: $p  [ok]"; else echo "  $n: '$p' != baseline '$b'  [MISMATCH]"; ok=0; fi
done
[ $ok -eq 1 ] && echo "RESULT: predictions all match baseline ✓" || echo "RESULT: PREDICTION MISMATCH ✗"
