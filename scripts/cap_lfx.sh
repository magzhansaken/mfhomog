#!/bin/bash
# stop the low-fidelity-only set at 800 points (enough for LF:HF ratios used in the study); the driver then continues
cd "$(dirname "$0")/.."
while true; do
  n=$(wc -l < data/lfx.jsonl)
  if [ "$n" -ge 800 ]; then
    for p in $(pgrep -f "generate_data.py lfx"); do kill $p; done
    echo "lfx capped at $n $(date)" >> data/gen.log; break
  fi
  sleep 15
done
