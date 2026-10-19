#!/bin/bash
cd "$(dirname "$0")/.."
for s in pool val test scatter ref lfx; do
  python3 scripts/generate_data.py $s data/$s.jsonl 2 >> data/gen.log 2>&1
  echo "done $s $(date)" >> data/gen.log
done
