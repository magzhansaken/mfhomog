#!/bin/bash
# continue dataset generation after the pool set: validation and low-fidelity sets first (needed for development)
cd "$(dirname "$0")/.."
while pgrep -f "generate_data.py pool" > /dev/null; do sleep 20; done
echo "pool finished $(date)" >> data/gen.log
for s in val lfx test scatter ref; do
  python3 scripts/generate_data.py $s data/$s.jsonl 2 >> data/gen.log 2>&1
  echo "done $s $(date)" >> data/gen.log
done
