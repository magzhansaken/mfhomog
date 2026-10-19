#!/bin/bash
# Download the external code and data used in sections 3.2 and 4.8 at the exact commits used in the paper.
# Usage: EXT_DIR=external bash scripts/get_external.sh   (then run the scripts with the same EXT_DIR)
set -e
EXT_DIR=${EXT_DIR:-external}; mkdir -p "$EXT_DIR"; cd "$EXT_DIR"
get() { [ -d "$2" ] || git clone -q "$1" "$2"; (cd "$2" && git checkout -q "$3"); echo "$2 @ $3"; }
get https://github.com/Rose-STL-Lab/MFRNP.git mfrnp 614590a4ae7e0c9f2d0cc194640b0c41b7116a70                     # MIT
get https://github.com/elejeune11/Mechanical-MNIST-Transfer-Learning.git mm_Mechanical-MNIST-Transfer-Learning 600949a66077f774d81d148969fc71016412e3bf   # MIT code; data CC BY-SA 4.0 (OpenBU hdl 2144/38693)
get https://github.com/roysondsouza316/natural-fibre-rve-dataset-surrogate.git nf 5a3c7e2648f3cce988a75750e0691a8d570571a4   # code MIT, data CC BY 4.0
