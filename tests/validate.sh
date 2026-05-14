#!/usr/bin/env bash
set -u

runner="./build/validation_runner"

if [[ ! -x "$runner" ]]; then
    echo "[FAIL] validation runner not found at $runner"
    echo "Run: make build"
    exit 1
fi

cases=(
    "128 128 128"
    "128 512 256"
    "256 256 128"
    "1024 256 64"
    "512 512 256"
    "1024 512 256"
    "4096 4096 4096"
)

failed=0

for case in "${cases[@]}"; do
    echo "Checking GEMM ${case}"
    $runner $case
    status=$?
    if [[ $status -eq 0 ]]; then
        echo "[PASS] ${case}"
    else
        echo "[FAIL] ${case}"
        failed=1
    fi
    echo
done

if [[ $failed -eq 0 ]]; then
    echo "Validation result: PASS"
else
    echo "Validation result: FAIL"
fi

exit $failed
