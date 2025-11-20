#!/usr/bin/env python3
"""Small helper to run LoRA_FineTuning_Sentiment.py with --dry_run and assert it prints dataset sizes.

This is useful for quick local checks or CI smoke tests. It sets KMP_DUPLICATE_LIB_OK on macOS to avoid OpenMP runtime aborts
on some environments.
"""
import os
import subprocess
import sys


def main():
    cmd = [sys.executable, "LoRA_FineTuning_Sentiment.py", "--dry_run", "--max_train_samples", "8", "--max_eval_samples", "4"]
    env = os.environ.copy()
    # macOS workaround for duplicate OpenMP runtimes when necessary
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print("Running dry-run smoke check:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        print("Error: script not found or Python not available:", e)
        sys.exit(2)

    print("Exit code:", res.returncode)
    print("Stdout:\n", res.stdout)
    print("Stderr:\n", res.stderr)

    if res.returncode != 0:
        print("Dry-run failed (non-zero exit). This may be due to missing dependencies like 'transformers' in the environment.")
        sys.exit(res.returncode)

    # Expect the script to print a [DRY RUN] or Train size line
    if "[DRY RUN]" in res.stdout or "Train size:" in res.stdout:
        print("Dry-run smoke check passed.")
        sys.exit(0)
    else:
        print("Dry-run did not produce expected output. Check environment or script changes.")
        sys.exit(3)


if __name__ == "__main__":
    main()
