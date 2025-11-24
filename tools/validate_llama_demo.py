#!/usr/bin/env python3
"""Run the LLaMA LoRA demo in dry-run mode for quick validation.

This script mirrors `tools/validate_lora_dryrun.py` behavior but targets the LLaMA demo.
It sets `KMP_DUPLICATE_LIB_OK=TRUE` on macOS to avoid common OpenMP aborts during quick checks.
"""
import os
import subprocess
import sys


def main():
    cmd = [sys.executable, "LoRA_Llama_Demo.py", "--dry_run"]
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print("Running LLaMA demo dry-run:", " ".join(cmd))
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Exit code:", res.returncode)
    print("Stdout:\n", res.stdout)
    print("Stderr:\n", res.stderr)
    if res.returncode != 0:
        print("Dry-run failed. Ensure the script is present and runnable.")
        return res.returncode
    if "[DRY RUN]" in res.stdout:
        print("LLaMA demo dry-run passed.")
        return 0
    print("Unexpected output from demo; inspect logs.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
