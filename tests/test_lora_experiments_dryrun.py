import os
import subprocess
import sys


def test_lora_experiments_dry_run():
    """Run LoRA_Experiments.py with no args (defaults to dry-run) and ensure it reports the dry-run marker."""
    cmd = [sys.executable, "LoRA_Experiments.py"]
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Ensure the script exited successfully
    assert res.returncode == 0, f"LoRA_Experiments exited non-zero: {res.returncode}\nSTDERR:\n{res.stderr}"
    # Expect the dry-run marker in stdout
    assert "[DRY RUN]" in res.stdout, f"Unexpected stdout:\n{res.stdout}\nSTDERR:\n{res.stderr}"
