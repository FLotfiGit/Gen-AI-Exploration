import os
import subprocess
import sys


def test_llama_demo_dry_run():
    """Run the LLaMA demo in dry-run mode and ensure it reports the dry-run message."""
    cmd = [sys.executable, "LoRA_Llama_Demo.py", "--dry_run"]
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Ensure the script exited successfully
    assert res.returncode == 0, f"Demo exited non-zero: {res.returncode}\nSTDERR:\n{res.stderr}"
    # Expect the dry-run marker in stdout
    assert "[DRY RUN]" in res.stdout, f"Unexpected stdout:\n{res.stdout}\nSTDERR:\n{res.stderr}"
