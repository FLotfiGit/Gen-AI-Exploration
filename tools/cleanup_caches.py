#!/usr/bin/env python3
"""Cleanup helper to clear pip and Hugging Face caches.

Usage:
  python tools/cleanup_caches.py --pip --hf
  python tools/cleanup_caches.py --all

By default this script does nothing unless flags are provided. It prints what
it removed. Designed for local dev; CI should rely on cache keys instead.
"""
import argparse
import shutil
from pathlib import Path


def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pip", action="store_true", help="Clear ~/.cache/pip")
    parser.add_argument("--hf", action="store_true", help="Clear ~/.cache/huggingface")
    parser.add_argument("--all", action="store_true", help="Clear both caches")
    args = parser.parse_args()

    do_pip = args.pip or args.all
    do_hf = args.hf or args.all

    if not (do_pip or do_hf):
        print("Nothing to do. Pass --pip, --hf, or --all.")
        return 0

    home = Path.home()
    pip_cache = home / ".cache" / "pip"
    hf_cache = home / ".cache" / "huggingface"

    if do_pip:
        removed = safe_rmtree(pip_cache)
        print(f"Cleared pip cache: {pip_cache}" if removed else f"pip cache not present: {pip_cache}")
    if do_hf:
        removed = safe_rmtree(hf_cache)
        print(f"Cleared HF cache: {hf_cache}" if removed else f"HF cache not present: {hf_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
