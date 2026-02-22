import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Install Python dependencies via pip.")
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to requirements.txt"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip"
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Extra args passed to pip"
    )
    args = parser.parse_args()

    req_path = os.path.abspath(args.requirements)
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements not found: {req_path}")

    cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
    if args.upgrade:
        cmd.append("--upgrade")
    if args.extra_args:
        cmd.extend(args.extra_args)

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
