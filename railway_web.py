import os
import subprocess
import sys


APP_FILE = os.getenv("STREAMLIT_APP", "app.py")


def main() -> int:
    port = os.getenv("PORT", "8501")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_FILE,
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())