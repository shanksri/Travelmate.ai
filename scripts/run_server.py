"""Launch the API from an absolute path, independent of the caller's cwd.

Exists because `python -m uvicorn app.main:app` resolves the `app` package
against the process's current working directory — if something launches this
project from the wrong cwd (a stale working directory in a dev-server
manager, for instance), it can silently import a same-named `app` package
from an entirely different project instead of failing loudly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
