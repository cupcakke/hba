import sys
from pathlib import Path

import uvicorn

backend_directory = str(Path(__file__).resolve().parent / "backend")
if backend_directory not in sys.path:
    sys.path.insert(0, backend_directory)

from app.config import get_settings


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        app_dir=backend_directory,
        host="0.0.0.0",
        port=5000,
        timeout_graceful_shutdown=get_settings().server_shutdown_timeout_seconds,
    )
