"""Production entry point with startup checks."""
import platform
import sys
from pathlib import Path

import uvicorn

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings


def _detect_loop() -> str:
    """Return the best available event loop for the current platform."""
    if platform.system() == "Windows":
        return "asyncio"
    try:
        import uvloop  # noqa: F401
        return "uvloop"
    except ImportError:
        return "asyncio"


def main():
    warnings = settings.check_startup()
    if warnings:
        print("=" * 60)
        print("STARTUP WARNINGS:")
        for w in warnings:
            print(f"  [WARN] {w}")
        print("=" * 60)

    loop_choice = _detect_loop()
    http_choice = "httptools" if platform.system() != "Windows" else "h11"

    print(f"Platform: {platform.system()} | Loop: {loop_choice} | HTTP: {http_choice}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug and not settings.api_key,
        log_level=settings.log_level.lower(),
        workers=1,
        loop=loop_choice,
        http=http_choice,
    )


if __name__ == "__main__":
    main()
