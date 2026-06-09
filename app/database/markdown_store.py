from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings
from app.exceptions import BadRequestError
from app.monitoring import get_logger

logger = get_logger(__name__)


class MarkdownStore:
    def __init__(self):
        self._dir = settings.resolved_knowledge_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, title: str) -> str:
        safe = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "._-")
        if not safe:
            safe = "untitled"
        return f"{safe}.md"

    def _validate_path(self, filename: str) -> Path:
        filepath = (self._dir / filename).resolve()
        if not str(filepath).startswith(str(self._dir.resolve())):
            raise BadRequestError("Invalid file path")
        return filepath

    def save(self, title: str, content: str, tags: Optional[list[str]] = None) -> str:
        filename = self._safe_filename(title)
        filepath = self._dir / filename

        # Avoid collision: append timestamp suffix if file exists
        if filepath.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            stem = filepath.stem
            filename = f"{stem}_{ts}.md"
            filepath = self._dir / filename

        header = f"# {title}\n\n"
        if tags:
            header += f"> Tags: {', '.join(tags)}\n\n"
        header += f"> Created: {datetime.now(timezone.utc).isoformat()}\n\n---\n\n"

        full_content = header + content
        filepath.write_text(full_content, encoding="utf-8")
        return filename

    def read(self, filename: str) -> Optional[str]:
        filepath = self._validate_path(filename)
        if not filepath.exists():
            return None
        return filepath.read_text(encoding="utf-8")

    def list_files(self) -> list[dict]:
        files = []
        for f in sorted(self._dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            files.append({
                "filename": f.name,
                "title": f.stem.replace("_", " "),
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
        return files

    def delete(self, filename: str) -> bool:
        filepath = self._validate_path(filename)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def get_path(self, filename: str) -> Path:
        return self._validate_path(filename)
