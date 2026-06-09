from __future__ import annotations

import pytest

from app.database.markdown_store import MarkdownStore
from app.exceptions import BadRequestError


class TestMarkdownStore:
    def test_save_and_read(self):
        store = MarkdownStore()
        filename = store.save("测试文档", "这是内容", tags=["test"])
        assert filename.endswith(".md")
        content = store.read(filename)
        assert content is not None
        assert "测试文档" in content
        assert "这是内容" in content

    def test_list_files(self):
        store = MarkdownStore()
        store.save("文档1", "内容1")
        store.save("文档2", "内容2")
        files = store.list_files()
        assert len(files) >= 2

    def test_delete(self):
        store = MarkdownStore()
        filename = store.save("要删除的", "内容")
        assert store.delete(filename) is True
        assert store.read(filename) is None

    def test_path_traversal_prevention(self):
        store = MarkdownStore()
        with pytest.raises(BadRequestError):
            store.read("../../../etc/passwd")
