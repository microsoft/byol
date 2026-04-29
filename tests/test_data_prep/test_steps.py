# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for path resolution and shared utilities."""

import json
import os
import tempfile

import pytest

from byol.data_prep.constants import (
    DATA_PREP_PACKAGE_DIR,
    DEFAULT_CONFIGS_DIR,
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
)
from byol.data_prep.steps.common import (
    build_user_payload,
    chunked,
    ensure_stable_ids,
    load_jsonl,
    parse_model_json,
    read_processed_ids,
    save_jsonl,
    strip_code_fences,
)


class TestPathResolution:
    """Verify path constants resolve correctly."""

    def test_repo_root_exists(self):
        assert REPO_ROOT.exists(), f"REPO_ROOT not found: {REPO_ROOT}"

    def test_package_dir_exists(self):
        assert DATA_PREP_PACKAGE_DIR.exists()

    def test_configs_dir_exists(self):
        cfg_dir = REPO_ROOT / DEFAULT_CONFIGS_DIR
        assert cfg_dir.exists(), f"Config dir not found: {cfg_dir}"


class TestJSONLIO:
    """Test JSONL read/write utilities."""

    def test_save_and_load(self, tmp_path):
        records = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        path = str(tmp_path / "test.jsonl")
        save_jsonl(records, path)
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

    def test_save_appends(self, tmp_path):
        path = str(tmp_path / "append.jsonl")
        save_jsonl([{"id": "1"}], path)
        save_jsonl([{"id": "2"}], path)
        loaded = load_jsonl(path)
        assert len(loaded) == 2

    def test_empty_save(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        save_jsonl([], path)
        assert not os.path.exists(path)


class TestChunked:
    """Test batch chunking."""

    def test_even_split(self):
        result = chunked([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = chunked([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty(self):
        assert chunked([], 5) == []


class TestJSONParsing:
    """Test robust JSON parsing."""

    def test_valid_json(self):
        result = parse_model_json('{"items": [{"id": "1", "text": "hi"}]}')
        assert result is not None
        assert result["items"][0]["id"] == "1"

    def test_code_fenced_json(self):
        raw = '```json\n{"items": []}\n```'
        result = parse_model_json(raw)
        assert result is not None
        assert result["items"] == []

    def test_invalid_json(self):
        assert parse_model_json("not json at all") is None

    def test_json_with_prefix(self):
        raw = 'Here is the result: {"items": [{"id": "1"}]}'
        result = parse_model_json(raw)
        assert result is not None

    def test_strip_code_fences(self):
        assert strip_code_fences("```json\n{}\n```") == "{}"
        assert strip_code_fences("{}") == "{}"


class TestIDs:
    """Test ID assignment and processing."""

    def test_ensure_stable_ids(self):
        data = [{"text": "a"}, {"text": "b", "id": "custom"}]
        ensure_stable_ids(data)
        assert data[0]["id"] == "0"
        assert data[1]["id"] == "custom"

    def test_read_processed_ids(self, tmp_path):
        path = str(tmp_path / "output.jsonl")
        with open(path, "w") as f:
            f.write('{"id": "abc", "text": "hello"}\n')
            f.write('{"id": "def", "text": "world"}\n')
        ids = read_processed_ids(path)
        assert ids == {"abc", "def"}

    def test_read_processed_ids_missing_file(self):
        ids = read_processed_ids("/nonexistent/path.jsonl")
        assert ids == set()


class TestBuildPayload:
    """Test user payload construction."""

    def test_payload_structure(self):
        batch = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        raw = build_user_payload(batch)
        parsed = json.loads(raw)
        assert "items" in parsed
        assert len(parsed["items"]) == 2
        assert parsed["items"][0]["id"] == "1"
        assert parsed["items"][1]["text"] == "world"
