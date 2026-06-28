from pathlib import Path

import pytest

from course_vllm.model.model_path import resolve_local_model_path


def test_resolve_local_model_path_accepts_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_local_model_path(tmp_path) == tmp_path


def test_resolve_local_model_path_uses_hf_snapshot_cache(monkeypatch):
    expected = Path("/tmp/model-snapshot")

    def fake_snapshot_download(model_id, *, local_files_only):
        assert model_id == "Qwen/Qwen3-0.6B"
        assert local_files_only is True
        return str(expected)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert resolve_local_model_path("Qwen/Qwen3-0.6B") == expected


def test_resolve_local_model_path_reports_missing_cache(monkeypatch):
    def fake_snapshot_download(model_id, *, local_files_only):
        raise RuntimeError("missing")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    with pytest.raises(FileNotFoundError):
        resolve_local_model_path("Qwen/Missing")
