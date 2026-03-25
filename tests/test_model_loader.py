"""Tests for pre-trained model loader utilities."""

from pathlib import Path

from turep._model_loader import _get_cache_dir, load_model


def test_load_model_downloads_and_extracts_when_missing(monkeypatch, tmp_path):
    """Downloads zip and extracts when model directory is missing."""
    calls = {"download": 0, "extract": 0, "load_path": None}

    def fake_urlretrieve(url, filename):
        calls["download"] += 1
        Path(filename).write_bytes(b"zip-content")
        return str(filename), None

    class FakeZipFile:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, path):
            calls["extract"] += 1
            target = Path(path) / "turep_cctr"
            target.mkdir(parents=True, exist_ok=True)

    class FakeFANVI:
        @staticmethod
        def load(path):
            calls["load_path"] = Path(path)
            return "loaded-model"

    monkeypatch.setattr("turep._model_loader.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("turep._model_loader.zipfile.ZipFile", FakeZipFile)
    monkeypatch.setattr("turep._model_loader.FANVI", FakeFANVI)

    model = load_model(tmp_path)

    assert model == "loaded-model"
    assert calls["download"] == 1
    assert calls["extract"] == 1
    assert calls["load_path"] == tmp_path / "turep_cctr"


def test_load_model_skips_download_when_folder_exists(monkeypatch, tmp_path):
    """Skips download and extraction when model directory already exists."""
    model_dir = tmp_path / "turep_cctr"
    model_dir.mkdir(parents=True, exist_ok=True)
    calls = {"download": 0, "zip": 0, "load_path": None}

    def fake_urlretrieve(url, filename):
        calls["download"] += 1
        return str(filename), None

    class FakeZipFile:
        def __init__(self, filename, mode):
            calls["zip"] += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeFANVI:
        @staticmethod
        def load(path):
            calls["load_path"] = Path(path)
            return "loaded-existing-model"

    monkeypatch.setattr("turep._model_loader.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("turep._model_loader.zipfile.ZipFile", FakeZipFile)
    monkeypatch.setattr("turep._model_loader.FANVI", FakeFANVI)

    model = load_model(tmp_path)

    assert model == "loaded-existing-model"
    assert calls["download"] == 0
    assert calls["zip"] == 0
    assert calls["load_path"] == model_dir


def test_get_cache_dir_windows_uses_localappdata(monkeypatch, tmp_path):
    """Cache dir should use platformdirs.user_cache_dir output."""
    expected_cache_dir = tmp_path / "LocalAppData" / "turep"

    def fake_user_cache_dir(app_name):
        assert app_name == "turep"
        return str(expected_cache_dir)

    monkeypatch.setattr("turep._model_loader.user_cache_dir", fake_user_cache_dir)
    cache_dir = _get_cache_dir()
    assert cache_dir == expected_cache_dir


def test_get_cache_dir_macos(monkeypatch, tmp_path):
    """Cache dir should convert platformdirs return value to Path."""
    expected_cache_dir = tmp_path / "Library" / "Caches" / "turep"

    def fake_user_cache_dir(app_name):
        assert app_name == "turep"
        return str(expected_cache_dir)

    monkeypatch.setattr("turep._model_loader.user_cache_dir", fake_user_cache_dir)
    cache_dir = _get_cache_dir()
    assert cache_dir == expected_cache_dir


def test_get_cache_dir_linux_uses_xdg_cache_home(monkeypatch, tmp_path):
    """Cache dir should be delegated to platformdirs on Linux."""
    expected_cache_dir = tmp_path / "xdg_cache" / "turep"

    def fake_user_cache_dir(app_name):
        assert app_name == "turep"
        return str(expected_cache_dir)

    monkeypatch.setattr("turep._model_loader.user_cache_dir", fake_user_cache_dir)
    cache_dir = _get_cache_dir()
    assert cache_dir == expected_cache_dir


def test_get_cache_dir_linux_falls_back_to_home_cache(monkeypatch, tmp_path):
    """Cache dir should work with any string path from platformdirs."""
    expected_cache_dir = tmp_path / ".cache" / "turep"

    def fake_user_cache_dir(app_name):
        assert app_name == "turep"
        return str(expected_cache_dir)

    monkeypatch.setattr("turep._model_loader.user_cache_dir", fake_user_cache_dir)
    cache_dir = _get_cache_dir()
    assert cache_dir == expected_cache_dir


def test_load_model_uses_cache_dir_when_local_dir_not_provided(monkeypatch, tmp_path):
    """load_model should use cache dir by default when local_dir is None."""
    cache_dir = tmp_path / "cache" / "turep"
    calls = {"download": 0, "extract": 0, "load_path": None}

    def fake_get_cache_dir(app_name="turep"):
        return cache_dir

    def fake_urlretrieve(url, filename):
        calls["download"] += 1
        Path(filename).write_bytes(b"zip-content")
        return str(filename), None

    class FakeZipFile:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, path):
            calls["extract"] += 1
            target = Path(path) / "turep_cctr"
            target.mkdir(parents=True, exist_ok=True)

    class FakeFANVI:
        @staticmethod
        def load(path):
            calls["load_path"] = Path(path)
            return "loaded-default-cache-model"

    monkeypatch.setattr("turep._model_loader._get_cache_dir", fake_get_cache_dir)
    monkeypatch.setattr("turep._model_loader.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("turep._model_loader.zipfile.ZipFile", FakeZipFile)
    monkeypatch.setattr("turep._model_loader.FANVI", FakeFANVI)

    model = load_model()

    assert model == "loaded-default-cache-model"
    assert calls["download"] == 1
    assert calls["extract"] == 1
    assert calls["load_path"] == cache_dir / "turep_cctr"
