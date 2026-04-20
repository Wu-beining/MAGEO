from pathlib import Path

from config import base


def test_get_default_config_falls_back_to_example(monkeypatch):
    monkeypatch.delenv("EA_DEFAULT_CONFIG", raising=False)

    example_path = base.CONFIG_DIR / "config.yaml.example"
    default_path = base._get_default_config()

    assert default_path == example_path
    assert isinstance(default_path, Path)
