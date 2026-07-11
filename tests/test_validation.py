from pathlib import Path

from src.training.validation import validate_setup


def test_validate_setup_allows_training_when_mlflow_is_unavailable(monkeypatch, tmp_path, caplog):
    required_paths = [tmp_path / "data"]
    create_dirs = [tmp_path / "checkpoints"]

    def raise_on_tracking_uri(_: str):
        raise RuntimeError("mlflow is down")

    monkeypatch.setattr("src.training.validation.mlflow.set_tracking_uri", raise_on_tracking_uri)

    result = validate_setup(required_paths=required_paths, mlflow_tracking_uri="http://localhost:5000", create_dirs=create_dirs)

    assert result is True
    assert (tmp_path / "data").exists()
    assert (tmp_path / "checkpoints").exists()
    assert "MLflow unavailable" in caplog.text
