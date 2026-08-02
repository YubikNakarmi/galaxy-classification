"""
Tests for ModelLoader.

Run with:
    pytest tests/test_cache.py -v

Strategy:
- MlflowClient is mocked entirely (patch.object on the imported module reference --
  NOT a dotted string path -- to avoid the pytest --import-mode=importlib issue where
  your own src package modules don't get bound as parent-package attributes).
- mlflow.pytorch.load_model is patched via a plain string path, since that's a real
  third-party global module and doesn't have the same import-mode problem.
- _init_model is monkeypatched to return a tiny real nn.Module.
- cfg is a minimal fake object satisfying cfg.env.device (duck-typed; RootConfig is
  only used as a type hint, not enforced at runtime).

Adjust the import below to match your actual module path.
"""
import hashlib
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.utils import cached_models as cache_module
from src.utils.cached_models import ModelLoader


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)


class FakeEnv:
    device = "cpu"


class FakeCfg:
    env = FakeEnv()


@pytest.fixture
def loader(tmp_path, monkeypatch):
    with patch.object(cache_module, "MlflowClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        ml = ModelLoader(
            registered_model_name="galaxy_classifier",
            cfg=FakeCfg(),
            local_cache=True,
            local_cache_dir=tmp_path / "cache",
            num_classes=5,
            cache_metadata=True,
            model_top_n=5,
            metadata_save_path=tmp_path / "metadata.json",
            variant="Default",
            checksum=True,
        
        )
    monkeypatch.setattr(ml, "_init_model", lambda: TinyModel())
    ml.client = mock_client
    return ml


def make_mock_run(run_id, f1=0.8, val_f1=0.75):
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = {"val_f1_macro": val_f1, "f1_macro": f1}
    return run


def _candidate(run_id, f1, loader):
    from src.utils.schemas import CandidateMetadata
    from datetime import datetime, timezone
    return CandidateMetadata(
        rank=None, run_id=run_id, model_name=loader.registered_model_name,
        experiment_name=loader.registered_model_name, f1_macro=f1, val_f1_macro=f1,
        num_classes=loader.num_classes, cache_file=str(loader._cache_path(run_id)),
        cached_at=datetime.now(timezone.utc).isoformat(), save_path=str(loader.local_cache_dir),
    )


# ---------- constructor / cfg wiring ----------

def test_loader_reads_device_from_cfg(loader):
    assert loader.device == "cpu"


def test_loader_top_n_from_model_top_n_kwarg(tmp_path):
    with patch.object(cache_module, "MlflowClient"):
        ml = ModelLoader(
            registered_model_name="x", cfg=FakeCfg(), local_cache=False,
            cache_metadata=False, model_top_n=7,
        )
    assert ml.top_n == 7


# ---------- compute_checksum / compare_checksum ----------

def test_compute_checksum_matches_hashlib(loader):
    run_id = "run1"
    cache_path = loader._cache_path(run_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"fake model weights data")

    expected = hashlib.sha256(b"fake model weights data").hexdigest()
    assert loader.compute_checksum(run_id) == expected


def test_compute_checksum_raises_without_cache_dir():
    with patch.object(cache_module, "MlflowClient"):
        ml = ModelLoader(registered_model_name="x", cfg=FakeCfg(), local_cache=False,
                          local_cache_dir=None, cache_metadata=False)
    with pytest.raises(RuntimeError):
        ml.compute_checksum("run1")


def test_compare_checksum_returns_false_if_no_checksum_file(loader):
    cache_path = loader._cache_path("run1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"data")
    assert loader.compare_checksum("run1") is False


def test_compare_checksum_true_when_matching(loader):
    run_id = "run1"
    cache_path = loader._cache_path(run_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"data")
    checksum = loader.compute_checksum(run_id)
    loader._save_checksum(run_id, checksum)
    assert loader.compare_checksum(run_id) is True


def test_compare_checksum_false_when_tampered(loader):
    run_id = "run1"
    cache_path = loader._cache_path(run_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"original data")
    checksum = loader.compute_checksum(run_id)
    loader._save_checksum(run_id, checksum)
    cache_path.write_bytes(b"tampered data")
    assert loader.compare_checksum(run_id) is False


# ---------- check_cache ----------

def test_check_cache_miss_pulls_from_mlflow_and_caches(loader):
    run_id = "run1"
    with patch("mlflow.pytorch.load_model", return_value=TinyModel()) as mock_load:
        model = loader.check_cache(run_id)

    mock_load.assert_called_once_with(f"runs:/{run_id}/model")
    assert isinstance(model, TinyModel)
    assert loader._cache_path(run_id).exists()
    assert loader._checksum_path(run_id).exists()


def test_check_cache_uses_cfg_device_for_map_location(loader, monkeypatch):
    run_id = "run1"
    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        loader.check_cache(run_id)  # populate cache

    captured = {}
    real_load = torch.load

    def spy_load(*args, **kwargs):
        captured["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    loader.check_cache(run_id)  # cache hit path
    assert captured["map_location"] == "cpu"


def test_check_cache_hit_loads_from_disk_no_mlflow_call(loader):
    run_id = "run1"
    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        loader.check_cache(run_id)

    with patch("mlflow.pytorch.load_model") as mock_load:
        model = loader.check_cache(run_id)
        mock_load.assert_not_called()

    assert isinstance(model, TinyModel)


def test_check_cache_corrupt_checksum_falls_through_to_mlflow(loader):
    run_id = "run1"
    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        loader.check_cache(run_id)

    loader._cache_path(run_id).write_bytes(b"corrupted garbage")

    with patch("mlflow.pytorch.load_model", return_value=TinyModel()) as mock_load:
        model = loader.check_cache(run_id)
        mock_load.assert_called_once()

    assert isinstance(model, TinyModel)
    assert loader.compare_checksum(run_id) is True


def test_check_cache_reraises_on_mlflow_failure(loader):
    with patch("mlflow.pytorch.load_model", side_effect=RuntimeError("mlflow down")):
        with pytest.raises(RuntimeError):
            loader.check_cache("run1")


# ---------- _best_run ----------

def test_best_run_returns_none_if_experiment_missing(loader):
    loader.client.get_experiment_by_name.return_value = None
    assert loader._best_run("galaxy_classifier") is None


def test_best_run_returns_none_if_no_runs(loader):
    exp = MagicMock()
    exp.experiment_id = "exp1"
    loader.client.get_experiment_by_name.return_value = exp
    loader.client.search_runs.return_value = []
    assert loader._best_run("galaxy_classifier") is None


def test_best_run_returns_runs_sorted_by_metric(loader):
    exp = MagicMock()
    exp.experiment_id = "exp1"
    loader.client.get_experiment_by_name.return_value = exp
    runs = [make_mock_run("run1"), make_mock_run("run2")]
    loader.client.search_runs.return_value = runs

    result = loader._best_run("galaxy_classifier", metric="val_f1_macro")

    assert result == runs
    _, kwargs = loader.client.search_runs.call_args
    assert kwargs["order_by"] == ["metrics.val_f1_macro DESC"]
    assert kwargs["max_results"] == 5


# ---------- sync_metadata_with_mlflow ----------

def test_sync_metadata_writes_all_top_n_candidates(loader):
    exp = MagicMock()
    exp.experiment_id = "exp1"
    loader.client.get_experiment_by_name.return_value = exp
    runs = [make_mock_run("run_a", f1=0.9), make_mock_run("run_b", f1=0.7)]
    loader.client.search_runs.return_value = runs

    synced = loader.sync_metadata_with_mlflow()

    assert {c.run_id for c in synced} == {"run_a", "run_b"}
    meta = loader.metadata._load()
    assert {c.run_id for c in meta.Default.candidates} == {"run_a", "run_b"}


def test_sync_metadata_returns_empty_when_cache_metadata_false(tmp_path):
    with patch.object(cache_module, "MlflowClient"):
        ml = ModelLoader(registered_model_name="x", cfg=FakeCfg(), local_cache=False,
                          cache_metadata=False)
    assert ml.sync_metadata_with_mlflow() == []


def test_sync_metadata_returns_empty_when_no_runs_found(loader):
    loader.client.get_experiment_by_name.return_value = None
    assert loader.sync_metadata_with_mlflow() == []


# ---------- load_with_fallback ----------
# NOTE: requires the load_with_fallback -> sync_metadata_with_mlflow() call-site fix
# (no top_n kwarg) described at the top of this file.

def test_load_with_fallback_success_syncs_metadata_and_loads_best(loader):
    exp = MagicMock()
    exp.experiment_id = "exp1"
    loader.client.get_experiment_by_name.return_value = exp
    runs = [make_mock_run("run_a", f1=0.9), make_mock_run("run_b", f1=0.7)]
    loader.client.search_runs.return_value = runs

    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        model = loader.load_with_fallback()

    assert isinstance(model, TinyModel)
    meta = loader.metadata._load()
    assert {c.run_id for c in meta.Default.candidates} == {"run_a", "run_b"}


def test_load_with_fallback_skips_broken_run_and_uses_next(loader):
    exp = MagicMock()
    exp.experiment_id = "exp1"
    loader.client.get_experiment_by_name.return_value = exp
    runs = [make_mock_run("run_bad", f1=0.9), make_mock_run("run_good", f1=0.7)]
    loader.client.search_runs.return_value = runs

    call_count = {"n": 0}

    def load_model_side_effect(uri):
        call_count["n"] += 1
        if "run_bad" in uri:
            raise RuntimeError("corrupt artifact")
        return TinyModel()

    with patch("mlflow.pytorch.load_model", side_effect=load_model_side_effect):
        model = loader.load_with_fallback()

    assert isinstance(model, TinyModel)
    assert call_count["n"] == 2


def test_load_with_fallback_raises_if_all_runs_fail(loader):
    exp = MagicMock()
    exp.experiment_id = "exp1"
    loader.client.get_experiment_by_name.return_value = exp
    loader.client.search_runs.return_value = [make_mock_run("run1"), make_mock_run("run2")]

    with patch("mlflow.pytorch.load_model", side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError, match="Failed to load any model"):
            loader.load_with_fallback()


def test_load_with_fallback_uses_local_metadata_when_mlflow_unreachable(loader):
    loader.metadata.update_metadata(_candidate("run_local", f1=0.6, loader=loader), variant="Default")
    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        loader.check_cache("run_local")

    # sync_metadata_with_mlflow calls _best_run -> get_experiment_by_name; make THAT raise
    loader.client.get_experiment_by_name.side_effect = ConnectionError("mlflow unreachable")

    with patch("mlflow.pytorch.load_model") as mock_load:
        model = loader.load_with_fallback()
        mock_load.assert_not_called()

    assert isinstance(model, TinyModel)


# ---------- load_champion_model ----------

def test_load_champion_model_uses_alias(loader):
    version = MagicMock()
    version.run_id = "champion_run"
    loader.client.get_model_version_by_alias.return_value = version
    loader.client.get_run.return_value = make_mock_run("champion_run", f1=0.95)

    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        model = loader.load_champion_model()

    assert isinstance(model, TinyModel)
    meta = loader.metadata._load()
    assert any(c.run_id == "champion_run" for c in meta.Default.candidates)


def test_load_champion_model_falls_back_to_local_metadata_on_alias_failure(loader):
    loader.metadata.update_metadata(_candidate("run_prev", f1=0.5, loader=loader), variant="Default")
    loader.client.get_model_version_by_alias.side_effect = Exception("no champion alias set")
    loader.client.get_run.return_value = make_mock_run("run_prev", f1=0.5)

    with patch("mlflow.pytorch.load_model", return_value=TinyModel()):
        model = loader.load_champion_model()

    assert isinstance(model, TinyModel)


def test_load_champion_model_returns_none_on_total_failure(loader):
    loader.client.get_model_version_by_alias.side_effect = Exception("no alias")
    loader.client.get_run.side_effect = Exception("no run either")
    result = loader.load_champion_model()
    assert result is None