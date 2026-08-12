"""
Tests for the Metadata class.
Run with: pytest tests/test_metadata.py -v
Adjust the import below to match your actual module path.
"""
import json
import pytest
from datetime import datetime, timezone
from dataclasses import asdict

from src.utils.cached_models import Metadata          # <-- adjust: wherever Metadata lives
from src.utils.schemas import CandidateMetadata, MetadataFile


def make_candidate(run_id, f1=0.5, rank=None):
    return CandidateMetadata(
        rank=rank,
        run_id=run_id,
        model_name="galaxy_classifier",
        experiment_name="galaxy_classifier",
        num_classes=5,
        cached_at=datetime.now(timezone.utc).isoformat(),
        save_path="/cache",
        metric={"f1_macro": f1, "val_f1_macro": f1},
    )


@pytest.fixture
def metadata(tmp_path):
    return Metadata(save_path=tmp_path / "metadata.json")


# ---------- _write_json / _load ----------

def test_write_json_creates_parent_dirs(tmp_path):
    md = Metadata(save_path=tmp_path / "nested" / "dir" / "metadata.json")
    md._write_json(md.save_path, {"hello": "world"})
    assert md.save_path.exists()
    assert json.loads(md.save_path.read_text()) == {"hello": "world"}


def test_write_json_is_atomic_no_leftover_tmp_file(tmp_path):
    md = Metadata(save_path=tmp_path / "metadata.json")
    md._write_json(md.save_path, {"a": 1})
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_overwrites_cleanly(tmp_path):
    md = Metadata(save_path=tmp_path / "metadata.json")
    md._write_json(md.save_path, {"a": 1})
    md._write_json(md.save_path, {"a": 2})
    assert json.loads(md.save_path.read_text()) == {"a": 2}


def test_load_raises_if_file_missing(metadata):
    with pytest.raises(FileNotFoundError):
        metadata._load()


def test_load_raises_on_invalid_json(metadata):
    metadata.save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.save_path.write_text("{not valid json")
    with pytest.raises(ValueError):
        metadata._load()


def test_load_roundtrip(metadata):
    file = MetadataFile()
    file.Default.candidates.append(make_candidate("run1"))
    metadata._write_json(metadata.save_path, asdict(file))

    loaded = metadata._load()
    assert loaded.Default.candidates[0].run_id == "run1"
    assert loaded.Hubble.candidates == []


def test_load_supports_legacy_metadata_fields(metadata):
    legacy_payload = {
        "Default": {
            "latest_update": "2026-01-01T00:00:00+00:00",
            "candidates": [{
                "rank": 1,
                "run_id": "legacy-run",
                "model_name": "galaxy_classifier",
                "experiment_name": "galaxy_classifier",
                "num_classes": 5,
                "f1_macro": 0.91,
                "val_f1_macro": 0.88,
                "cache_file": "/tmp/cache/legacy-run.pt",
                "cached_at": "2026-01-01T00:00:00+00:00",
                "save_path": "/tmp/cache",
                "other_metrics": {"cache_file": "/tmp/cache/legacy-run.pt"},
                "metric": {"f1_macro": 0.91, "val_f1_macro": 0.88},
            }],
        },
        "Hubble": {"latest_update": "", "candidates": []},
        "Dirichlet": {"latest_update": "", "candidates": []},
    }
    metadata.save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.save_path.write_text(json.dumps(legacy_payload))

    loaded = metadata._load()
    candidate = loaded.Default.candidates[0]
    assert candidate.run_id == "legacy-run"
    assert candidate.metric["f1_macro"] == 0.91
    assert candidate.metric["val_f1_macro"] == 0.88
    assert not hasattr(candidate, "cache_file")


# ---------- update_metadata ----------

def test_update_metadata_creates_file_if_missing(metadata):
    metadata.update_metadata(make_candidate("run1"), variant="Default")
    loaded = metadata._load()
    assert len(loaded.Default.candidates) == 1
    assert loaded.Default.candidates[0].run_id == "run1"
    assert loaded.Default.candidates[0].rank == 1


def test_update_metadata_dedups_by_run_id(metadata):
    metadata.update_metadata(make_candidate("run1", f1=0.5), variant="Default")
    metadata.update_metadata(make_candidate("run1", f1=0.9), variant="Default")
    loaded = metadata._load()
    assert len(loaded.Default.candidates) == 1
    assert loaded.Default.candidates[0].metric["f1_macro"] == 0.9


def test_update_metadata_sorts_by_f1_macro_descending(metadata):
    metadata.update_metadata(make_candidate("run_low", f1=0.3), variant="Default")
    metadata.update_metadata(make_candidate("run_high", f1=0.9), variant="Default")
    metadata.update_metadata(make_candidate("run_mid", f1=0.6), variant="Default")

    loaded = metadata._load()
    run_ids = [c.run_id for c in loaded.Default.candidates]
    ranks = [c.rank for c in loaded.Default.candidates]
    assert run_ids == ["run_high", "run_mid", "run_low"]
    assert ranks == [1, 2, 3]


def test_update_metadata_handles_none_f1_macro(metadata):
    metadata.update_metadata(make_candidate("run_none", f1=None), variant="Default")
    metadata.update_metadata(make_candidate("run_scored", f1=0.5), variant="Default")

    loaded = metadata._load()
    assert loaded.Default.candidates[0].run_id == "run_scored"
    assert loaded.Default.candidates[1].run_id == "run_none"


def test_update_metadata_default_top_n_caps_at_5(metadata):
    # top_n now defaults to 5 (not unlimited) -- verify default behavior with no explicit arg
    for i in range(7):
        metadata.update_metadata(make_candidate(f"run{i}", f1=i / 10), variant="Default")

    loaded = metadata._load()
    assert len(loaded.Default.candidates) == 5
    run_ids = {c.run_id for c in loaded.Default.candidates}
    assert run_ids == {"run6", "run5", "run4", "run3", "run2"}  # top 5 by f1_macro


def test_update_metadata_explicit_top_n(metadata):
    for i in range(5):
        metadata.update_metadata(make_candidate(f"run{i}", f1=i / 10), variant="Default", top_n=3)

    loaded = metadata._load()
    assert len(loaded.Default.candidates) == 3
    run_ids = {c.run_id for c in loaded.Default.candidates}
    assert run_ids == {"run4", "run3", "run2"}


def test_update_metadata_returns_dropped_candidates(metadata):
    for i in range(4):
        metadata.update_metadata(make_candidate(f"run{i}", f1=i / 10), variant="Default", top_n=3)

    dropped = metadata.update_metadata(make_candidate("run_new", f1=0.05), variant="Default", top_n=3)
    # run_new has the lowest score, so it should be the one immediately dropped
    assert dropped is not None
    dropped_ids = {c.run_id for c in dropped}
    assert "run_new" in dropped_ids


def test_update_metadata_returns_empty_list_when_nothing_dropped(metadata):
    dropped = metadata.update_metadata(make_candidate("run1"), variant="Default", top_n=5)
    assert dropped == []


def test_update_metadata_variants_are_independent(metadata):
    metadata.update_metadata(make_candidate("run1"), variant="Default")
    metadata.update_metadata(make_candidate("run2"), variant="Hubble")

    loaded = metadata._load()
    assert len(loaded.Default.candidates) == 1
    assert len(loaded.Hubble.candidates) == 1
    assert loaded.Default.candidates[0].run_id == "run1"
    assert loaded.Hubble.candidates[0].run_id == "run2"


def test_update_metadata_returns_none_on_internal_error(metadata):
    # invalid variant -> MetadataFile.__getitem__ raises AttributeError internally,
    # which update_metadata catches and logs -- returns None, doesn't raise, doesn't write.
    result = metadata.update_metadata(make_candidate("run1"), variant="NotARealVariant")
    assert result is None
    assert not metadata.save_path.exists()


# ---------- __len__ ----------

def test_len_counts_across_all_variants(metadata):
    metadata.update_metadata(make_candidate("run1"), variant="Default")
    metadata.update_metadata(make_candidate("run2"), variant="Default")
    metadata.update_metadata(make_candidate("run3"), variant="Hubble")
    assert len(metadata) == 3