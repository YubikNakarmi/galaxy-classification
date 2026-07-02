# tests/integration/test_azure_live.py
import os
import pytest
import torch
from pathlib import Path

from src.utils.azure_utils import AzureBlobStorage
from src.training import CheckpointManager
from dotenv import load_dotenv

root = Path(__file__).parent.parent
load_dotenv(dotenv_path=root / ".env")

requires_azure = pytest.mark.skipif(
    not os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    reason="AZURE_STORAGE_CONNECTION_STRING not set — skipping live Azure test"
)

TEST_CONTAINER = "test-experiments"


@pytest.fixture(scope="module")
def azure_container():
    with AzureBlobStorage(container_name=TEST_CONTAINER) as blob:
        blob.create_container(name=TEST_CONTAINER)
    yield TEST_CONTAINER
    with AzureBlobStorage(container_name=TEST_CONTAINER) as blob:
        blob.client.delete_container()  # adjust to your real delete method

@requires_azure
def test_checkpoint_upload_and_restore(tmp_path, azure_container, minimal_cfg):
    import torch.nn as nn

    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)

    run_id = "live-test-run"
    minimal_cfg.storage.cloud.experiments.container = azure_container
    minimal_cfg.train.checkpoints.resume_from_checkpoint = True

    mgr = CheckpointManager(
        checkpoint_dir_local=tmp_path,
        checkpoint_dir_azure=f"{run_id}/best.pt",
        cfg=minimal_cfg,
        run_id=run_id,
    )

    mgr.save(model, optimizer, scheduler, epoch=2, val_loss=0.5,
              val_f1_macro=0.7, cfg_snapshot={"lr": 1e-3})
    url = mgr.upload_checkpoint_to_azure()
    assert url is not None

    # fresh manager simulating a new process resuming training
    model2 = nn.Linear(4, 2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=1)

    mgr2 = CheckpointManager(
        checkpoint_dir_local=tmp_path / "resume",
        checkpoint_dir_azure=f"{run_id}/best.pt",
        cfg=minimal_cfg,
        run_id=run_id,
    )
    state = mgr2.load_checkpoint(model2, optimizer2, scheduler2, device="cpu")

    assert state.start_epoch == 3
    assert state.best_val_loss == pytest.approx(0.5)

@requires_azure
def test_upload_artifact_real_files(tmp_path, azure_container, monkeypatch):
    import mlflow
    from src.train import upload_artifact  # adjust import path

    log_file = tmp_path / "training.log"
    log_file.write_text("dummy log line\n")

    with AzureBlobStorage(container_name=azure_container) as blob:
        with mlflow.start_run(run_name="live-artifact-test") as run:
            upload_artifact(blob, run.info.run_id)
            # verify tags actually got set with real URLs
            active_run = mlflow.get_run(run.info.run_id)
            assert "log_url" in active_run.data.tags