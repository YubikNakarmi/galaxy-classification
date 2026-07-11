import os
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

from src.utils.azure_utils import AzureBlobStorage
from src.training import CheckpointManager

root = Path(__file__).parent.parent
load_dotenv(dotenv_path=root / ".env")

requires_azure = pytest.mark.skipif(
    not os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    reason="AZURE_STORAGE_CONNECTION_STRING not set — skipping live Azure test"
)

TEST_CONTAINER = "test-experiments"


@pytest.fixture
def minimal_cfg():
    return SimpleNamespace(
        storage=SimpleNamespace(
            cloud=SimpleNamespace(
                experiments=SimpleNamespace(container=TEST_CONTAINER)
            )
        ),
        train=SimpleNamespace(
            checkpoints=SimpleNamespace(
                checkpoint_resume_hard=False,
                resume_from_checkpoint=True,
            )
        ),
    )


@pytest.fixture(scope="module")
def azure_container():
    with AzureBlobStorage(container_name=TEST_CONTAINER) as blob:
        blob.create_container(name=TEST_CONTAINER)
    yield TEST_CONTAINER
    # no delete_container method exists on AzureBlobStorage — drop to the
    # underlying SDK client directly for teardown
    with AzureBlobStorage(container_name=TEST_CONTAINER) as blob:
        try:
            blob.container_client.delete_container()
        except Exception:
            pass  # best-effort cleanup


@requires_azure
@pytest.mark.integration
def test_checkpoint_upload_and_restore(tmp_path, azure_container, minimal_cfg, caplog):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)

    run_id = "live-test-run"
    filename = "best.pt"
    minimal_cfg.storage.cloud.experiments.container = azure_container
    minimal_cfg.train.checkpoints.resume_from_checkpoint = True

    mgr = CheckpointManager(
        checkpoint_dir_local=tmp_path,
        checkpoint_dir_azure="checkpoints",
        cfg=minimal_cfg,
        run_id=run_id,
    )

    mgr.save(model, optimizer, scheduler, epoch=2, val_loss=0.5,
              val_f1_macro=0.7, cfg_snapshot={"lr": 1e-3}, filename=filename)
    url = mgr.upload_checkpoint_to_azure(filename=filename)
    assert url is not None, "upload_checkpoint_to_azure returned None — check logs for the real Azure error"

    model2 = nn.Linear(4, 2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=1)

    mgr2 = CheckpointManager(
        checkpoint_dir_local=tmp_path / "resume",
        checkpoint_dir_azure="checkpoints",
        cfg=minimal_cfg,
        run_id=run_id,
    )
    state = mgr2.load_checkpoint(model2, optimizer2, scheduler2, device="cpu", filename=filename)

    assert state.start_epoch == 3, (
        f"Expected resume from epoch 3 but got {state.start_epoch} — "
        f"likely fell into _handle_missing(). Captured logs:\n{caplog.text}"
    )
    assert state.best_val_loss == pytest.approx(0.5)