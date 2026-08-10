from src.utils.cached_models import ModelLoader
from src.cli.cli_bootstrap import load_context
import pytest

@pytest.fixture
def test_load_model()-> ModelLoader:
    model_name = "galaxy-classification-default"

    ctx = load_context(service="bentoml")
    cfg = ctx.cfg

    loader = ModelLoader(registered_model_name=model_name,local_cache_dir = "../models/cached/",
                        local_cache=True,cfg = cfg,cache_metadata=True,metric ="val_f1_macro")
    return loader



def test_load_model_with_fallback(test_load_model: ModelLoader):
    model = test_load_model.load_with_fallback()
    assert model is not None

def test_load_model_champion(test_load_model: ModelLoader):
    model = test_load_model.load_champion_model()
    assert model is not None