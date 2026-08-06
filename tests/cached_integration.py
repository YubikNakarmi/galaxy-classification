from src.utils.cached_models import ModelLoader
from src.utils.cli_bootstrap import load_context


model_name = "galaxy-classification-default"

ctx = load_context(service="bentoml")
cfg = ctx.cfg

loader = ModelLoader(registered_model_name=model_name,
                     local_cache=True,cfg = cfg,cache_metadata=True,metric ="val_f1_macro")

model = loader.load_with_fallback()