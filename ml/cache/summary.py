import click
from ml.cache import cache

@cache.command(name="current-cache")
@click.option("--metadata_path",default=None,help="")
@click.option("--maximum_rows",default = 5)
@click.pass_context
def show_all(ctx,metadata_path,maximum_rows):

    from src.utils.bootstrap import load_context
    from src.cli.style import UI


    from src.utils.cached_models import Metadata
    from pathlib import Path
    from loguru import logger

    root = root = Path(__file__).parent.parent.parent

    try:


        contxt= ctx.obj or load_context(service="ml",std_out=False)  # Load ML context for pipeline commands and type hint
        ui= ctx.find_object(UI) or UI(verbose=0)
        cfg = contxt.cfg
        if not metadata_path:
            metadata_path = Path(cfg.storage.local.checkpoints.model_cache_dir) / "model.json"

        met = Metadata(save_file_path= root / metadata_path)
        data = (met._load())


        columns = [
            "Rank",
            "Run ID",
            "Model",
            "Experiment",
            "Classes",
            "Cached",]

        for var,metadata in data:
        
            rows = [
                [
                    candidate.rank,
                    candidate.run_id,
                    candidate.model_name,
                    candidate.experiment_name,
                    candidate.num_classes,
                    candidate.cached_at,
                ]
                for candidate in metadata
            ]

            ui.table.show(
            columns,
            rows,
            title=var,
        )

    except Exception as e:
        ui.error(f"Couldnt display table{e}",show_execption=True)
        logger.exception(e)
        








    

