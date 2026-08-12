from src.utils.bootstrap import Context,load_context


def validate_mlflow_connection(ctx: Context) -> bool:
    from urllib.request import Request, urlopen

    mlflow_tracking_uri = ctx.mlflow_tracking_uri
    health_url = f"{mlflow_tracking_uri}/health" #mlflow_tracking_uri set by env var in container or out of.
    try:
        req = Request(health_url)
        with urlopen(req) as response:
            if response.status == 200:
                return True
            
    except Exception:
        return False


