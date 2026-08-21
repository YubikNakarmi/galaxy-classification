
from src.cli.style import UI
from src.cli.validators import get_cuda_stats, get_cpu_stats
from rich.table import Table
from rich.panel import Panel



class ConfigUI:
    def __init__(self, ui:UI):
        self.ui = ui

    def show_train_config(self,cfg, variant, mlflow_uri):

        self.ui.rule(
            "[bold green]Current Configuration[/bold green]",
            style="cyan",
            align="center",width=80,characters="*"
        )

        summary = Table.grid(padding=(0, 2))
        summary.add_column()
        summary.add_column(style="bold cyan", justify="right")
        summary.add_column()

        summary.add_row("","Active Variant", cfg.active_variant)
        summary.add_row("","Training Variant", variant)
        summary.add_row("","MLflow URI", f"[blue]{mlflow_uri}[/blue]")
        summary.add_row("","Seed\n", str(cfg.env.seed))


        cuda_warning:bool = False

        if cfg.env.device == "cuda":
            summary.add_row("Configured for CUDA", "True")
            cuda_stats = get_cuda_stats()
            if not cuda_stats:
                summary.add_row("[error]CUDA Available[/error]", "[error]False[/error]")
                cuda_warning = True
            else:
                summary.add_row("type", cuda_stats.get("type", "N/A"))
                summary.add_row("name", cuda_stats.get("name", "N/A"))
                summary.add_row("load", cuda_stats.get("load", "N/A"))
                summary.add_row("memory", cuda_stats.get("memory", "N/A"))

        elif cfg.env.device == "cpu":
            summary.add_row("[success]CPU Available[/success]", "True")
            cpu_stats = get_cpu_stats()
            summary.add_row("type", cpu_stats.get("type", "N/A"))
            summary.add_row("load", cpu_stats.get("load", "N/A"))
            summary.add_row("memory", cpu_stats.get("memory", "N/A"))

        self.ui.print(Panel(summary, border_style="cyan", padding=(0, 1)),width=80)

        if cuda_warning:
            self.ui.warning("[error]CUDA is configured but not available. Check your GPU setup.[/error]")
                            


        # self.ui.tree.show("tree",asdict(cfg.train))
        #need to add tree for confgis and env vars
        #env vars
        