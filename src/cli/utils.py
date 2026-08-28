
from __future__ import annotations

from src.cli.style import UI
from src.cli.validators import get_cuda_stats, get_cpu_stats
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.console import Group
from rich.rule import Rule
from rich.text import Text
from pathlib import Path
from src.config import RootConfig
from typing import Optional

ROOT = Path(__file__).parent.parent.parent


class ConfigUI:
    def __init__(self, ui:UI):
        self.ui = ui
        self.root = Path(__file__).parent.parent.parent

    def show_train_config(self,cfg:RootConfig, variant, mlflow_uri):

        self.ui.rule(
            "[bold green]Current Configuration[/bold green]",
            style="cyan",
            align="center",width=80,characters="*"
        )

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan", justify="right")
        summary.add_column()

        summary.add_row("Active Variant", cfg.active_variant)
        summary.add_row("Training Variant", variant)
        summary.add_row("MLflow URI", f"[blue]{mlflow_uri}[/blue]")
        summary.add_row("Seed\n", str(cfg.env.seed))


        cuda_warning:bool = False

        if cfg.env.device == "cuda":
            summary.add_row("Configured for CUDA", "[green]True[/green]")
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


        # if cuda_warning:
        #     self.ui.warning("[error]CUDA is configured but not available. Check your GPU setup.[/error]")
                            
        tree = Tree(
            "\n[bold cyan]Training Configuration[/bold cyan]",
            guide_style="dim cyan",
        )

        training = tree.add("[bold green]Training[/bold green]")
        checkpoints = training.add("[bold yellow]Checkpoints[/bold yellow]")

        checkpoints.add(f"Enabled: [red]{cfg.train.checkpoints.resume_from_checkpoint}[red]" if 
                        not cfg.train.checkpoints.resume_from_checkpoint else "[green]Enabled: False[green]")

        if cfg.train.checkpoints.resume_from_checkpoint:
            run_id = cfg.train.checkpoints.run_id

            try:
                local_checkpoint_path = cfg.train.checkpoints.local_checkpoint_path
        
            except AttributeError:
                local_checkpoint_path = None

            try:
                cloud_path = cfg.train.checkpoints.checkpoint_restore_path
            except AttributeError:
                cloud_path = None


            if run_id is None:
                checkpoints.add(
                    "[error]No run_id specified for checkpointing.[/error]"
                )
            else:
                checkpoints.add(
                    f"run_id: [bold]{run_id}[/bold]"
                )

            if local_checkpoint_path is None:
                checkpoints.add(
                    "[error]No local checkpoint path specified.[/error]"
                )
            else:
                checkpoints.add(
                    f"local_checkpoint_dir: "
                    f"{self.ui.link_path(local_checkpoint_path) if local_checkpoint_path else '[error]Not specified[/error]'}"
                )

            if cloud_path is None:
                checkpoints.add(
                    "[error]No cloud checkpoint path specified.[/error]"
                )
            else:
                checkpoints.add(
                    f"mlflow_checkpoint_dir: "
                    f"{self.ui.link_path(cloud_path) if cloud_path else '[error]Not specified[/error]'}"
                )

        if cuda_warning:
            warning = Text(
                "\n⚠ CUDA is configured but not available. Check your GPU setup.",
                style="bold red",
            )
        else:
            warning = None

        training.add(f"Batch Size: {cfg.train.batch_size}")
        training.add(f"[bold]Epochs: {cfg.train.num_epochs}[/bold]")
        training.add(f"Learning Rate: {cfg.train.learning_rate}")
        training.add(f"Optimizer: {cfg.train.optimizer}")
        training.add(f"Scheduler: {cfg.train.scheduler.type}")


        #groupd sequentially
        content = Group(
            summary,
            warning if warning else "",Rule(style="cyan"),#cant use normal obj print, as it needs to be renderable
            tree,
        )

        self.ui.print(Panel(content, border_style="cyan", padding=(0, 1)),width=80)
        self.ui.print(f"Full configuration at: {self.ui.link_path(self.root / 'configs' / 'ml.yaml')} ", 
                      style="cyan", justify="left")



class ModelExporter:

    def __init__(self,model_path:str| Path,
                  export_path:str|Path,
                    ui:Optional[UI]=None,
                    input_shape:tuple=(1,3,128,128)):

        
        import torch 
        from src.utils.validation import validate_path_exists

        self.torch = torch
        self.model_path = Path(model_path)
        self.export_path = Path(export_path)

        validate_path_exists(root = ROOT, 
                             path = [self.model_path, self.export_path.parent], 
                             create=True, kind="file", 
                             validate_root=False)

        self.ui = ui or UI(verbose=0) #fallback

        self.model = self.torch.load(self.model_path,weights_only=False ,map_location=self.torch.device("cpu")).eval()

        self.dummy_input = self.torch.randn(input_shape) #default input shape for galaxy images

    
    def export_onnx(self):

        self.ui.info(f"Exporting model to ONNX format at {self.export_path} ...")
        self.torch.onnx.export(self.model, 
                               (self.dummy_input,),
                               str(self.export_path),
                               input_names=["image"],
                               output_names=["output"],
                               opset_version=18,
                                dynamo=True,
                               )

    def export_torchscript(self):

        self.ui.info(f"Exporting model to TorchScript format at {self.export_path} ...")
        traced_model = self.torch.jit.script(self.model, (self.dummy_input,))
        traced_model.save(str(self.export_path))

    def export_tensorRT(self, precision:str="fp16"):
        pass

      