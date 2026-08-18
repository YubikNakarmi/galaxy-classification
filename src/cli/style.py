from __future__ import annotations
from typing import Optional, Sequence

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rich.console import Console

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from contextlib import contextmanager

class UI:
    """
    A simple UI class for printing in consonle with rich formatting and verbosity control.

    Example:
    ```python
        ui = UI(verbose=2)
        ui.success("Model loaded")           # always shown
        ui.info("Processing batch 3/10")     # shown at level 1 and above
        ui.debug(f"Batch shape: {shape}")    # shown at level 2 and above
        ui.trace(f"Raw tensor: {tensor}")    # only shown at level 3 
    ```
    """

    THEME = {
            "info": "bright_blue",
            "success": "bold green",
            "warning": "bold gold3",
            "error": "bold red",
            "debug": "grey50",
            "trace": "grey50",
            "banner": "bold magenta",
            }

    
    def __init__(self, verbose: int = 0, 
                 console:Optional[Console| None] = None,
                 theme:Optional[dict[str,str]]=None,):
        """
        Constructor for UI class
        Args:
            verbose: Verbose level from 0 to 3, 0 being default. 
            console: Rich console instance if required, the class initialized it by default. 
            theme: If you want custom theme of the ui levels youc an pass it. 
        """

        if theme is not None:
            self.theme = theme
        else:
            self.theme = self.THEME.copy()
            
        self.verbose = verbose
        self._console = console
        self.table = TableUI(self)#composition
        self.progress = ProgressUI(self)
        self.train = TrainDashboard(self)   

       
    @property
    def console(self): #lazy import with property decorator to avoid circular imports
        if self._console is None:
            from rich.console import Console
            from rich.theme import Theme
            
            self._console = Console(theme = Theme(self.theme))
        return self._console

    def print(self,*args,**kwargs):
        self.console.print(*args, **kwargs)

    def success(self, message: str,style:Optional[str]=None):
        self.console.print(f"[success]✔ {message}[/success] ",style = style)

    def warning(self, message: str,style:Optional[str]=None):
        self.console.print(f"[warning]⚠[/warning] [white] {message}[/white]",style = style)

    def error(self, message: str,style:Optional[str]=None, show_execption:bool=False):
        self.console.print(f"[error]✖ {message}[/error] ",style = style)

        if show_execption:
            self.console.print_exception(show_locals=self.verbose >= 3,max_frames=5)
            
    # verbosity gated    
    def info(self, message:str,style:Optional[str]=None):
        if self.verbose >=1:
            self.console.print(f"[info]ℹ[/info] {message}",style = style)

    def debug(self, message:str,style:Optional[str]=None):
        if self.verbose >=2:
            self.console.print(f"[debug]•[/debug] {message}",style = style)

    def trace(self, message:str,style:Optional[str]=None):
        if self.verbose >=3:
            self.console.print(f"[trace] {message}[/trace] ",style = style)


    def summary(self, title: str, items: dict[str, str]):
        """ Summary box 
        Args:
            title: title of the bost
            items: dictoinary of key value pairs
        """

        body = "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in items.items())
        self.console.print(Panel(body, title=title, border_style="success"))


    def rule(self, title: str = "",style:Optional[str]=None):
        self.console.rule(f"[banner]{title}[/banner]", style=style)

    @contextmanager
    def status(self, message, spinner="dots", level=0): #editable levels
        if self.verbose >= level:
            with self.console.status(message, spinner=spinner) as status:
                yield status
class TableUI:
    def __init__(self,ui:UI):
        self.ui = ui

    def show(self,columns:list[str],
               rows:Sequence[Sequence[str]],
               title:Optional[str]=None,
               show_header:bool=True,
               header_style:str="bold blue"):

        
        table = Table(title=title,
                                show_header=show_header,
                                header_style=header_style)
        
        for col in columns:
            table.add_column(col)

        for row in rows:
            table.add_row(*(str(value) for value in row)) #str conversion in case int
        self.ui.console.print(table)

class ProgressUI:
    """
    A simple progress bar class for printing in console with rich formatting and verbosity control."""
    def __init__(self, ui):
        self.ui = ui

    def bar(self, description:str="Working"):

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=self.ui.console,
        )
class TrainDashboard:
    """
    Composed onto UI as ui.train.

    Example:
        with ui.train.session(epochs=10, steps_per_epoch=50) as dash:
            for epoch in range(10):
                dash.start_epoch(epoch)
                for step in range(50):
                    loss, acc = train_step(...)
                    dash.update_batch(loss=loss, acc=acc)
                dash.end_epoch(val_loss=..., val_acc=...)
    """

    def __init__(self, ui: UI):
        self._ui = ui
        self.metrics_history: list[dict] = []

    def session(self, epochs: int, steps_per_epoch: int):
        return _TrainSession(self._ui.console, epochs, steps_per_epoch, self.metrics_history)

class _TrainSession:
    def __init__(self, console, epochs, steps_per_epoch, history):
        self.console = console
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.history = history

        self.epoch_progress = Progress(
            TextColumn("[bold cyan]Epoch"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self.batch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bright_blue]Batch"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[stats]}"),
            console=console,
        )

        self.epoch_task = None
        self.batch_task = None
        self.live = None
        self._current_epoch = 0

    def _render(self):
        table = Table(title="Latest Metrics", expand=True)
        table.add_column("Epoch", justify="right")
        table.add_column("Train Loss", justify="right", style="bright_blue")
        table.add_column("Train Acc", justify="right", style="bright_blue")
        table.add_column("Val Loss", justify="right", style="magenta")
        table.add_column("Val Acc", justify="right", style="magenta")

        for row in self.history[-5:]:
            table.add_row(
                str(row["epoch"]),
                f"{row.get('loss', 0):.4f}",
                f"{row.get('acc', 0):.2%}",
                f"{row.get('val_loss', float('nan')):.4f}" if "val_loss" in row else "-",
                f"{row.get('val_acc', float('nan')):.2%}" if "val_acc" in row else "-",
            )

        return Group(
            Panel(Group(self.epoch_progress, self.batch_progress), title="Progress", border_style="cyan"),
            table,
        )

    def __enter__(self):
        self.epoch_task = self.epoch_progress.add_task("epochs", total=self.epochs)
        self.live = Live(self._render(), console=self.console, refresh_per_second=8)
        self.live.__enter__()
        return self

    def __exit__(self, *exc):
        self.live.__exit__(*exc)

    def start_epoch(self, epoch: int):
        self._current_epoch = epoch
        self.batch_task = self.batch_progress.add_task("batches", total=self.steps_per_epoch, stats="")
        self.live.update(self._render())

    def start_validation(self, val_steps: int):
        self.val_task = self.batch_progress.add_task(
            "[magenta]Val batch", total=val_steps, stats=""
        )
        self.live.update(self._render())

    
    def update_val_batch(self, **stats):
        stat_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in stats.items())
        self.batch_progress.update(self.val_task, advance=1, stats=stat_str)
        self.live.update(self._render())
        
    def update_batch(self, **stats):
        stat_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in stats.items()
        )
        self.batch_progress.update(self.batch_task, advance=1, stats=stat_str)
        self.live.update(self._render())

    def end_epoch(self, **final_stats):
        self.epoch_progress.update(self.epoch_task, advance=1)
        self.batch_progress.remove_task(self.batch_task)
        self.history.append({"epoch": self._current_epoch, **final_stats})
        self.live.update(self._render())




# ui = UI(verbose=2)
# ui.summary("Import Complete", {
#     "Run ID": "1010101010",
#     "Metric": f"hello=hi",
#     "Cache path": str(121212),
# })

# ui = UI(verbose=2)

# ui.error("prrrint")
# ui.warning("hello")
# ui.rule("rule")
# ui.success("suc")
# import time
# with ui.console.status("[info]Loading model from cache...[/info]"):
#     time.sleep(10)