from __future__ import annotations
import sys
from typing import Optional, Sequence

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rich.console import Console

from rich.rule import Rule
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, MofNCompleteColumn,
)
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from loguru import logger
from contextlib import contextmanager
from src.config import RootConfig
from pydantic import BaseModel
from pathlib import Path


class UI:
    """
    A simple UI class for printing in consonle with rich formatting and verbosity control.

    Args:
        verbose (int, optional):
            Verbosity level controlling which messages are displayed.

            - ``0``: Only success and important messages.
            - ``1``: Include informational messages.
            - ``2``: Include debug messages.
            - ``3``: Include trace messages.

            Defaults to ``0``.

        console (Console | None, optional):
            A Rich ``Console`` instance to use for output. If ``None``,
            a console is initialized automatically.

        theme (dict[str, str] | None, optional):
            Custom color/style mapping for different message types.
            If ``None``, the default :attr:`THEME` is used.

        log (bool, optional):
            Whether logging is enabled for the UI. Defaults to ``True``.
    Attributes:
        THEME (dict[str, str]):
            Default Rich style definitions for different message types.

        verbose (int):
            Current verbosity level.

        theme (dict[str, str]):
            Active color/style configuration.

        table (TableUI):
            Table rendering component.

        progress (ProgressUI):
            Progress bar component.

        train (TrainDashboard):
            Training dashboard component.

        eval (EvalDashboard):
            Evaluation dashboard component.

        tree (TreeUI):
            Tree rendering component.

        log (bool):
            Whether logging is enabled.
        
        
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
                 theme:Optional[dict[str,str]]=None,
                 log:bool=True):
        """
        Constructor for UI class
        Args:
            verbose: Verbose level from 0 to 3, 0 being default. 
            console: Rich console instance if required, the class initialized it by default. 
            theme: If you want custom theme of the ui levels youc an pass it. 
            log: If you want to disable logging, you can set this to False.
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
        self.eval = EvalDashboard(self)
        self.tree = TreeUI(self)
        self.log = log #if you want to disable logging for some reason, you can set this to False
        
       
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
        if self.log:
            logger.opt(depth=1).success(message) #one step up from the curr file for proper file resolution

    def warning(self, message: str,style:Optional[str]=None):
        self.console.print(f"[warning]⚠[/warning] [white] {message}[/white]",style = style)
        
        if self.log:
            logger.opt(depth=1).warning(message)

    def error(self, message: str,style:Optional[str]= None, 
              show_execption:bool=False):
        
        self.console.print(f"[error]✖ {message}[/error] ",style = style)
        
        if self.log:
            logger.opt(depth=1).error(message)

        if show_execption and sys.exc_info()[0] is not None:
            self.console.print_exception(show_locals=self.verbose >= 3,max_frames=5)
            
            if self.log:
                logger.opt(depth=1).exception(message)
            
    # verbosity gated    
    def info(self, message:str,style:Optional[str]=None):
        if self.verbose >=1:
            self.console.print(f"[info]ℹ[/info] {message}",style = style)
        if self.log:
            logger.opt(depth=1).info(message)

    def debug(self, message:str,style:Optional[str]=None):
        if self.verbose >=2:
            self.console.print(f"[debug]•[/debug] {message}",style = style)
            
            if self.log:
                logger.opt(depth=1).debug(message)

    def trace(self, message:str,style:Optional[str]=None):
        if self.verbose >=3:
            self.console.print(f"[trace] {message}[/trace] ",style = style)
            
            if self.log:
                logger.opt(depth=1).trace(message)


    def summary(self, title: str, items: dict[str, str]):
        """ Summary box 
        Args:
            title: title of the bost
            items: dictoinary of key value pairs
        """

        body = "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in items.items())
        self.console.print(Panel(body, title=title, border_style="success"))

    def rule(self, title: str = "",style:Optional[str]=None, 
             align: str = "center",
             characters: str = "─",width:Optional[int]=None):
        """Print a rule with optional title and style.
        
        Args:
            title: The title of the rule. Can use inline markup for styling.
            style: The style of the rule.
            align: The alignment of the rule.
            characters: The characters to use for the rule.
            width: The width of the rule.
        Example:
            ```python
            ui = UI(verbose=2)
            ui.rule("[bold green]Project Status[/bold green]", style="bold green",align="center",characters=".",width=80)
            ```
        """


        self.console.print(Rule(title, style=style, 
                                align=align,characters=characters),
                                  width=width) #rich doesnt have width optoin limit to deafult print width

    def link_path(self,path: str|Path) -> str:

        path = Path(path)

        uri = path.resolve().as_uri() #multi platform

        if not uri.startswith("file:///"):
            uri = f"file:///{uri}"

        return f"[link={uri}]{path}[/link]"
  
    @contextmanager
    def status(self, message, spinner="dots", level=0): #editable levels
        if self.verbose >= level:
            with self.console.status(message, spinner=spinner) as status:
                yield status
        else:
            yield _SilentStatus()


class _SilentStatus:
    def update(self, message):
        pass


class TableUI:
    def __init__(self,ui:UI,table:Optional[Table|None]=None):
        self.ui = ui
        self.table = table#custom obj

    def show(self,columns:list[str],
               rows:Sequence[Sequence[str]],
               title:Optional[str]=None,
               show_header:bool=True,
               header_style:str="bold blue"):

        if self.table:
                    table = self.table

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
        
class EvalDashboard:
    """
    Composed onto UI as ui.eval.

    Example:
        with ui.eval.session(steps=50) as dash:
            for step in range(50):
                loss, acc = eval_step(...)
                dash.update_batch(loss=loss, acc=acc)
    """

    def __init__(self, ui):
        self._ui = ui
        self.metrics_history: list[dict] = []

    def session(self, steps: int) -> "_EvalSession":
        return _EvalSession(self._ui.console, steps, self.metrics_history)


class _EvalSession:
    def __init__(self, console: Console, steps: int, history: list[dict]):
        self.console = console
        self.steps = steps
        self.history = history
        self.live: Live | None = None
        self.batch_task = None

        self.batch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bright_blue]Batch"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[stats]}"),
            console=console,
        )

    def __enter__(self) -> "_EvalSession":
        self.batch_task = self.batch_progress.add_task(
            "[magenta]Val batch", total=self.steps, stats=""
        )
        self.live = Live(self._render(), console=self.console, refresh_per_second=8)
        self.live.__enter__()
        return self

    def __exit__(self, *exc):
        self.live.__exit__(*exc)

    def _render(self) -> Group:
        table = Table(title="Latest Metrics", expand=True)
        table.add_column("Step", justify="right")

        if self.history:
            for key in self.history[-1]:
                if key != "step":
                    table.add_column(key, justify="right", style="bright_blue")

            for row in self.history[-5:]:
                table.add_row(
                    str(row["step"]),
                    *[f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k])
                    for k in row if k != "step"],
                )

        return Group(
            self.batch_progress,
            Panel(table, title="Evaluation Metrics", border_style="cyan"),)

    def update_batch(self, **stats):
        step = self.batch_progress.tasks[self.batch_task].completed + 1
        self.history.append({"step": step, **stats})

        stat_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in stats.items()
        )
        self.batch_progress.update(self.batch_task, advance=1, stats=stat_str)
        self.live.update(self._render())
        
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
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[stats]}"),
            console=console,
        )
        self.val_task = None
        self._last_train = {}

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
        self._last_train = {}
        self.batch_task = self.batch_progress.add_task(
            "[bright_blue]Batch", total=self.steps_per_epoch, stats=""
        )
        self.live.update(self._render())

    def update_batch(self, **stats):
        self._last_train = stats  # remember latest train metrics
        stat_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in stats.items()
        )
        self.batch_progress.update(self.batch_task, advance=1, stats=stat_str)
        self.live.update(self._render())

    def end_epoch(self, **final_stats):
        self.epoch_progress.update(self.epoch_task, advance=1)
        self.batch_progress.remove_task(self.batch_task)
        if self.val_task is not None:
            self.batch_progress.remove_task(self.val_task)
            self.val_task = None
        self.history.append({"epoch": self._current_epoch, **self._last_train, **final_stats})
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
        
   

class TreeUI:
    def __init__(self, ui: UI):
        self.ui = ui

    def show(self, root_label: str, items: dict[str, str]):
        tree = Tree(f"[bold cyan]{root_label}[/bold cyan]")
        for key, value in items.items():
            tree.add(f"[bold]{key}:[/bold] {value}")
            
        self.ui.console.print(tree)
        
    def _add_config_tree(self, parent, name, config):
        branch = parent.add(f"[bold]{name}[/bold]")

        for key, value in vars(config).items():
            branch.add(
                f"[dim]{key}[/dim]: {value}"
            )

class ConfigTree:

    def __init__(self,config:RootConfig|dict|BaseModel, ui:UI):

        self.config = config
        self.ui = ui

    def _add_fields(self, tree: Tree):
        pass

    def build_config_tree(self, cfg: BaseModel, label: str = "RootConfig") -> Tree:
        tree = Tree(f"[bold cyan]{label}[/bold cyan]")
        self._add_fields(tree)
        return tree

  

    def show_config(self, cfg: BaseModel):
        Console().print(self.build_config_tree(cfg))



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