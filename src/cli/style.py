from __future__ import annotations
from typing import Optional, Sequence

from typing import TYPE_CHECKING
#type checking without importing on runtime
if TYPE_CHECKING:
    from rich.console import Console

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

class TableUI:
    def __init__(self,ui:UI):
        self.ui = ui

    def show(self,columns:list[str],
               rows:Sequence[Sequence[str]],
               title:Optional[str]=None,
               show_header:bool=True,
               header_style:str="bold blue"):

        from rich.table import Table
        
        table = Table(title=title,
                                show_header=show_header,
                                header_style=header_style)
        
        for col in columns:
            table.add_column(col)

        for row in rows:
            table.add_row(*(str(value) for value in row)) #str conversion in case int
        self.ui.console.print(table)


