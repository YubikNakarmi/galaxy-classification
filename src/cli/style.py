from __future__ import annotations
from typing import Literal,Optional
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
    def __init__(self, verbose: int = 0, console=None):
      
        self.theme ={
            "info": "bright_blue",
            "success": "bold spring_green2",
            "warning": "bold gold3",
            "error": "bold deep_pink3",
            "debug": "grey50",
            "trace": "dim grey37",
            "banner": "bold magenta",
        }
        
        self.verbose = verbose
        self._console = console

       
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
        self.console.print(f"[success]✔[/success] {message}",style = style)

    def warning(self, message: str,style:Optional[str]=None):
        self.console.print(f"[warning]⚠[/warning] {message}",style = style)

    def error(self, message: str,style:Optional[str]=None):
        self.console.print(f"[error]✖[/error] {message}",style = style)

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

