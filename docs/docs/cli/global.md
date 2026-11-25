# Global usage and logging

## Global options

```text
galax [OPTIONS] COMMAND [ARGS]...

Options:
  --version      Show the installed version.
  -v, --verbose  Increase verbosity level (-v, -vv, -vvv).
  -h, --help     Show help and exit.
```

Global options must appear before the command:

```bash
galax -vv pipeline train --status
galax --version
```

## `status`

```bash
galax status
galax status --help
```

Displays the resolved configuration, MLflow tracking URI, and project
environment information. It has no project-specific options.

## `log`

`log` is a command group with one command: `show`.

```bash
galax log --help
galax log show [OPTIONS]
```

### `log show` options

| Option | Default | Description |
| --- | --- | --- |
| `-t, --type [ml\|app]` | `ml` | Select the ML or application log. |
| `-l, --level [DEBUG\|INFO\|SUCCESS\|WARNING\|ERROR\|CRITICAL]` | — | Keep only one log level. |
| `-n, --lines INTEGER` | `50` | Number of matching lines to display. |
| `-s, --service TEXT` | — | Filter by service name. |

The `--type` value selects either the ML log (`ml`) or application log
(`app`). `--lines` limits output after the other filters are applied.

```bash
galax log show --type ml --level ERROR --lines 25
galax log show --service training
```
