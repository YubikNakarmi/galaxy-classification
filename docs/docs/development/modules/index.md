# Development modules

The source tree is organized into focused modules so that CLI behavior,
datasets, models, and shared utilities can evolve independently.

- [CLI](cli.md): terminal UI and command-facing helpers.
- [Dataset](dataset.md): dataset loading and image/label preparation.
- [Model](model.md): classifier architectures and model behavior.
- [Utils](utils.md): shared configuration, logging, and infrastructure helpers.

Use the [configuration guide](../configs.md) and [testing guide](../testing.md)
when changing module behavior.