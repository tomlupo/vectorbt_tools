# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

try:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from pathlib import Path

    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    __version__ = pyproject["project"]["version"]
except Exception as e:
    import importlib.metadata

    __version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    "__version__",
]
