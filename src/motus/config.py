"""Read and write motus.toml project configuration."""

import tomllib
from collections import UserDict
from pathlib import Path

import tomli_w


class Config(UserDict):
    """A dict-like view over ``motus.toml``

    We're caching the read, but writes are always flushed (write through).
    Config writes shouldn't be common.
    """

    def __init__(self, start: Path | None = None):
        current = (start or Path.cwd()).resolve()
        self._path = None
        while True:
            candidate = current / "motus.toml"
            if candidate.is_file():
                self._path = candidate
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

        if self._path is not None:
            with open(self._path, "rb") as f:
                self.data = tomllib.load(f)
        else:
            self._path = (start or Path.cwd()).resolve() / "motus.toml"
            self.data = {}

    def _flush(self):
        self._path.write_text(tomli_w.dumps(self.data))

    def __setitem__(self, key: str, value: str):
        super().__setitem__(key, value)
        self._flush()

    def __delitem__(self, key: str):
        super().__delitem__(key)
        self._flush()

    def update(self, other=(), **kwargs):
        self.data.update(other, **kwargs)
        if other or kwargs:
            self._flush()


CONFIG = Config()
