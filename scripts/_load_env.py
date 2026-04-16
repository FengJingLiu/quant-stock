"""Import this module to auto-load the project .env file into os.environ."""
import os
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k, _v = _k.strip(), _v.strip()
        if _k and _v and _k not in os.environ:
            os.environ[_k] = _v
