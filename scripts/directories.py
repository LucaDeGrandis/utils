from pathlib import Path

def make_dir(
  path: str
) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
