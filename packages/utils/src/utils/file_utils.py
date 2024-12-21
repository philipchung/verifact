import json
import os
import pickle
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd


def gzip_directory(dir_path, output_path) -> None:
    "Compresses directory into archive. Output path should be a .tar.gz file."
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))


def save_text(text: str, filepath: Path | str | os.PathLike[str]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(text)


def load_text(filepath: Path | str | os.PathLike[str]) -> str:
    with open(filepath) as f:
        return f.read()


def save_pickle(obj, filepath: Path | str | os.PathLike[str], **kwargs) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, **kwargs)


def load_pickle(filepath: Path | str | os.PathLike[str], **kwargs) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f, **kwargs)


def save_pandas(df, filepath: Path | str | os.PathLike[str], **kwargs) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(filepath.suffixes)
    match suffix:
        case ".feather":
            df.to_feather(filepath, **kwargs)
        case ".csv":
            kwargs.setdefault("index", False)
            df.to_csv(filepath, **kwargs)
        case ".csv.gz":
            kwargs.setdefault("index", False)
            kwargs.setdefault("compression", "gzip")
            df.to_csv(filepath, **kwargs)
        case ".parquet":
            df.to_parquet(filepath, **kwargs)
        case ".pkl":
            df.to_pickle(filepath, **kwargs)
        case ".xlsx" | ".xlsm":
            kwargs.setdefault("engine", "openpyxl")
            df.to_excel(filepath, **kwargs)
        case _:
            raise ValueError(f"Unsupported file type: {suffix}")


def load_pandas(filepath: Path | str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    filepath = Path(filepath)
    suffix = "".join(filepath.suffixes)
    match suffix:
        case ".feather":
            return pd.read_feather(filepath, **kwargs)
        case ".csv" | ".csv.gz":
            return pd.read_csv(filepath, **kwargs)
        case ".parquet":
            return pd.read_parquet(filepath, **kwargs)
        case ".pkl":
            return pd.read_pickle(filepath, **kwargs)
        case ".xlsx" | ".xlsm":
            kwargs.setdefault("engine", "openpyxl")
            return pd.read_excel(filepath, **kwargs)
        case _:
            raise ValueError(f"Unsupported file type: {suffix}")


def save_json(obj: dict, filepath: Path | str | os.PathLike[str], **kwargs) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, **kwargs)


def load_json(filepath: Path | str | os.PathLike[str], **kwargs) -> dict:
    with open(filepath) as f:
        return json.load(f, **kwargs)
