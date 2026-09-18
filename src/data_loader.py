from pathlib import Path
import pandas as pd

def find_data_file(filename, start_path=None):
    base = Path(start_path) if start_path else Path.cwd()
    candidates = [
        base / "Data" / filename,
        base.parent / "Data" / filename,
    ]

    for path in candidates:
        if path.is_file():
            return path.resolve()

    project_root = base.parent if base.name.lower() in {"notebook", "src", "dashboard"} else base
    recursive_matches = list(project_root.glob(f"Data/{filename}"))

    if recursive_matches:
        return recursive_matches[0].resolve()

    raise FileNotFoundError(f"{filename} was not found in the project Data folder.")

def discover_csv(start_path=None, exclude_keywords=("cleaned", "features", "drift_results", "output", "processed")):
    base = Path(start_path) if start_path else Path.cwd()
    directories = [
        base / "Data",
        base.parent / "Data",
    ]

    csv_files = []
    for directory in directories:
        if directory.is_dir():
            csv_files.extend(directory.glob("*.csv"))

    csv_files = sorted(set(path.resolve() for path in csv_files))

    if not csv_files:
        raise FileNotFoundError("No CSV file was found in the project Data folder.")

    usable = [
        path for path in csv_files
        if not any(keyword in path.name.lower() for keyword in exclude_keywords)
    ]

    return usable[0] if usable else csv_files[0]

def load_csv(filename=None, start_path=None):
    path = find_data_file(filename, start_path) if filename else discover_csv(start_path)
    return pd.read_csv(path), path
