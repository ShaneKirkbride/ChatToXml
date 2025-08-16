from pathlib import Path
import shutil
from zipfile import ZipFile

# Directory for persisting generated files
OUTPUT_DIR = Path("data") / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(name: str) -> Path:
    """Return a safe file path within OUTPUT_DIR for the given name."""
    return OUTPUT_DIR / Path(name).name


def create_zip_file(file_name: str, source_zip: str) -> str:
    """Persist an uploaded ZIP archive to the output directory."""
    path = _safe_path(file_name)
    if path.exists():
        return f"File {path} already exists."
    shutil.copy(Path(source_zip), path)
    return f"Created {path}"


def read_zip_file(file_name: str) -> tuple[str, str]:
    """Read a ZIP archive and return a newline separated listing of its contents."""
    path = _safe_path(file_name)
    if not path.exists():
        return "", f"File {path} does not exist."
    with ZipFile(path) as zf:
        listing = "\n".join(zf.namelist())
    return listing, f"Loaded {path}"


def update_zip_file(file_name: str, source_zip: str) -> str:
    """Replace the contents of an existing ZIP archive."""
    path = _safe_path(file_name)
    if not path.exists():
        return f"File {path} does not exist."
    shutil.copy(Path(source_zip), path)
    return f"Updated {path}"


def delete_zip_file(file_name: str) -> str:
    """Delete a ZIP archive."""
    path = _safe_path(file_name)
    if not path.exists():
        return f"File {path} does not exist."
    path.unlink()
    return f"Deleted {path}"


def extract_zip_file(source_zip: str) -> str:
    """Extract the given ZIP archive into OUTPUT_DIR."""
    path = Path(source_zip)
    if not path.exists():
        return f"File {path} does not exist."
    with ZipFile(path) as zf:
        zf.extractall(OUTPUT_DIR)
    return f"Extracted {path} to {OUTPUT_DIR}"
