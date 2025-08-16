from pathlib import Path

# Directory for persisting generated XML files
OUTPUT_DIR = Path("data") / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(name: str) -> Path:
    """Return a safe file path within OUTPUT_DIR for the given name."""
    return OUTPUT_DIR / Path(name).name


def create_xml_file(file_name: str, xml_content: str) -> str:
    """Create a new XML file with the provided content."""
    path = _safe_path(file_name)
    if path.exists():
        return f"File {path} already exists."
    path.write_text(xml_content, encoding="utf-8")
    return f"Created {path}"


def read_xml_file(file_name: str) -> tuple[str, str]:
    """Read an XML file and return its contents and status message."""
    path = _safe_path(file_name)
    if not path.exists():
        return "", f"File {path} does not exist."
    content = path.read_text(encoding="utf-8")
    return content, f"Loaded {path}"


def update_xml_file(file_name: str, xml_content: str) -> str:
    """Update an existing XML file with new content."""
    path = _safe_path(file_name)
    if not path.exists():
        return f"File {path} does not exist."
    path.write_text(xml_content, encoding="utf-8")
    return f"Updated {path}"


def delete_xml_file(file_name: str) -> str:
    """Delete an XML file."""
    path = _safe_path(file_name)
    if not path.exists():
        return f"File {path} does not exist."
    path.unlink()
    return f"Deleted {path}"
