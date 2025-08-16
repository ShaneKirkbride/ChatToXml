import sys
from pathlib import Path
from zipfile import ZipFile

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import file_store


def _make_zip(tmp_path: Path, name: str) -> Path:
    """Create a zip file containing three XML files in separate folders."""
    for i in range(3):
        folder = tmp_path / f"folder{i}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"file{i}.xml").write_text(f"<root{i}/>")
    zip_path = tmp_path / name
    with ZipFile(zip_path, "w") as zf:
        for i in range(3):
            folder = tmp_path / f"folder{i}"
            for file in folder.iterdir():
                zf.write(file, arcname=f"{folder.name}/{file.name}")
    return zip_path


def test_zip_crud_lifecycle(monkeypatch, tmp_path):
    # Redirect output directory to a temporary path
    monkeypatch.setattr(file_store, "OUTPUT_DIR", tmp_path)

    src_zip = _make_zip(tmp_path, "src.zip")
    status = file_store.create_zip_file("archive.zip", str(src_zip))
    assert "Created" in status
    assert (tmp_path / "archive.zip").exists()

    listing, msg = file_store.read_zip_file("archive.zip")
    assert "folder0/file0.xml" in listing
    assert "Loaded" in msg

    # Create another zip to update
    src_zip2 = _make_zip(tmp_path, "src2.zip")
    update_status = file_store.update_zip_file("archive.zip", str(src_zip2))
    assert "Updated" in update_status

    delete_status = file_store.delete_zip_file("archive.zip")
    assert "Deleted" in delete_status
    assert not (tmp_path / "archive.zip").exists()


def test_safe_path(monkeypatch, tmp_path):
    monkeypatch.setattr(file_store, "OUTPUT_DIR", tmp_path)
    src_zip = _make_zip(tmp_path, "src.zip")
    file_store.create_zip_file("../evil.zip", str(src_zip))
    # Path traversal should be sanitized
    assert (tmp_path / "evil.zip").exists()
