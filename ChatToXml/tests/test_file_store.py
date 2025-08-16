import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import file_store


def test_crud_lifecycle(monkeypatch, tmp_path):
    # Redirect output directory to a temporary path
    monkeypatch.setattr(file_store, "OUTPUT_DIR", tmp_path)

    status = file_store.create_xml_file("sample.xml", "<root/>")
    assert "Created" in status
    assert (tmp_path / "sample.xml").exists()

    content, msg = file_store.read_xml_file("sample.xml")
    assert content == "<root/>"
    assert "Loaded" in msg

    update_status = file_store.update_xml_file("sample.xml", "<root>updated</root>")
    assert "Updated" in update_status
    updated_content, _ = file_store.read_xml_file("sample.xml")
    assert updated_content == "<root>updated</root>"

    delete_status = file_store.delete_xml_file("sample.xml")
    assert "Deleted" in delete_status
    assert not (tmp_path / "sample.xml").exists()


def test_safe_path(monkeypatch, tmp_path):
    monkeypatch.setattr(file_store, "OUTPUT_DIR", tmp_path)
    file_store.create_xml_file("../evil.xml", "<root/>")
    # Path traversal should be sanitized
    assert (tmp_path / "evil.xml").exists()
