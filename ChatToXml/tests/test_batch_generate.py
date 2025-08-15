import sys
from pathlib import Path

# Ensure src is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import config
from batch_generate import generate_xml_files
from xml_utils import validate_xml


def test_generate_multiple_files(tmp_path):
    prompt = "make 3 users named Adam. There are 3 users named Adam"
    paths = generate_xml_files(prompt, "user", tmp_path)
    assert len(paths) == 3
    for idx, p in enumerate(paths, 1):
        assert p.name == f"user_{idx}.xml"
        xml_content = p.read_text()
        schema_path = config.SCHEMA_DIR / 'user.xsd'
        valid, error = validate_xml(xml_content, str(schema_path))
        assert valid, error
        assert "<name>Adam</name>" in xml_content
