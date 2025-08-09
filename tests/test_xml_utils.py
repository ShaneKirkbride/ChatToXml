import sys
from pathlib import Path

# Ensure src is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import xml_utils
import config


def test_validate_xml_valid():
    schema_path = Path(__file__).resolve().parents[1] / 'schema' / 'user.xsd'
    xml = '<user><id>1</id><name>Ada</name><email>ada@example.com</email></user>'
    valid, error = xml_utils.validate_xml(xml, str(schema_path))
    assert valid is True
    assert error is None


def test_validate_xml_invalid():
    schema_path = Path(__file__).resolve().parents[1] / 'schema' / 'user.xsd'
    xml = '<user><id>1</id></user>'  # Missing required name element
    valid, error = xml_utils.validate_xml(xml, str(schema_path))
    assert valid is False
    assert 'name' in error


def test_pretty_formats_xml():
    xml = '<root><child>value</child></root>'
    result = xml_utils.pretty(xml)
    assert '\n' in result
    assert '  <child>value</child>' in result


def test_config_paths_are_paths():
    # Importing config executes all lines and exposes Path constants
    assert config.SCHEMA_DIR.name == 'schema'
    assert config.DATA_CSV.name == 'sample_data.csv'
