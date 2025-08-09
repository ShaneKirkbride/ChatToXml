from __future__ import annotations

from lxml import etree
from typing import Optional, Tuple


def validate_xml(xml_str: str, xsd_path: str) -> Tuple[bool, Optional[str]]:
    try:
        xml_doc = etree.fromstring(xml_str.encode("utf-8"))
    except Exception as e:  # pragma: no cover - defensive
        return False, f"XML parse error: {e}"
    with open(xsd_path, "rb") as f:
        schema_doc = etree.XML(f.read())
    schema = etree.XMLSchema(schema_doc)
    is_valid = schema.validate(xml_doc)
    if not is_valid:
        return False, "; ".join(str(e) for e in schema.error_log)
    return True, None


def pretty(xml_str: str) -> str:
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str.encode("utf-8"), parser)
    return etree.tostring(root, pretty_print=True, encoding="unicode")
