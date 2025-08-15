"""Utility functions for working with XML.

This module avoids external dependencies so that it can run in the
restricted execution environment used by the tests and CI pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple
from xml.dom import minidom
from xml.etree import ElementTree as ET


def validate_xml(xml_str: str, xsd_path: str) -> Tuple[bool, Optional[str]]:
    """Validate ``xml_str`` against a very small subset of XSD features.

    The original implementation relied on :mod:`lxml` which is not available in
    the execution environment.  To keep the public API intact we perform a
    minimal validation by parsing the XSD and ensuring required child elements
    exist in the provided XML document.

    Parameters
    ----------
    xml_str:
        XML document as a string.
    xsd_path:
        Path to an XSD file describing the expected structure.

    Returns
    -------
    ``Tuple[bool, Optional[str]]``
        ``True`` and ``None`` if the XML satisfies the schema, otherwise
        ``False`` and an error message describing the first problem found.
    """

    try:
        xml_doc = ET.fromstring(xml_str)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        return False, f"XML parse error: {exc}"

    try:
        schema_tree = ET.parse(xsd_path)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        return False, f"XSD parse error: {exc}"

    ns = {"xs": "http://www.w3.org/2001/XMLSchema"}
    root_schema = schema_tree.getroot().find("xs:element", ns)
    if root_schema is None:
        return False, "Invalid schema: missing root element"

    root_name = root_schema.attrib.get("name")
    if xml_doc.tag != root_name:
        return False, f"Root element '{xml_doc.tag}' does not match '{root_name}'"

    sequence = root_schema.find(".//xs:sequence", ns)
    if sequence is not None:
        for child in sequence.findall("xs:element", ns):
            name = child.attrib.get("name")
            min_occurs = child.attrib.get("minOccurs", "1")
            exists = xml_doc.find(name) is not None
            if min_occurs != "0" and not exists:
                return False, f"Element '{name}' is missing"

    return True, None


def pretty(xml_str: str) -> str:
    """Return a nicely formatted representation of ``xml_str``.

    ``xml.dom.minidom`` is used instead of :mod:`lxml`'s pretty printer to
    avoid an optional C dependency.
    """

    element = ET.fromstring(xml_str)
    rough = ET.tostring(element, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")

