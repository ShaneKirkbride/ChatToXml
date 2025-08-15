from __future__ import annotations

import re
from pathlib import Path
from typing import List

from config import SCHEMA_DIR
from repair import repair_to_schema
from xml_utils import pretty, validate_xml


def _parse_quantity(text: str) -> int:
    """Extract the first integer from *text*.

    If no number is found a quantity of ``1`` is assumed.
    """
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else 1


def _generate_single(prompt: str, schema: str) -> str:
    """Generate a single XML document from ``prompt`` for ``schema``.

    The function relies on :func:`repair_to_schema` which uses lightweight
    heuristics to build a valid XML snippet.  The resulting XML is validated
    against the schema and pretty formatted before returning.
    """
    xml = repair_to_schema(prompt, schema)
    schema_path = SCHEMA_DIR / f"{schema}.xsd"
    valid, err = validate_xml(xml, str(schema_path))
    if not valid:
        raise ValueError(f"Generated XML failed validation: {err}")
    return pretty(xml)


def generate_xml_files(prompt: str, schema: str, output_dir: Path) -> List[Path]:
    """Generate one or more XML files based on ``prompt``.

    Parameters
    ----------
    prompt:
        Natural language description of the desired XML content.  If the text
        contains a number (e.g. ``"make 3 users"``) that quantity of files will
        be produced.
    schema:
        Name of the XML schema to validate against (e.g. ``"user"``).
    output_dir:
        Directory where generated files are written.  It will be created if it
        does not already exist.

    Returns
    -------
    ``List[Path]``
        Paths to all generated files, in creation order.
    """
    count = _parse_quantity(prompt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for idx in range(1, count + 1):
        xml = _generate_single(prompt, schema)
        file_path = output_dir / f"{schema}_{idx}.xml"
        file_path.write_text(xml)
        generated.append(file_path)
    return generated
