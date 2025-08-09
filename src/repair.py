import re


def extract_user_slots(text: str) -> str:
    name = uid = email = None
    m = re.search(r"\b(id|ID)\s*[:=#]?\s*([A-Za-z0-9-]+)", text)
    uid = m.group(2) if m else None
    m = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    email = m.group(1) if m else None
    m = re.search(r"(?:named|name\s*[:=#])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    name = m.group(1) if m else None
    if not name:
        m = re.search(r"user\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
        name = m.group(1) if m else None
    name = name or "Unknown"
    uid = uid or "000"
    xml = f"<user><id>{uid}</id><name>{name}</name>"
    if email:
        xml += f"<email>{email}</email>"
    xml += "</user>"
    return xml


def extract_product_slots(text: str) -> str:
    sku = name = price = currency = None
    m = re.search(r"\bSKU[-\s:]?([A-Za-z0-9-]+)", text, re.IGNORECASE)
    if m:
        sku_value = m.group(1)
        sku = sku_value if sku_value.upper().startswith("SKU") else f"SKU-{sku_value}"
    m = re.search(r"(\d+\.\d{2}|\d+)\s*(USD|EUR|JPY)?", text, re.IGNORECASE)
    if m:
        price, currency = m.group(1), (m.group(2) or "USD").upper()
    m = re.search(r"(?:called|product\s+)([A-Z][\w\s-]+)", text)
    if m:
        name = m.group(1).strip()
    return (
        f"<product><sku>{sku or 'SKU-0000'}</sku><name>{name or 'Unnamed'}</name>"
        f"<price>{price or '0.00'}</price><currency>{currency or 'USD'}</currency></product>"
    )


def extract_order_slots(text: str) -> str:
    oid = user = total = None
    m = re.search(r"\b(order\s*#?|id)\s*[:=#]?\s*([A-Za-z0-9-]+)", text, re.IGNORECASE)
    if m:
        oid = m.group(2)
    m = re.search(
        r"(?:total(?:ing)?|amount|sum)\s*[:=#]?\s*(\d+\.\d{2}|\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        total = m.group(1)
    m = re.search(r"\buser\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text) or re.search(
        r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    )
    if m:
        user = m.group(1)
    return f"<order><id>{oid or '0'}</id><user>{user or 'Unknown'}</user><total>{total or '0.00'}</total></order>"


def repair_to_schema(text: str, schema: str) -> str:
    s = schema.lower()
    if s == "user":
        return extract_user_slots(text)
    if s == "product":
        return extract_product_slots(text)
    if s == "order":
        return extract_order_slots(text)
    raise ValueError(f"Unknown schema: {schema}")
