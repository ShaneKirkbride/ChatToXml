import csv
import random
from pathlib import Path

# Default location for the generated dataset
OUT = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Default number of synthetic examples
N = 2500

NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Hank",
    "Ivy",
    "Jules",
]
PRODUCTS = [
    "Widget X",
    "Gizmo 2000",
    "UltraHammer",
    "NanoLight",
    "Quantum Cup",
    "ThermoPad",
]
CURRENCIES = ["USD", "EUR", "JPY"]


def user_example() -> tuple[str, str]:
    name = random.choice(NAMES)
    if random.random() < 0.6:
        name += f" {random.choice(NAMES)}"
    uid = str(random.randint(100, 9999))
    email = f"{name.split()[0].lower()}@example.com" if random.random() < 0.7 else None
    nl = random.choice(
        [
            f"Create a user named {name} with ID {uid}",
            f"Register user {name}, id {uid}",
            f"Add a new user: name={name}, id={uid}",
            f"Set up user {name} ({uid})",
        ]
    )
    xml = f"<user><id>{uid}</id><name>{name}</name>"
    if email:
        xml += f"<email>{email}</email>"
    xml += "</user>"
    return nl, xml


def product_example() -> tuple[str, str]:
    name = random.choice(PRODUCTS)
    sku = f"SKU-{random.randint(1000, 9999)}"
    price = f"{random.randint(1, 999)}.{random.randint(0, 99):02d}"
    cur = random.choice(CURRENCIES)
    nl = random.choice(
        [
            f"Add a product called {name} priced at {price} {cur} with sku {sku}",
            f"Register product {name}; price {price} {cur}; SKU {sku}",
            f"Create product {name} (sku {sku}) costing {price} {cur}",
        ]
    )
    xml = (
        f"<product><sku>{sku}</sku><name>{name}</name>"
        f"<price>{price}</price><currency>{cur}</currency></product>"
    )
    return nl, xml


def order_example() -> tuple[str, str]:
    oid = str(random.randint(100, 99999))
    user = random.choice(NAMES)
    total = f"{random.randint(10, 4999)}.{random.randint(0, 99):02d}"
    nl = random.choice(
        [
            f"Create an order {oid} for user {user} totaling {total}",
            f"Add order id {oid} belonging to {user}, total {total}",
            f"Register order {oid} for {user} with total {total}",
        ]
    )
    xml = f"<order><id>{oid}</id><user>{user}</user><total>{total}</total></order>"
    return nl, xml


GENERATORS = [user_example, product_example, order_example]


def generate_dataset(n: int = N, out_path: Path = OUT) -> Path:
    """Generate a synthetic dataset and write it to ``out_path``.

    Args:
        n: Number of records to create.
        out_path: Destination path for the CSV file.

    Returns:
        Path to the generated CSV file.
    """
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "text_output"])
        for _ in range(n):
            nl, xml = random.choice(GENERATORS)()
            writer.writerow([nl, xml])
    return out_path


if __name__ == "__main__":
    generate_dataset()
