import csv
from dataclasses import dataclass, field
from random import Random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

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
DEFAULT_CURRENCIES = ["USD", "EUR", "JPY"]


@dataclass
class SyntheticDataGenerator:
    """Generate synthetic natural-language/XML pairs.

    The generator can be configured with custom value pools. A dedicated
    :class:`random.Random` instance is used so callers may supply a seed for
    deterministic outputs.
    """

    names: Sequence[str] = field(default_factory=lambda: NAMES)
    products: Sequence[str] = field(default_factory=lambda: PRODUCTS)
    currencies: Sequence[str] = field(default_factory=lambda: DEFAULT_CURRENCIES)
    rng: Random = field(default_factory=Random)

    def user_example(self) -> Tuple[str, str]:
        name = self.rng.choice(self.names)
        if self.rng.random() < 0.6:
            name += f" {self.rng.choice(self.names)}"
        uid = str(self.rng.randint(100, 9999))
        email = f"{name.split()[0].lower()}@example.com" if self.rng.random() < 0.7 else None
        nl = self.rng.choice(
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

    def product_example(self) -> Tuple[str, str]:
        name = self.rng.choice(self.products)
        sku = f"SKU-{self.rng.randint(1000, 9999)}"
        price = f"{self.rng.randint(1, 999)}.{self.rng.randint(0, 99):02d}"
        cur = self.rng.choice(self.currencies)
        nl = self.rng.choice(
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

    def order_example(self) -> Tuple[str, str]:
        oid = str(self.rng.randint(100, 99999))
        user = self.rng.choice(self.names)
        total = f"{self.rng.randint(10, 4999)}.{self.rng.randint(0, 99):02d}"
        nl = self.rng.choice(
            [
                f"Create an order {oid} for user {user} totaling {total}",
                f"Add order id {oid} belonging to {user}, total {total}",
                f"Register order {oid} for {user} with total {total}",
            ]
        )
        xml = f"<order><id>{oid}</id><user>{user}</user><total>{total}</total></order>"
        return nl, xml

    @property
    def generators(self) -> List[Callable[[], Tuple[str, str]]]:
        return [self.user_example, self.product_example, self.order_example]

    def generate_dataset(self, n: int = N, out_path: Path = OUT) -> Path:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "text_output"])
            for _ in range(n):
                nl, xml = self.rng.choice(self.generators)()
                writer.writerow([nl, xml])
        return out_path


def generate_dataset(
    n: int = N,
    out_path: Path = OUT,
    *,
    names: Sequence[str] | None = None,
    products: Sequence[str] | None = None,
    currencies: Sequence[str] | None = None,
    seed: int | None = None,
) -> Path:
    """Generate a synthetic dataset and write it to ``out_path``.

    Args:
        n: Number of records to create.
        out_path: Destination path for the CSV file.
        names: Optional pool of user names.
        products: Optional pool of product names.
        currencies: Optional pool of currency codes.
        seed: Optional seed for deterministic output.

    Returns:
        Path to the generated CSV file.
    """
    gen = SyntheticDataGenerator(
        names=names or NAMES,
        products=products or PRODUCTS,
        currencies=currencies or DEFAULT_CURRENCIES,
        rng=Random(seed),
    )
    return gen.generate_dataset(n, out_path)


if __name__ == "__main__":
    generate_dataset()

