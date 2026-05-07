"""
Database Exploration CL Task.

Four sequential regimes simulating schema drift across an inventory database:
  0 — schema_v1       (simple: products, stores, sales tables)
  1 — schema_v2       (column renames: 'qty' → 'quantity', 'amt' → 'amount')
  2 — schema_v3       (table split: sales split into sales + returns)
  3 — schema_v4       (schema merge: stores + regions collapsed into locations)

Each regime = 30 questions.
Each episode: LLM sees current schema + synthetic data summary + NL question
              → answers with a single integer or short value.
Scoring rewards exact numeric answers; ≥10% relative error scores 0.
"""
import random
import re
from dataclasses import dataclass
from typing import Iterator

EPISODES_PER_REGIME = 30

_REGIMES = ["schema_v1", "schema_v2", "schema_v3", "schema_v4"]

# Schema snapshots shown to the model
_SCHEMAS = {
    "schema_v1": """\
Tables:
  products(id, name, category, unit_price)
  stores(id, city, region)
  sales(id, product_id, store_id, qty, amt, sale_date)""",

    "schema_v2": """\
Tables:
  products(id, name, category, unit_price)
  stores(id, city, region)
  sales(id, product_id, store_id, quantity, amount, sale_date)
Note: 'qty' renamed to 'quantity'; 'amt' renamed to 'amount'.""",

    "schema_v3": """\
Tables:
  products(id, name, category, unit_price)
  stores(id, city, region)
  sales(id, product_id, store_id, quantity, amount, sale_date)
  returns(id, sale_id, quantity, reason, return_date)
Note: returns are now tracked in a separate table.""",

    "schema_v4": """\
Tables:
  products(id, name, category, unit_price)
  locations(id, city, region, type)   -- merged stores + regions
  sales(id, product_id, location_id, quantity, amount, sale_date)
  returns(id, sale_id, quantity, reason, return_date)
Note: stores and regions merged into locations; store_id → location_id.""",
}

_QUESTION_TEMPLATES = [
    ("How many {category} products were sold across all stores in {month}?",         "total_units"),
    ("What is the total revenue from {city} in {month}?",                            "total_revenue"),
    ("How many distinct products were sold in region {region} during {month}?",      "distinct_products"),
    ("What was the average sale amount per transaction in {city} for {month}?",      "avg_amount"),
    ("How many stores recorded sales of more than {threshold} units in {month}?",    "stores_above_threshold"),
]


@dataclass
class Episode:
    task_id: int
    regime: str
    target: float
    text: str


def _generate_data(rng: random.Random, question_type: str, **kwargs) -> tuple[str, float]:
    """Return (data_summary_string, ground_truth_answer)."""
    if question_type == "total_units":
        n = rng.randint(120, 2400)
        summary = f"  {kwargs['category']} units sold in {kwargs['month']}: {n}"
        return summary, float(n)
    if question_type == "total_revenue":
        rev = rng.randint(5000, 80000)
        summary = f"  Revenue from {kwargs['city']} in {kwargs['month']}: ${rev:,}"
        return summary, float(rev)
    if question_type == "distinct_products":
        n = rng.randint(8, 60)
        summary = f"  Distinct products sold in region {kwargs['region']} in {kwargs['month']}: {n}"
        return summary, float(n)
    if question_type == "avg_amount":
        avg = round(rng.uniform(25.0, 350.0), 2)
        summary = f"  Average transaction amount in {kwargs['city']} for {kwargs['month']}: ${avg:.2f}"
        return summary, float(round(avg))
    # stores_above_threshold
    n = rng.randint(1, 20)
    summary = f"  Stores with >{kwargs['threshold']} units in {kwargs['month']}: {n}"
    return summary, float(n)


def _fill_template(template: str, rng: random.Random) -> tuple[str, dict]:
    cities     = ["Austin", "Denver", "Seattle", "Miami", "Chicago"]
    regions    = ["West", "East", "Central", "South"]
    categories = ["furniture", "electronics", "office", "outdoor"]
    months     = ["January", "February", "March", "April", "May", "June"]
    kwargs = {
        "city":      rng.choice(cities),
        "region":    rng.choice(regions),
        "category":  rng.choice(categories),
        "month":     rng.choice(months),
        "threshold": rng.choice([100, 200, 500, 1000]),
    }
    return template.format(**kwargs), kwargs


def _format_prompt(schema: str, data_summary: str, question: str) -> str:
    return (
        "You are analysing a retail database.\n\n"
        f"Current schema:\n{schema}\n\n"
        f"Data snapshot:\n{data_summary}\n\n"
        f"Question: {question}\n\n"
        "Respond with a single integer (round floats to nearest whole number). "
        "Do not include units, symbols, or explanation."
    )


def _parse(response: str) -> float | None:
    cleaned = response.replace(",", "").replace("$", "").strip()
    m = re.search(r"\b(\d+)\b", cleaned)
    return float(m.group(1)) if m else None


class DatabaseExplorationTask:
    """Four-regime CL task with progressive database schema drift."""

    def __init__(self, seed: int = 42):
        self._regimes = self._build(seed)

    def _build(self, seed: int) -> list[list[Episode]]:
        regimes = []
        for task_id, regime in enumerate(_REGIMES):
            rng    = random.Random(seed + task_id * 17)
            schema = _SCHEMAS[regime]
            episodes = []
            for _ in range(EPISODES_PER_REGIME):
                tmpl, q_type = rng.choice(_QUESTION_TEMPLATES)
                question, kwargs = _fill_template(tmpl, rng)
                summary, target  = _generate_data(rng, q_type, **kwargs)
                text = _format_prompt(schema, summary, question)
                episodes.append(Episode(task_id=task_id, regime=regime, target=target, text=text))
            regimes.append(episodes)
        return regimes

    @property
    def num_tasks(self) -> int:
        return len(self._regimes)

    def iter_episodes(self, task_id: int) -> Iterator[Episode]:
        yield from self._regimes[task_id]

    @staticmethod
    def score(response: str, target: float) -> float:
        """1.0 = exact; 0.0 = ≥10% relative error or unparseable."""
        pred = _parse(response)
        if pred is None:
            return 0.0
        return max(0.0, 1.0 - abs(pred - target) / max(abs(target), 1.0))
