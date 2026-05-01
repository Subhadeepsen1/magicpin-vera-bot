#!/usr/bin/env python3
"""
Generate submission.jsonl — produce messages for the 30 canonical test pairs.

Usage:
    python generate_submission.py

Reads test_pairs.json from the expanded dataset and composes a message for each.
"""

import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.composer import compose

DATASET_DIR = Path(__file__).parent.parent / "data" / "dataset" / "expanded"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    # Load test pairs
    test_pairs_path = DATASET_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"ERROR: {test_pairs_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    test_pairs = load_json(test_pairs_path)["pairs"]
    print(f"Loaded {len(test_pairs)} test pairs")

    # Load categories
    categories = {}
    cat_dir = DATASET_DIR / "categories"
    for f in cat_dir.glob("*.json"):
        data = load_json(f)
        categories[data["slug"]] = data

    # Load merchants
    merchants = {}
    merch_dir = DATASET_DIR / "merchants"
    for f in merch_dir.glob("*.json"):
        data = load_json(f)
        merchants[data["merchant_id"]] = data

    # Load customers
    customers = {}
    cust_dir = DATASET_DIR / "customers"
    for f in cust_dir.glob("*.json"):
        data = load_json(f)
        customers[data["customer_id"]] = data

    # Load triggers
    triggers = {}
    trig_dir = DATASET_DIR / "triggers"
    for f in trig_dir.glob("*.json"):
        data = load_json(f)
        triggers[data["id"]] = data

    print(f"Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    # Generate submissions
    output_path = Path(__file__).parent.parent / "data" / "submission.jsonl"
    results = []

    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        trigger = triggers.get(trigger_id, {})
        merchant = merchants.get(merchant_id, {})
        category_slug = merchant.get("category_slug", trigger.get("payload", {}).get("category", ""))
        category = categories.get(category_slug, {})
        customer = customers.get(customer_id) if customer_id else None

        print(f"\n[{test_id}] {trigger.get('kind', '?')} → {merchant.get('identity', {}).get('name', '?')}")

        try:
            composed = compose(category, merchant, trigger, customer)
            result = {
                "test_id": test_id,
                "body": composed["body"],
                "cta": composed["cta"],
                "send_as": composed["send_as"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }
            print(f"  ✓ Body: {composed['body'][:80]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                "test_id": test_id,
                "body": f"Hi, checking in with an update for your business.",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Fallback: {str(e)[:100]}",
            }

        results.append(result)
        import time
        time.sleep(3)  # Respect Groq 30 RPM limit

    # Write JSONL
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Written {len(results)} submissions to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
