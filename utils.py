
import os, io, json, csv, zipfile, datetime
from typing import List, Dict
import pandas as pd

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "exports")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def export_snapshot(items: List[Dict], transactions: List[Dict], tag: str = "", note: str = "") -> str:
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    ts = timestamp()
    base = f"StockTake_{ts}" + (f"_{tag}" if tag else "")
    zip_path = os.path.join(EXPORTS_DIR, base + ".zip")

    # Prepare CSV/JSON in-memory
    items_df = pd.DataFrame(items)
    tx_df = pd.DataFrame(transactions)

    items_csv = items_df.to_csv(index=False).encode("utf-8")
    tx_csv = tx_df.to_csv(index=False).encode("utf-8")

    items_json = items_df.to_json(orient="records", indent=2).encode("utf-8")
    tx_json = tx_df.to_json(orient="records", indent=2).encode("utf-8")

    summary = [
        f"Snapshot: {ts}",
        f"Tag: {tag}",
        f"Note: {note}",
        f"Items: {len(items_df)}",
        f"Transactions included: {len(tx_df)}",
    ]
    summary_text = "\n".join(summary).encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("items.csv", items_csv)
        z.writestr("transactions.csv", tx_csv)
        z.writestr("items.json", items_json)
        z.writestr("transactions.json", tx_json)
        z.writestr("README.txt", summary_text)

    return zip_path
