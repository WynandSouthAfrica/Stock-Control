# OMEC Stock Take (Streamlit)

A simple, robust stock take and maintenance app you can run locally or on an internal server.
It stores data in a local SQLite database (in `data/stocktake.db`), supports versioned exports as ZIP files,
and lets you brand with PG Bison or OMEC logos.

## Quick start

```bash
cd OMEC_Stock_Take_App
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Key Features
- Inventory items (SKU, name, category, location, unit, min qty, unit cost)
- Transaction log (in/out, reason, project, reference, user)
- Versioned snapshots to ZIP (CSV + JSON + summary)
- Brand with your logo (Settings page)
- Simple maintenance log per item
- Import/Export CSV

## Files
- `app.py`: Main entry with navigation
- `pages/`: Additional pages (Inventory, Transactions, Versions, Reports, Maintenance, Settings)
- `data/`: SQLite DB & temp exports
- `exports/`: Timestamped ZIP snapshots
- `assets/`: Logos
- `config.json`: Basic branding config
