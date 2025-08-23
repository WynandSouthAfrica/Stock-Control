# OMEC Stock Take (Single-file Streamlit App)

This variant avoids Streamlit's `pages/` feature and keeps everything in **app.py** with a sidebar menu.
Ideal when emoji filenames or the `pages` folder cause issues on Windows/GitHub.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Data stored in `data/stocktake.db`. Snapshots saved as timestamped ZIPs in `exports/`.