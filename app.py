# app.py — OMEC Inventory Manager (Quick Edit removed)
# Streamlit >= 1.31 recommended

import os
import io
import time
from datetime import datetime
import pandas as pd
import streamlit as st

__version__ = "v14.0.3"

# ───────────────────────── Page Setup ─────────────────────────
st.set_page_config(
    page_title="OMEC Inventory Manager",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── Utilities ─────────────────────────
DATA_DIR_DEFAULT = os.path.join(os.getcwd(), "data")
DB_FILE_DEFAULT = os.path.join(DATA_DIR_DEFAULT, "inventory_db.csv")

EXPECTED_COLS = [
    "SKU",
    "Name",
    "Category",
    "Location",
    "Unit",
    "Quantity",
    "Min Qty",
    "Unit Cost (R)",
    "Notes",
    "Conversion to",
    "Conversion factor",
]

NUM_COLS = ["Quantity", "Min Qty", "Unit Cost (R)", "Conversion factor"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected columns exist and are ordered
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0.0 if col in NUM_COLS else ""
    df = df[EXPECTED_COLS].copy()

    # Types
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Sort: Category → SKU (alphabetical)
    df["Category"] = df["Category"].astype(str)
    df["SKU"] = df["SKU"].astype(str)
    df = df.sort_values(["Category", "SKU"], kind="stable", na_position="last").reset_index(drop=True)
    return df

def load_db(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        ensure_dir(os.path.dirname(path))
        df = pd.DataFrame(columns=EXPECTED_COLS)
        df.to_csv(path, index=False)
    return normalise_df(df)

def save_db(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    normalise_df(df).to_csv(path, index=False)

def upsert_rows(db: pd.DataFrame, rows: pd.DataFrame, key_col="SKU") -> pd.DataFrame:
    """Replace/insert rows from 'rows' into 'db' by key."""
    base = normalise_df(db).copy()
    patch = normalise_df(rows).copy()

    base[key_col] = base[key_col].astype(str).str.strip()
    patch[key_col] = patch[key_col].astype(str).str.strip()
    patch = patch[patch[key_col] != ""]
    patch = patch.drop_duplicates(subset=[key_col], keep="last")

    # Remove existing keys and append
    base = base[~base[key_col].isin(patch[key_col])]
    out = pd.concat([base, patch], ignore_index=True)
    return normalise_df(out)

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    return normalise_df(df).to_csv(index=False).encode("utf-8-sig")

def export_xlsx_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        normalise_df(df).to_excel(writer, index=False, sheet_name="Inventory")
    return bio.getvalue()

def snapshot_save(folder: str, df: pd.DataFrame) -> str:
    ensure_dir(folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(folder, f"inventory_snapshot_{ts}.csv")
    normalise_df(df).to_csv(fpath, index=False)
    return fpath

# ───────────────────────── Session State ─────────────────────────
if "db_path" not in st.session_state:
    st.session_state.db_path = DB_FILE_DEFAULT
if "snapshot_dir" not in st.session_state:
    st.session_state.snapshot_dir = DATA_DIR_DEFAULT
if "db" not in st.session_state:
    st.session_state.db = load_db(st.session_state.db_path)
if "filters" not in st.session_state:
    st.session_state.filters = dict(category=[], location=[], search="", below_min=False)

# ───────────────────────── Header ─────────────────────────
hdrL, hdrC, hdrR = st.columns([1, 3, 1])
with hdrL:
    try:
        st.image("/mnt/data/Logo R0.1.png", use_container_width=True)
    except Exception:
        st.write("")
with hdrC:
    st.markdown(
        f"<h2 style='text-align:center;margin:0;'>OMEC Inventory Manager</h2>"
        f"<div style='text-align:center;opacity:0.7;'>PG Bison • {__version__}</div>",
        unsafe_allow_html=True,
    )
with hdrR:
    try:
        st.image("/mnt/data/PG Bison.jpg", use_container_width=True)
    except Exception:
        st.write("")
st.divider()

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.subheader("Data Locations")
    st.text_input("Database CSV path", key="db_path")
    st.text_input("Snapshot folder", key="snapshot_dir")

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("Reload DB"):
            st.session_state.db = load_db(st.session_state.db_path)
    with col_sb2:
        if st.button("Save DB"):
            save_db(st.session_state.db, st.session_state.db_path)

    st.subheader("Export (Filtered)")
    filtered_preview = st.session_state.get("filtered_df", st.session_state.db)
    st.download_button("CSV", data=export_csv_bytes(filtered_preview), file_name="inventory_filtered.csv")
    st.download_button("XLSX", data=export_xlsx_bytes(filtered_preview), file_name="inventory_filtered.xlsx")

    st.subheader("Snapshots")
    if st.button("Save Snapshot"):
        path = snapshot_save(st.session_state.snapshot_dir, st.session_state.db)
        st.toast(f"Snapshot saved:\n{path}", icon="💾")
    snap_file = st.file_uploader("Load Snapshot (CSV)", type=["csv"])
    if snap_file is not None:
        snap_df = pd.read_csv(snap_file)
        st.session_state.db = normalise_df(snap_df)
        st.toast("Snapshot loaded into working DB", icon="✅")

# ───────────────────────── Main (Quick Edit removed) ─────────────────────────
tab_list, tab_about = st.tabs(["📋 Inventory List (Editable)", "ℹ️ About"])

with tab_list:
    db = st.session_state.db

    # Filters row
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2, 2, 3, 2, 2])
    cats = sorted([c for c in db["Category"].dropna().unique() if str(c).strip() != ""])
    locs = sorted([c for c in db["Location"].dropna().unique() if str(c).strip() != ""])

    with fcol1:
        st.session_state.filters["category"] = st.multiselect(
            "Filter: Category", cats, default=st.session_state.filters["category"]
        )
    with fcol2:
        st.session_state.filters["location"] = st.multiselect(
            "Filter: Location", locs, default=st.session_state.filters["location"]
        )
    with fcol3:
        st.session_state.filters["search"] = st.text_input(
            "Search (SKU / Name / Notes)", value=st.session_state.filters["search"]
        )
    with fcol4:
        st.session_state.filters["below_min"] = st.toggle(
            "Only below Min", value=st.session_state.filters["below_min"]
        )
    with fcol5:
        n_new = st.number_input("Add empty rows", min_value=0, value=0, step=1)
        if st.button("➕ Add"):
            if n_new > 0:
                blank = pd.DataFrame([{c: (0.0 if c in NUM_COLS else "") for c in EXPECTED_COLS} for _ in range(int(n_new))])
                st.session_state.db = pd.concat([blank, st.session_state.db], ignore_index=True)
                st.session_state.db = normalise_df(st.session_state.db)
                st.rerun()

    # Apply filters
    filt = st.session_state.filters
    filtered = db.copy()
    if filt["category"]:
        filtered = filtered[filtered["Category"].isin(filt["category"])]
    if filt["location"]:
        filtered = filtered[filtered["Location"].isin(filt["location"])]
    if (q := filt["search"].strip()):
        q = q.lower()
        mask = (
            filtered["SKU"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["Name"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["Notes"].astype(str).str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]
    if filt["below_min"]:
        qty = pd.to_numeric(filtered["Quantity"], errors="coerce").fillna(0)
        mn = pd.to_numeric(filtered["Min Qty"], errors="coerce").fillna(0)
        filtered = filtered[qty < mn]

    st.session_state.filtered_df = filtered.copy()
    st.caption(f"{len(filtered):,} of {len(db):,} rows shown • Sorted by Category → SKU")

    # ── Editable grid (window increased) ──
    edited = st.data_editor(
        filtered,
        use_container_width=True,
        height=900,  # increased window height
        num_rows="dynamic",
        key="inventory_editor",
        column_config={
            "SKU": st.column_config.TextColumn("SKU", required=True, help="Unique stock code"),
            "Name": st.column_config.TextColumn("Name"),
            "Category": st.column_config.TextColumn("Category"),
            "Location": st.column_config.TextColumn("Location"),
            "Unit": st.column_config.SelectboxColumn(
                "Unit",
                options=["ea","mm","m","m²","kg","sheet","bar","plate","pc","box"],
                default="ea",
            ),
            "Quantity": st.column_config.NumberColumn("Quantity", step=0.001, format="%.3f"),
            "Min Qty": st.column_config.NumberColumn("Min Qty", step=0.001, format="%.3f"),
            "Unit Cost (R)": st.column_config.NumberColumn("Unit Cost (R)", step=0.01, format="%.2f"),
            "Notes": st.column_config.TextColumn("Notes"),
            "Conversion to": st.column_config.TextColumn("Convert to"),
            "Conversion factor": st.column_config.NumberColumn("Conversion factor", step=0.001, format="%.3f"),
        },
    )

    # Actions
    a1, a2, a3, _ = st.columns([2, 2, 2, 6])
    with a1:
        if st.button("💾 Save Edits (in-memory)"):
            st.session_state.db = upsert_rows(db, edited, key_col="SKU")
            st.toast("Edits saved (in-memory)", icon="✅")
            time.sleep(0.05)
            st.rerun()
    with a2:
        if st.button("⬆️ Upsert Visible Rows to DB File"):
            st.session_state.db = upsert_rows(db, edited, key_col="SKU")
            save_db(st.session_state.db, st.session_state.db_path)
            st.toast("Visible rows upserted to DB file", icon="📀")
    with a3:
        if st.button("🧹 Remove Empty SKU Rows"):
            cleaned = st.session_state.db.copy()
            cleaned["SKU"] = cleaned["SKU"].astype(str)
            cleaned = cleaned[cleaned["SKU"].str.strip() != ""]
            st.session_state.db = normalise_df(cleaned)
            st.toast("Removed empty SKU rows", icon="🗑️")
            st.rerun()

with tab_about:
    st.markdown(
        f"""
**About**

- Company: PG Bison • OMEC Tools  
- Module: Inventory Manager (Quick Edit removed)  
- Version: **{__version__}**  

Edit directly in the grid. Use **Save Edits** to keep changes in memory and **Upsert Visible Rows** to persist to the CSV database.
"""
    )
