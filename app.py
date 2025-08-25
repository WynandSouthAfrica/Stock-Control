# app.py — OMEC Inventory Manager (Quick Edit removed)
# Streamlit ≥ 1.31 recommended

import os
import io
import time
from datetime import datetime
import pandas as pd
import streamlit as st

__version__ = "v14.0.2"

# ---------- Page Setup ----------
st.set_page_config(
    page_title="OMEC Inventory Manager",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Utility ----------
DATA_DIR_DEFAULT = os.path.join(os.getcwd(), "data")
DB_FILE_DEFAULT = os.path.join(DATA_DIR_DEFAULT, "inventory_db.csv")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_db(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Make sure expected columns exist
        expected = [
            "SKU","Name","Category","Location","Unit",
            "Quantity","Min Qty","Unit Cost (R)","Notes",
            "Conversion to","Conversion factor"
        ]
        for col in expected:
            if col not in df.columns:
                df[col] = "" if col not in ["Quantity","Min Qty","Unit Cost (R)","Conversion factor"] else 0.0
        df = df[expected]
    else:
        ensure_dir(os.path.dirname(path))
        df = pd.DataFrame(
            columns=[
                "SKU","Name","Category","Location","Unit",
                "Quantity","Min Qty","Unit Cost (R)","Notes",
                "Conversion to","Conversion factor",
            ]
        )
        df.to_csv(path, index=False)
    # Normalise types
    num_cols = ["Quantity","Min Qty","Unit Cost (R)","Conversion factor"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Always display sorted by Category then SKU (user request)
    return df.sort_values(["Category","SKU"], kind="stable", na_position="last").reset_index(drop=True)

def save_db(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def upsert_rows(db: pd.DataFrame, rows: pd.DataFrame, key_col="SKU") -> pd.DataFrame:
    rows = rows.copy()
    rows = rows.dropna(subset=[key_col])
    rows[key_col] = rows[key_col].astype(str)
    db[key_col] = db[key_col].astype(str)

    # Remove empty keys
    rows = rows[rows[key_col].str.strip() != ""]
    if rows.empty:
        return db

    # Use last occurrence per key in the visible set
    rows = rows.drop_duplicates(subset=[key_col], keep="last")

    # Split new vs existing
    existing_mask = db[key_col].isin(rows[key_col])
    db_existing = db[existing_mask].copy()
    db_newkeep = db[~existing_mask].copy()

    # Align columns and update existing
    aligned = rows.reindex(columns=db.columns)
    updated = db_existing.drop(columns=db.columns, errors="ignore")
    updated = aligned.set_index(key_col).combine_first(db_existing.set_index(key_col)).reset_index()

    out = pd.concat([db_newkeep, updated, aligned[~aligned[key_col].isin(db_existing[key_col])]], ignore_index=True)
    # Resort for display
    out = out.sort_values(["Category","SKU"], kind="stable", na_position="last").reset_index(drop=True)
    return out

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def export_xlsx_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Inventory")
    return bio.getvalue()

def snapshot_save(folder: str, df: pd.DataFrame) -> str:
    ensure_dir(folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"inventory_snapshot_{ts}.csv"
    fpath = os.path.join(folder, fname)
    df.to_csv(fpath, index=False)
    return fpath

# ---------- Session ----------
if "db_path" not in st.session_state:
    st.session_state.db_path = DB_FILE_DEFAULT
if "snapshot_dir" not in st.session_state:
    st.session_state.snapshot_dir = DATA_DIR_DEFAULT
if "db" not in st.session_state:
    st.session_state.db = load_db(st.session_state.db_path)
if "filters" not in st.session_state:
    st.session_state.filters = {
        "category": [],
        "location": [],
        "search": "",
        "show_below_min": False,
    }

# ---------- Header ----------
with st.container():
    colL, colC, colR = st.columns([1,3,1])
    with colL:
        try:
            st.image("/mnt/data/Logo R0.1.png", use_container_width=True)
        except Exception:
            st.write("")
    with colC:
        st.markdown(
            f"<h2 style='text-align:center;margin:0;'>OMEC Inventory Manager</h2>"
            f"<div style='text-align:center;opacity:0.7;'>PG Bison • {__version__}</div>",
            unsafe_allow_html=True,
        )
    with colR:
        try:
            st.image("/mnt/data/PG Bison.jpg", use_container_width=True)
        except Exception:
            st.write("")

st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Data Locations")
    st.text_input("Database CSV path", key="db_path")
    st.text_input("Snapshot folder", key="snapshot_dir")
    col_sb_a, col_sb_b = st.columns(2)
    with col_sb_a:
        if st.button("Reload DB"):
            st.session_state.db = load_db(st.session_state.db_path)
    with col_sb_b:
        if st.button("Save DB"):
            save_db(st.session_state.db, st.session_state.db_path)

    st.subheader("Export")
    filtered_preview = st.session_state.get("filtered_df", st.session_state.db)
    st.download_button("Download CSV (filtered)", data=export_csv_bytes(filtered_preview), file_name="inventory_filtered.csv")
    st.download_button("Download XLSX (filtered)", data=export_xlsx_bytes(filtered_preview), file_name="inventory_filtered.xlsx")

    st.subheader("Snapshots")
    if st.button("Save Snapshot"):
        path = snapshot_save(st.session_state.snapshot_dir, st.session_state.db)
        st.toast(f"Snapshot saved:\n{path}", icon="💾")
    uploader = st.file_uploader("Load Snapshot (CSV)", type=["csv"], accept_multiple_files=False)
    if uploader is not None:
        snap_df = pd.read_csv(uploader)
        # Ensure columns
        snap_df = snap_df.reindex(columns=st.session_state.db.columns, fill_value="")
        st.session_state.db = snap_df
        st.toast("Snapshot loaded into working DB", icon="✅")

# ---------- Main Tabs (Quick Edit removed) ----------
tab1, tab2 = st.tabs(["📋 Inventory List (Editable)", "ℹ️ About"])

with tab1:
    db = st.session_state.db

    # Filters row
    filt_cols = st.columns([2,2,3,2,2])
    unique_cat = sorted([c for c in db["Category"].dropna().unique() if str(c).strip() != ""])
    unique_loc = sorted([c for c in db["Location"].dropna().unique() if str(c).strip() != ""])
    with filt_cols[0]:
        st.session_state.filters["category"] = st.multiselect("Filter: Category", unique_cat, default=st.session_state.filters["category"])
    with filt_cols[1]:
        st.session_state.filters["location"] = st.multiselect("Filter: Location", unique_loc, default=st.session_state.filters["location"])
    with filt_cols[2]:
        st.session_state.filters["search"] = st.text_input("Search (SKU / Name / Notes)", value=st.session_state.filters["search"])
    with filt_cols[3]:
        st.session_state.filters["show_below_min"] = st.toggle("Only below Min", value=st.session_state.filters["show_below_min"])
    with filt_cols[4]:
        add_rows_n = st.number_input("Add empty rows", min_value=0, step=1, value=0)
        if st.button("➕ Add"):
            if add_rows_n > 0:
                blank = pd.DataFrame([{c:"" for c in db.columns} for _ in range(int(add_rows_n))])
                for c in ["Quantity","Min Qty","Unit Cost (R)","Conversion factor"]:
                    blank[c] = 0.0
                st.session_state.db = pd.concat([blank, db], ignore_index=True)
                st.rerun()

    # Apply filters
    f = st.session_state.filters
    filtered = db.copy()
    if f["category"]:
        filtered = filtered[filtered["Category"].isin(f["category"])]
    if f["location"]:
        filtered = filtered[filtered["Location"].isin(f["location"])]
    if f["search"].strip():
        q = f["search"].strip().lower()
        mask = (
            filtered["SKU"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["Name"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["Notes"].astype(str).str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]
    if f["show_below_min"]:
        filtered = filtered[(pd.to_numeric(filtered["Quantity"], errors="coerce").fillna(0) <= pd.to_numeric(filtered["Min Qty"], errors="coerce").fillna(0))]

    # Keep a copy for sidebar exports
    st.session_state.filtered_df = filtered.copy()

    st.caption(f"{len(filtered):,} of {len(db):,} rows shown • Sorted by Category → SKU")

    # ---- Editable grid (increased window height) ----
    edited = st.data_editor(
        filtered,
        use_container_width=True,
        height=820,  # Increased window height as requested
        num_rows="dynamic",
        key="inventory_editor",
        column_config={
            "SKU": st.column_config.TextColumn("SKU", required=True, help="Unique stock code"),
            "Name": st.column_config.TextColumn("Name"),
            "Category": st.column_config.TextColumn("Category"),
            "Location": st.column_config.TextColumn("Location"),
            "Unit": st.column_config.SelectboxColumn("Unit", options=["mm","m","m²","kg","ea","sheet","bar","plate","pc","box"], default="ea"),
            "Quantity": st.column_config.NumberColumn("Quantity", step=0.001, format="%.3f"),
            "Min Qty": st.column_config.NumberColumn("Min Qty", step=0.001, format="%.3f"),
            "Unit Cost (R)": st.column_config.NumberColumn("Unit Cost (R)", step=0.01, format="%.2f"),
            "Notes": st.column_config.TextColumn("Notes"),
            "Conversion to": st.column_config.TextColumn("Convert to"),
            "Conversion factor": st.column_config.NumberColumn("Conversion factor", step=0.001, format="%.3f"),
        },
    )

    # Actions row
    c1, c2, c3, c4 = st.columns([2,2,2,6])
    with c1:
        if st.button("💾 Save Edits (in-memory)"):
            # Replace filtered slice inside the DB with edited version by SKU
            # We upsert back only for SKUs present in 'edited' slice
            st.session_state.db = upsert_rows(db, edited, key_col="SKU")
            st.toast("Edits saved (in-memory)", icon="✅")
            time.sleep(0.1)
            st.rerun()
    with c2:
        if st.button("⬆️ Upsert Visible Rows to DB File"):
            st.session_state.db = upsert_rows(db, edited, key_col="SKU")
            save_db(st.session_state.db, st.session_state.db_path)
            st.toast("Visible rows upserted to DB file", icon="📀")
    with c3:
        if st.button("🧹 Remove Empty SKU Rows"):
            cleaned = st.session_state.db.copy()
            cleaned["SKU"] = cleaned["SKU"].astype(str)
            cleaned = cleaned[cleaned["SKU"].str.strip() != ""]
            st.session_state.db = cleaned.reset_index(drop=True)
            st.toast("Removed rows with empty SKU", icon="🗑️")
            st.rerun()

    st.caption("Tip: Use column headers to sort; use the filters above to select the rows you want to upsert.")

with tab2:
    st.markdown(
        """
**About**

- Company: PG Bison • OMEC Tools  
- Module: Inventory Manager (Quick Edit removed)  
- Version: **""" + __version__ + """**  
- Notes: Edit directly in the grid. Use **Save Edits** to keep changes in memory, and **Upsert Visible Rows** to persist to the DB file.
        """
    )
