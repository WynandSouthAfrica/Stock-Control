# app.py
# OMEC | Stock Controller Toolkit — Inventory + Maintenance (Category Manager)
# Run:  streamlit run app.py

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import streamlit as st

# ----------------------------- App Config ----------------------------------- #
APP_TITLE = "OMEC • Stock Controller"
st.set_page_config(page_title=APP_TITLE, page_icon="🧰", layout="wide")

# Prefer local assets; fall back to containers (works both locally & on Streamlit Cloud)
LOGO_OMEC_CANDIDATES = ["assets/logo_OMEC.png", "/mnt/data/Logo R0.1.png"]
LOGO_PGB_CANDIDATES = ["assets/logo_PG_Bison.png", "/mnt/data/PG Bison.jpg"]

DATA_PATH = Path("inventory_data.json")

# ----------------------------- Session init --------------------------------- #
def init_state():
    st.session_state.setdefault("items", None)
    st.session_state.setdefault("activity_log", [])  # list[dict]
    st.session_state.setdefault("df_history", [])    # stack of JSON strings (for Undo)

init_state()

# ----------------------------- Utilities ------------------------------------ #
def first_existing(paths: List[str]) -> str | None:
    for p in paths:
        try:
            if Path(p).exists():
                return p
        except Exception:
            pass
    return None

LOGO_OMEC = first_existing(LOGO_OMEC_CANDIDATES)
LOGO_PGB = first_existing(LOGO_PGB_CANDIDATES)

def safe_image(path: str | None, *, use_column_width: bool = False):
    """Show an image only if the path exists (prevents MediaFileStorageError)."""
    if not path:
        return False
    try:
        st.image(path, use_column_width=use_column_width)
        return True
    except Exception:
        return False

def page_header(title: str, subtitle: str = ""):
    cols = st.columns([1, 6, 1.2])
    with cols[0]:
        safe_image(LOGO_OMEC, use_column_width=True)
    with cols[1]:
        st.title(title)
        if subtitle:
            st.caption(subtitle)
    with cols[2]:
        safe_image(LOGO_PGB, use_column_width=True)

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def push_history(df: pd.DataFrame):
    """Keep a small history stack for Undo."""
    try:
        st.session_state.df_history.append(df.to_json(orient="records", force_ascii=False))
        if len(st.session_state.df_history) > 10:
            st.session_state.df_history.pop(0)
    except Exception:
        pass

def load_data() -> pd.DataFrame:
    if isinstance(st.session_state.get("items"), pd.DataFrame):
        return st.session_state.items.copy()

    if DATA_PATH.exists():
        try:
            data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        # Demo seed
        df = pd.DataFrame(
            [
                {"id": str(uuid.uuid4()), "name": "Cutting Discs 115mm", "category": "Consumables", "qty": 12, "location": "Stores A"},
                {"id": str(uuid.uuid4()), "name": "Welding Rods E6013 3.2", "category": "Consumables ", "qty": 2, "location": "Stores A"},
                {"id": str(uuid.uuid4()), "name": "Flat Bar 50x6x6m", "category": "Steel - Flat Bar", "qty": 8, "location": "Yard"},
                {"id": str(uuid.uuid4()), "name": "M12 Nylock Nut", "category": "Fasteners", "qty": 150, "location": "Bin F3"},
                {"id": str(uuid.uuid4()), "name": "Laser Test Plate", "category": "laser shop", "qty": 1, "location": "Fab Shop"},
            ]
        )

    st.session_state.items = df
    return df.copy()

def save_data(df: pd.DataFrame, log_action: str | None = None, log_details: str | None = None):
    if isinstance(st.session_state.get("items"), pd.DataFrame):
        push_history(st.session_state.items.copy())
    st.session_state.items = df.copy()
    DATA_PATH.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    if log_action:
        st.session_state.activity_log.append(
            {"time": timestamp(), "action": log_action, "details": log_details or ""}
        )

def get_categories(df: pd.DataFrame) -> List[str]:
    if "category" not in df.columns:
        return []
    cats = sorted({(c or "").strip() for c in df["category"].astype(str).tolist() if str(c).strip() != ""})
    return cats

def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def smart_title_case(s: str) -> str:
    """
    Title-case with light acronym protection:
    - Leave tokens that are fully UPPER and length <= 4 (e.g., M12, E6013) as-is.
    - Otherwise title-case each token (respect hyphens).
    """
    tokens = collapse_spaces(s).split(" ")
    fixed = []
    for t in tokens:
        if t.isupper() and len(t) <= 4:
            fixed.append(t)
        else:
            parts = re.split(r"(-)", t)
            parts = [p if p == "-" else (p[:1].upper() + p[1:].lower()) for p in parts]
            fixed.append("".join(parts))
    return " ".join(fixed)

def normalize_category_label(s: str) -> str:
    return smart_title_case(collapse_spaces(s))

def rename_or_merge_category(df: pd.DataFrame, source_cat: str, target_cat: str) -> Tuple[pd.DataFrame, int]:
    mask = df["category"].astype(str).str.strip() == source_cat.strip()
    affected = int(mask.sum())
    if affected > 0:
        df = df.copy()
        df.loc[mask, "category"] = target_cat
    return df, affected

def normalize_all_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping = {}
    if "category" not in df.columns:
        return df, mapping

    cats_before = df["category"].astype(str).tolist()
    cats_after = [normalize_category_label(c) for c in cats_before]
    df = df.copy()
    df["category"] = cats_after

    for before, after in zip(cats_before, cats_after):
        if collapse_spaces(before) != collapse_spaces(after):
            mapping[before] = after
    return df, mapping

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = ["id", "name", "category", "qty", "location"]
    for col in required:
        if col not in df.columns:
            if col == "id":
                df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
            elif col == "qty":
                df[col] = 0
            else:
                df[col] = ""
    return df[required]

# ----------------------------- Pages ---------------------------------------- #
def inventory_page():
    page_header("Inventory", "Add items first. Categories come from the items themselves.")
    df = ensure_schema(load_data())

    st.subheader("Quick Add Item")
    with st.form("add_item_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 3, 1.2, 2])
        with c1:
            name = st.text_input("Item name", placeholder="e.g., Flat Bar 50x6x6m")
        with c2:
            cat_suggestion = st.selectbox(
                "Pick an existing category (optional)",
                options=["—"] + get_categories(df),
                index=0,
                help="Optional: Select to auto-fill the Category field below.",
            )
            cat = st.text_input("Category", value=("" if cat_suggestion == "—" else cat_suggestion),
                                placeholder="e.g., Steel - Flat Bar")
        with c3:
            qty = st.number_input("Qty", min_value=0, step=1, value=1)
        with c4:
            loc = st.text_input("Location", placeholder="e.g., Stores A / Bin F3")
        submitted = st.form_submit_button("Add to Inventory", use_container_width=True)
        if submitted:
            if name.strip() and cat.strip():
                new_row = {
                    "id": str(uuid.uuid4()),
                    "name": collapse_spaces(name),
                    "category": collapse_spaces(cat),
                    "qty": int(qty),
                    "location": collapse_spaces(loc),
                }
                df2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(df2, log_action="Add Item", log_details=f"{new_row['name']} → {new_row['category']}")
                st.success(f"Added “{new_row['name']}” to {new_row['category']}.")
            else:
                st.warning("Please provide at least an Item name and a Category.")

    st.divider()
    st.subheader("Edit Inventory")
    st.caption("Tip: Edit directly and click **Save changes** below.")
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Item"),
            "category": st.column_config.TextColumn("Category"),
            "qty": st.column_config.NumberColumn("Qty", step=1, min_value=0),
            "location": st.column_config.TextColumn("Location"),
        },
        key="editor_inventory",
    )

    c1, c2, c3, c4 = st.columns([1.2, 1, 1.1, 2])
    with c1:
        if st.button("Save changes", type="primary", use_container_width=True):
            save_data(ensure_schema(edited), log_action="Save Inventory", log_details="Manual edit")
            st.success("Inventory saved.")
    with c2:
        if st.button("Reload", use_container_width=True):
            st.experimental_rerun()
    with c3:
        if st.button("Undo last change", use_container_width=True):
            if st.session_state.df_history:
                last_json = st.session_state.df_history.pop()
                df_prev = pd.DataFrame(json.loads(last_json))
                save_data(ensure_schema(df_prev), log_action="Undo", log_details="Reverted to previous state")
                st.success("Reverted to previous state.")
            else:
                st.info("Nothing to undo yet.")
    with c4:
        cats = get_categories(edited)
        st.metric("Distinct categories", len(cats))
        st.caption(", ".join(cats) if cats else "None yet")

def maintenance_page():
    page_header("Maintenance", "Maintenance usage • Category Manager / cleanup tools")

    st.info("Add items first in the **Inventory** page. This tool cleans category names and merges duplicates.")

    df = ensure_schema(load_data())
    cats = get_categories(df)

    st.divider()
    st.subheader("Category Manager")

    with st.expander("What is this & how to use it? (tap to open)", expanded=False):
        st.markdown(
            """
**Intention:** keep your category list clean and consistent – fix trailing spaces, odd casing,
and merge duplicate labels like _"laser shop"_, _"Laser Shop "_ and _"Laser   shop"_ into **Laser Shop**.

**How to use:**
1) **Rename/Merge**  
   Choose an **Existing** category on the left and type the **target** name on the right.  
   Click **Rename/Merge** → all items in the chosen category move into the target category.

2) **Normalize categories**  
   One-click cleanup: trims spaces, collapses multiple spaces to one, and Title-Cases labels
   (keeps short ALL-CAPS tokens like M12, E6013).  
   Use this after importing or bulk editing data.

_This never deletes items. Categories are always derived from the items themselves._
            """
        )

    c_left, c_right = st.columns([1, 1])
    with c_left:
        src = st.selectbox(
            "Existing category to rename/merge",
            options=["—"] + cats,
            index=0,
            help="Pick the category you want to rename or merge into another.",
        )
        run_merge = st.button("Rename/Merge", use_container_width=True)
    with c_right:
        target = st.text_input(
            "Replace with (target category, e.g., 'Laser Shop')",
            value="",
            placeholder="Laser Shop",
            help="The final category name. If it doesn't exist yet, it will be created by renaming items.",
        )
        run_normalize = st.button("Normalize categories (trim/collapse Title-Case)", use_container_width=True)

    if run_merge:
        if src == "—":
            st.warning("Please choose an **Existing** category.")
        elif not target.strip():
            st.warning("Please enter a **target** category.")
        else:
            target_clean = collapse_spaces(target)
            df2, affected = rename_or_merge_category(df.copy(), src, target_clean)
            save_data(df2, log_action="Rename/Merge", log_details=f"{src} → {target_clean} on {affected} item(s)")
            st.success(f"Renamed/Merged **{src} → {target_clean}** on **{affected}** item(s).")

    if run_normalize:
        df2, mapping = normalize_all_categories(df.copy())
        save_data(df2, log_action="Normalize Categories", log_details=f"{len(mapping)} change(s)")
        if mapping:
            map_rows = sorted({(k, v) for k, v in mapping.items()})
            map_df = pd.DataFrame(map_rows, columns=["Before", "After"])
            st.success(f"Normalized categories. {len(map_df)} change(s) detected.")
            st.dataframe(map_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nothing to normalize. Your categories already look clean.")

    st.divider()
    st.subheader("Category Snapshot")
    df_latest = ensure_schema(load_data())
    cats_now = get_categories(df_latest)
    st.caption("Current categories (after maintenance):")
    st.write(", ".join(cats_now) if cats_now else "—")
    st.dataframe(
        df_latest.sort_values(by=["category", "name"]).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Activity Log (last 25)", expanded=False):
        if st.session_state.activity_log:
            log_df = pd.DataFrame(st.session_state.activity_log)[-25:]
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No activity yet.")

# ----------------------------- Sidebar -------------------------------------- #
with st.sidebar:
    safe_image(LOGO_OMEC, use_column_width=True)
    st.markdown("### OMEC • Stock Controller")
    page = st.radio(
        "Navigate",
        options=["Inventory", "Maintenance"],
        index=1,
        help="Inventory = add/edit items • Maintenance = clean/merge categories",
    )
    st.divider()

    # Export always visible
    df_export = ensure_schema(load_data())
    st.download_button(
        "Download inventory_data.json",
        data=df_export.to_json(orient="records", force_ascii=False, indent=2),
        file_name="inventory_data.json",
        mime="application/json",
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Undo last", use_container_width=True):
            if st.session_state.df_history:
                last_json = st.session_state.df_history.pop()
                df_prev = pd.DataFrame(json.loads(last_json))
                save_data(ensure_schema(df_prev), log_action="Undo", log_details="Reverted to previous state")
                st.success("Reverted to previous state.")
            else:
                st.info("Nothing to undo yet.")
    with c2:
        if st.button("Clear demo & start fresh", type="secondary", use_container_width=True):
            empty = ensure_schema(pd.DataFrame([]))
            save_data(empty, log_action="Clear Inventory", log_details="Start fresh")
            st.success("Inventory cleared.")

# ----------------------------- Router --------------------------------------- #
if page == "Inventory":
    inventory_page()
else:
    maintenance_page()
