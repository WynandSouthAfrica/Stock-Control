# app.py
# OMEC | Stock Controller Toolkit — Inventory + Maintenance (Category Manager)
# Persistent storage on a work server + snapshots/revisions.
# Run:  streamlit run app.py

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ----------------------------- App Config ----------------------------------- #
APP_TITLE = "OMEC • Stock Controller"
st.set_page_config(page_title=APP_TITLE, page_icon="🧰", layout="wide")

# Prefer local assets; fall back to container-mounted images
LOGO_OMEC_CANDIDATES = ["assets/logo_OMEC.png", "/mnt/data/Logo R0.1.png"]
LOGO_PGB_CANDIDATES = ["assets/logo_PG_Bison.png", "/mnt/data/PG Bison.jpg"]

# App-local config (remembers the chosen data folder across runs)
CONFIG_PATH = Path("stock_config.json")

# ----------------------------- Session init --------------------------------- #
def init_state():
    st.session_state.setdefault("items", None)         # current working DF
    st.session_state.setdefault("activity_log", [])    # list of {time, action, details}
    st.session_state.setdefault("df_history", [])      # undo stack (json strings)
    st.session_state.setdefault("data_dir", None)      # base folder for data/snapshots

init_state()

# ----------------------------- Utilities ------------------------------------ #
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ts_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def first_existing(paths: List[str]) -> str | None:
    for p in paths:
        try:
            if Path(p).exists():
                return p
        except Exception:
            pass
    return None

LOGO_OMEC = first_existing(LOGO_OMEC_CANDIDATES)
LOGO_PGB  = first_existing(LOGO_PGB_CANDIDATES)

def safe_image(path: str | None, *, use_column_width: bool = False):
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

def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def smart_title_case(s: str) -> str:
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

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = ["id", "name", "category", "qty", "location"]
    df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    for col in required:
        if col not in df.columns:
            if col == "id":
                df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
            elif col == "qty":
                df[col] = 0
            else:
                df[col] = ""
    return df[required]

def get_categories(df: pd.DataFrame) -> List[str]:
    if "category" not in df.columns:
        return []
    return sorted({(c or "").strip() for c in df["category"].astype(str) if str(c).strip()})

# ----------------------------- Storage layer -------------------------------- #
def load_config():
    if CONFIG_PATH.exists():
        try:
            obj = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            st.session_state.data_dir = obj.get("data_dir") or st.session_state.data_dir
        except Exception:
            pass

def save_config():
    cfg = {"data_dir": st.session_state.get("data_dir")}
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def set_data_dir(path_str: str) -> Tuple[Path, Path]:
    """
    Configure the base folder to store live data and snapshots.
    Returns (data_file, snapshots_dir)
    """
    base = Path(path_str).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    data_file = base / "inventory_data.json"
    snapshots_dir = base / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.data_dir = str(base)
    save_config()
    return data_file, snapshots_dir

def get_paths() -> Tuple[Path, Path]:
    """
    Determine the current data file and snapshots folder.
    Defaults to ./data/ if nothing configured yet.
    """
    if st.session_state.get("data_dir"):
        base = Path(st.session_state["data_dir"])
    else:
        load_config()
        base = Path(st.session_state["data_dir"]) if st.session_state.get("data_dir") else Path("data")
        base.mkdir(parents=True, exist_ok=True)
        st.session_state["data_dir"] = str(base)
        save_config()
    data_file = base / "inventory_data.json"
    snapshots_dir = base / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return data_file, snapshots_dir

def can_write(dir_path: Path) -> bool:
    try:
        test = dir_path / f".write_test_{ts_filename()}.tmp"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def acquire_lock(lock_path: Path) -> bool:
    try:
        # Create exclusively; if exists, someone is writing
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return True  # best effort

def release_lock(lock_path: Path):
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass

def push_history(df: pd.DataFrame):
    try:
        st.session_state.df_history.append(df.to_json(orient="records", force_ascii=False))
        if len(st.session_state.df_history) > 10:
            st.session_state.df_history.pop(0)
    except Exception:
        pass

def list_snapshots(snapshots_dir: Path) -> List[Path]:
    files = sorted(snapshots_dir.glob("inventory_*.json"), reverse=True)
    return files

def save_snapshot(df: pd.DataFrame, snapshots_dir: Path, note: str = "", keep_last: int = 100) -> Path:
    safe_note = re.sub(r"[^A-Za-z0-9_-]+", "-", note.strip()) if note else ""
    name = f"inventory_{ts_filename()}" + (f"_{safe_note}" if safe_note else "") + ".json"
    out = snapshots_dir / name
    out.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    # trim old
    files = list_snapshots(snapshots_dir)
    for f in files[keep_last:]:
        f.unlink(missing_ok=True)
    return out

def load_snapshot_to_df(path: Path) -> pd.DataFrame:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ensure_schema(pd.DataFrame(data))
    except Exception:
        return ensure_schema(pd.DataFrame())

# ----------------------------- Data API ------------------------------------- #
def load_data() -> pd.DataFrame:
    data_file, _ = get_paths()
    # Use in-memory DF if present
    if isinstance(st.session_state.get("items"), pd.DataFrame):
        return ensure_schema(st.session_state.items.copy())

    if data_file.exists():
        try:
            raw = data_file.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            df = ensure_schema(pd.DataFrame(data))
        except Exception:
            df = ensure_schema(pd.DataFrame())
    else:
        # Seed with a tiny demo if brand-new
        df = ensure_schema(pd.DataFrame([
            {"id": str(uuid.uuid4()), "name": "Cutting Discs 115mm", "category": "Consumables", "qty": 12, "location": "Stores A"},
            {"id": str(uuid.uuid4()), "name": "Welding Rods E6013 3.2", "category": "Consumables ", "qty": 2, "location": "Stores A"},
            {"id": str(uuid.uuid4()), "name": "Flat Bar 50x6x6m", "category": "Steel - Flat Bar", "qty": 8, "location": "Yard"},
            {"id": str(uuid.uuid4()), "name": "M12 Nylock Nut", "category": "Fasteners", "qty": 150, "location": "Bin F3"},
            {"id": str(uuid.uuid4()), "name": "Laser Test Plate", "category": "laser shop", "qty": 1, "location": "Fab Shop"},
        ]))

    st.session_state.items = df.copy()
    return df.copy()

def save_data(df: pd.DataFrame, *, snapshot_note: str = "", log_action: str | None = None, log_details: str | None = None):
    df = ensure_schema(df.copy())
    data_file, snapshots_dir = get_paths()

    # best-effort lock
    lock_path = data_file.with_suffix(".lock")
    locked = acquire_lock(lock_path)
    try:
        # Save undo
        if isinstance(st.session_state.get("items"), pd.DataFrame):
            push_history(st.session_state.items.copy())

        # Save live data
        st.session_state.items = df.copy()
        data_file.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

        # Also write a snapshot/revision on every save
        save_snapshot(df, snapshots_dir, note=snapshot_note)
    finally:
        if locked:
            release_lock(lock_path)

    if log_action:
        st.session_state.activity_log.append({"time": ts(), "action": log_action, "details": log_details or ""})

def rename_or_merge_category(df: pd.DataFrame, source_cat: str, target_cat: str) -> Tuple[pd.DataFrame, int]:
    mask = df["category"].astype(str).str.strip() == source_cat.strip()
    affected = int(mask.sum())
    if affected > 0:
        df = df.copy()
        df.loc[mask, "category"] = target_cat
    return df, affected

def normalize_all_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
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

# ----------------------------- Pages ---------------------------------------- #
def inventory_page():
    page_header("Inventory", "Live data stored on your chosen folder (server or local).")
    df = load_data()

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
                save_data(df2, snapshot_note=f"Add {new_row['name']}", log_action="Add Item",
                          log_details=f"{new_row['name']} → {new_row['category']}")
                st.success(f"Added “{new_row['name']}” to {new_row['category']}.")
            else:
                st.warning("Please provide at least an Item name and a Category.")

    st.divider()
    st.subheader("Edit Inventory")
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
        note = st.text_input("Revision note (optional)", placeholder="e.g., bulk qty update")
        if st.button("Save changes", type="primary", use_container_width=True):
            save_data(edited, snapshot_note=note, log_action="Save Inventory", log_details="Manual edit")
            st.success("Inventory saved + snapshot created.")
    with c2:
        if st.button("Reload", use_container_width=True):
            st.experimental_rerun()
    with c3:
        if st.button("Undo last change", use_container_width=True):
            if st.session_state.df_history:
                last_json = st.session_state.df_history.pop()
                df_prev = ensure_schema(pd.DataFrame(json.loads(last_json)))
                save_data(df_prev, snapshot_note="Undo", log_action="Undo", log_details="Reverted to previous state")
                st.success("Reverted to previous state.")
            else:
                st.info("Nothing to undo yet.")
    with c4:
        cats = get_categories(edited)
        st.metric("Distinct categories", len(cats))
        st.caption(", ".join(cats) if cats else "None yet")

def maintenance_page():
    page_header("Maintenance", "Category Manager / cleanup tools")
    df = load_data()
    cats = get_categories(df)

    st.info("This never deletes items. It only updates their `category` field.")

    st.divider()
    st.subheader("Category Manager")

    c_left, c_right = st.columns([1, 1])
    with c_left:
        src = st.selectbox("Existing category to rename/merge", options=["—"] + cats, index=0)
        run_merge = st.button("Rename/Merge", use_container_width=True)
    with c_right:
        target = st.text_input("Replace with (target category, e.g., 'Laser Shop')", value="", placeholder="Laser Shop")
        run_normalize = st.button("Normalize categories (trim/collapse Title-Case)", use_container_width=True)

    if run_merge:
        if src == "—":
            st.warning("Choose an **Existing** category first.")
        elif not target.strip():
            st.warning("Enter a **target** category.")
        else:
            target_clean = collapse_spaces(target)
            df2, affected = rename_or_merge_category(df.copy(), src, target_clean)
            save_data(df2, snapshot_note=f"Merge {src}->{target_clean}",
                      log_action="Rename/Merge", log_details=f"{src} → {target_clean} on {affected} item(s)")
            st.success(f"Merged **{src} → {target_clean}** on **{affected}** item(s).")

    if run_normalize:
        df2, mapping = normalize_all_categories(df.copy())
        save_data(df2, snapshot_note="Normalize categories",
                  log_action="Normalize Categories", log_details=f"{len(mapping)} change(s)")
        if mapping:
            map_rows = sorted({(k, v) for k, v in mapping.items()})
            map_df = pd.DataFrame(map_rows, columns=["Before", "After"])
            st.success(f"Normalized categories. {len(map_df)} change(s) detected.")
            st.dataframe(map_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nothing to normalize.")

    st.divider()
    st.subheader("Category Snapshot")
    df_latest = load_data()
    cats_now = get_categories(df_latest)
    st.caption("Current categories:")
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

    # Storage configuration (server/local)
    st.markdown("**Storage**")
    data_file, snapshots_dir = get_paths()
    current_dir = Path(st.session_state["data_dir"])
    st.caption(f"Current folder: `{current_dir}`")
    path_input = st.text_input("Change data folder (server/UNC or local path)",
                               value=str(current_dir),
                               help="Examples: `\\\\\\SERVER\\Share\\OMEC_Stock_Control` (Windows) or `/mnt/share/OMEC_Stock_Control` (Linux)")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Use this folder", use_container_width=True):
            df_live = load_data()  # keep in-memory copy
            data_file_new, snapshots_dir_new = set_data_dir(path_input)
            # migrate existing data file & snapshots if moving to a new empty location
            if not data_file_new.exists() and isinstance(df_live, pd.DataFrame):
                data_file_new.write_text(df_live.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
                save_snapshot(df_live, snapshots_dir_new, note="Auto-migrate")
            st.success(f"Folder set to: {data_file_new.parent}")
    with colB:
        st.write("Writable:" if can_write(current_dir) else "Not writable")

    st.divider()
    page = st.radio("Navigate", options=["Inventory", "Maintenance"], index=0)

    st.divider()
    # Snapshots control
    st.markdown("**Snapshots / Revisions**")
    snap_note = st.text_input("Snapshot note", placeholder="e.g., after stock take")
    if st.button("Create snapshot now", use_container_width=True):
        df_now = load_data()
        p = save_snapshot(df_now, snapshots_dir, note=snap_note)
        st.success(f"Snapshot saved: {p.name}")

    snaps = list_snapshots(snapshots_dir)
    if snaps:
        names = [s.name for s in snaps]
        choice = st.selectbox("Restore snapshot", options=["— select —"] + names, index=0)
        if st.button("Restore selected", use_container_width=True, disabled=(choice == "— select —")):
            if choice != "— select —":
                df_snap = load_snapshot_to_df(snapshots_dir / choice)
                save_data(df_snap, snapshot_note=f"Restore {choice}", log_action="Restore Snapshot", log_details=choice)
                st.success(f"Restored: {choice}")
    else:
        st.caption("No snapshots yet.")

    st.divider()
    # Export & housekeeping
    df_export = load_data()
    st.download_button(
        "Download inventory_data.json",
        data=df_export.to_json(orient="records", force_ascii=False, indent=2),
        file_name="inventory_data.json",
        mime="application/json",
        use_container_width=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Undo last", use_container_width=True):
            if st.session_state.df_history:
                last_json = st.session_state.df_history.pop()
                df_prev = ensure_schema(pd.DataFrame(json.loads(last_json)))
                save_data(df_prev, snapshot_note="Undo", log_action="Undo", log_details="Reverted")
                st.success("Reverted to previous state.")
            else:
                st.info("Nothing to undo yet.")
    with col2:
        if st.button("Clear all data (keep folder)", type="secondary", use_container_width=True):
            empty = ensure_schema(pd.DataFrame([]))
            save_data(empty, snapshot_note="Clear", log_action="Clear Inventory", log_details="Start fresh")
            st.success("Inventory cleared.")

# ----------------------------- Router --------------------------------------- #
if page == "Inventory":
    inventory_page()
else:
    maintenance_page()
