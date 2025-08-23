# app.py — OMEC Stock Take (single-file)
# Restore fix: accept inventory/items*.csv and transactions/trans*.csv in snapshot ZIP
# (The rest of the app is the same feature set we built.)

import os, json, math, re, io, zipfile
import streamlit as st
import pandas as pd

from db import (
    init_db, get_items, add_or_update_item, delete_item,
    get_transactions, add_transaction,
    get_versions, save_version_record,
    upsert_setting, get_setting
)
from utils import export_snapshot, timestamp

# PDF engine
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

st.set_page_config(page_title="OMEC Stock Take", page_icon="🗃️", layout="wide")

# ---------- Config ----------
ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": "", "revision_tag": "Rev0.1"}

init_db()

# ---------- Helpers ----------
def _norm_path(p: str) -> str:
    if not isinstance(p, str) or not p.strip():
        return ""
    p = os.path.normpath(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def safe_show_logo(path: str):
    try:
        apath = _norm_path(path)
        if apath and os.path.exists(apath):
            st.sidebar.image(apath, use_container_width=True)
    except Exception:
        pass

def hex_to_rgb(h: str):
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (14, 165, 233)

def lighten(rgb, factor=0.85):
    r, g, b = rgb
    return (int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))

def rev_bump(tag: str) -> str:
    m = re.match(r"^\s*Rev(\d+)\.(\d+)\s*$", str(tag))
    if not m:
        return "Rev0.1"
    major, minor = int(m.group(1)), int(m.group(2))
    minor += 1
    return f"Rev{major}.{minor}"

def to_latin1(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.encode("latin-1", "replace").decode("latin-1")

# ---------- Settings ----------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
revision_tag = get_setting("revision_tag", CONFIG.get("revision_tag", "Rev0.1"))
prepared_by = get_setting("prepared_by", "")
brand_rgb = hex_to_rgb(brand_color)

# ---------- Sidebar ----------
st.sidebar.markdown(
    f"<h2 style='color:{brand_color}; margin-bottom:0'>{brand_name}</h2>",
    unsafe_allow_html=True
)
safe_show_logo(logo_path)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Inventory",
        "Transactions",
        "Versions & Snapshots",
        "Reports & Export (PDF)",
        "Maintenance",
        "Settings",
    ],
    index=0,
)

# ---------- PDF helpers ----------
if FPDF_AVAILABLE:
    class BrandedPDF(FPDF):
        def __init__(self, brand_name: str, brand_rgb: tuple, logo_path: str = "", revision_tag: str = "Rev0.1", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.brand_name = brand_name
            self.brand_rgb = brand_rgb
            self.revision_tag = revision_tag
            self.logo_path = _norm_path(logo_path) if logo_path else ""
            self.set_auto_page_break(auto=True, margin=12)

        def header(self):
            self.set_fill_color(*lighten(self.brand_rgb, 0.75))
            self.rect(x=0, y=0, w=self.w, h=14, style="F")
            x = 10
            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    self.image(self.logo_path, x=x, y=3, h=8)
                    x += 26
                except Exception:
                    pass
            self.set_xy(x, 4)
            self.set_text_color(25, 25, 25)
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 6, txt=to_latin1(self.brand_name), ln=0)
            self.set_xy(-80, 4)
            self.set_font("Helvetica", "", 9)
            self.cell(70, 6, txt=to_latin1(f"Revision: {self.revision_tag}"), ln=0, align="R")
            self.set_draw_color(*self.brand_rgb)
            self.set_line_width(0.4)
            self.line(8, 14, self.w - 8, 14)
            self.ln(9)

        def footer(self):
            self.set_y(-10)
            self.set_draw_color(*self.brand_rgb)
            self.set_line_width(0.2)
            self.line(8, self.get_y(), self.w - 8, self.get_y())
            self.set_y(-8)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, to_latin1(f"Page {self.page_no()}"), align="R")

def _compute_col_widths(pdf, columns, rows, page_w, font_size):
    pdf.set_font("Helvetica", "B", font_size)
    widths = []
    for c in columns:
        max_w = pdf.get_string_width(to_latin1(str(c))) + 6
        pdf.set_font("Helvetica", "", font_size)
        for r in rows[:200]:
            w = pdf.get_string_width(to_latin1(str(r.get(c, "")))) + 6
            if w > max_w:
                max_w = w
        max_w = max(18, min(max_w, 100))
        widths.append(max_w)
    total_w = sum(widths)
    if total_w > page_w:
        scale = page_w / total_w
        widths = [w * scale for w in widths]
    else:
        extra = page_w - total_w
        if extra > 0 and len(widths) > 0:
            bump = extra / len(widths)
            widths = [w + bump for w in widths]
    return widths

def _draw_header_row(pdf, columns, widths, font_size, brand_rgb):
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_fill_color(*lighten(brand_rgb, 0.85))
    pdf.set_text_color(20, 20, 20)
    row_h = 7
    y0 = pdf.get_y()
    for w, col in zip(widths, columns):
        x = pdf.get_x()
        pdf.multi_cell(w, row_h, to_latin1(str(col)), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_xy(x + w, y0)
    pdf.ln(row_h)
    return row_h

def _ensure_page_space(pdf, needed_h, columns, widths, font_size, brand_rgb):
    if pdf.get_y() + needed_h <= pdf.h - pdf.b_margin:
        return
    pdf.add_page()
    _draw_header_row(pdf, columns, widths, font_size, brand_rgb)

def _calc_row_height_exact(pdf, values, widths, row_h, wrap_idx_set):
    heights = []
    for idx, (w, v) in enumerate(zip(widths, values)):
        txt = "" if v is None else to_latin1(str(v))
        if wrap_idx_set and (idx in wrap_idx_set):
            try:
                lines = pdf.multi_cell(w, row_h, txt, new_x="RIGHT", new_y="TOP", split_only=True)
                n = max(1, len(lines))
            except Exception:
                n = 1
        else:
            n = 1
        heights.append(n * row_h)
    return max(heights) if heights else row_h

def _draw_row(pdf, values, widths, row_h, align_map=None, fill_rgb=None, bold=False, text_rgb=(15,15,15), wrap_idx_set=None, border="1"):
    if align_map is None:
        align_map = {}
    if wrap_idx_set is None:
        wrap_idx_set = set()
    max_h = _calc_row_height_exact(pdf, values, widths, row_h, wrap_idx_set)
    pdf.set_font("Helvetica", "B" if bold else "", 9)
    if fill_rgb:
        pdf.set_fill_color(*fill_rgb)
    pdf.set_text_color(*text_rgb)
    y_start = pdf.get_y()
    x_left = pdf.get_x()
    for idx, (w, v) in enumerate(zip(widths, values)):
        x = pdf.get_x()
        align = align_map.get(idx, "L")
        txt = "" if v is None else to_latin1(str(v))
        if fill_rgb:
            pdf.set_fill_color(*fill_rgb)
        pdf.multi_cell(w, row_h, txt, border=border, align=align, new_x="RIGHT", new_y="TOP", fill=bool(fill_rgb))
        pdf.set_xy(x + w, y_start)
    pdf.set_xy(x_left, y_start + max_h)

# ---------- PDF Inventory ----------
def _inventory_pdf_bytes_grouped(
    df: pd.DataFrame,
    brand_name, brand_rgb, logo_path, revision_tag,
    prepared_by: str, only_low: bool, sort_by: str,
    categories=None, only_available: bool=False
) -> bytes:
    df = df.copy()
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
    df["min_qty"] = pd.to_numeric(df.get("min_qty"), errors="coerce").fillna(0.0)
    df["unit_cost"] = pd.to_numeric(df.get("unit_cost"), errors="coerce").fillna(0.0)
    df["value"] = (df["quantity"] * df["unit_cost"]).round(2)

    if categories and "category" in df.columns:
        df = df[df["category"].isin(categories)]
    if only_available:
        df = df[df["quantity"] > 0]
    if only_low:
        df = df[df["quantity"] < df["min_qty"]]

    if sort_by in {"sku", "name", "category"} and sort_by in df.columns:
        df = df.sort_values(by=[sort_by, "sku", "name"], kind="stable")

    key_cols = [
        "sku", "name", "category", "location", "unit",
        "quantity", "min_qty", "unit_cost", "value", "notes", "updated_at"
    ]
    present = [c for c in key_cols if c in df.columns]
    if not present:
        present = list(df.columns)
    df = df[present]

    col_names = {
        "sku": "SKU", "name": "Name", "category": "Category", "location": "Location",
        "unit": "Unit", "quantity": "Qty", "min_qty": "Min",
        "unit_cost": "Unit Cost (R)", "value": "Value (R)",
        "notes": "Notes", "updated_at": "Updated"
    }
    display_cols = [col_names.get(c, c) for c in present]
    def fmt_num(x, places=2): return f"{x:,.{places}f}"

    rows_for_width = []
    for _, r in df.iterrows():
        row = {}
        for c in present:
            v = r[c]
            if c in ("quantity", "min_qty"):
                v = fmt_num(float(v), 2)
            elif c in ("unit_cost", "value"):
                v = fmt_num(float(v), 2)
            row[col_names.get(c, c)] = "" if v is None else str(v)
        rows_for_width.append(row)

    pdf = BrandedPDF(
        brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path, revision_tag=revision_tag,
        orientation="L", unit="mm", format="A3"
    )
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, to_latin1("Inventory Report"), ln=1)
    pdf.set_font("Helvetica", "", 10)

    meta = [f"Rows: {len(df)}"]
    if categories:
        meta.append("Categories: " + ", ".join([to_latin1(c) for c in categories]))
    if only_available:
        meta.append("Only available (>0 qty)")
    if only_low:
        meta.append("Low-stock only")
    if prepared_by:
        meta.append(f"Prepared by: {prepared_by}")
    pdf.cell(0, 6, to_latin1(" | ".join(meta)), ln=1)
    pdf.ln(2)

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    widths = _compute_col_widths(pdf, display_cols, rows_for_width, page_w, font_size=9)
    header_h = _draw_header_row(pdf, display_cols, widths, 9, brand_rgb)
    align_map = {present.index(c): "R" for c in present if c in ("quantity", "min_qty", "unit_cost", "value")}
    wrap_idx_set = set()
    if "notes" in present:
        wrap_idx_set.add(display_cols.index("Notes"))

    if "category" in df.columns:
        df_sorted = df.sort_values(by=["category", "sku", "name"], kind="stable")
        categories_iter = df_sorted["category"].fillna("(Unspecified)").unique().tolist()
    else:
        df_sorted = df.copy()
        df_sorted["category"] = "(All)"
        categories_iter = ["(All)"]

    light_brand = lighten(brand_rgb, 0.92)
    cat_bar = lighten(brand_rgb, 0.80)
    low_stock_fill = (255, 235, 235)

    grand_qty = 0.0
    grand_value = 0.0

    for cat in categories_iter:
        block = df_sorted[df_sorted["category"].fillna("(Unspecified)") == cat]

        span_h = 7
        _ensure_page_space(pdf, span_h + header_h, display_cols, widths, 9, brand_rgb)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(*cat_bar)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(sum(widths), span_h, to_latin1(f"Category: {cat}"), border=1, ln=1, fill=True)

        cat_qty = float(block["quantity"].sum())
        cat_value = float((block["quantity"] * block["unit_cost"]).sum())
        grand_qty += cat_qty
        grand_value += cat_value

        for _, r in block.iterrows():
            vals = []
            for c in present:
                v = r[c]
                if c in ("quantity", "min_qty"):
                    v = fmt_num(float(v), 2)
                elif c in ("unit_cost", "value"):
                    v = fmt_num(float(v), 2)
                vals.append(v if v is not None else "")
            fill = low_stock_fill if float(r["quantity"]) < float(r["min_qty"]) else None
            row_h = 7
            _ensure_page_space(pdf, row_h + 2, display_cols, widths, 9, brand_rgb)
            _draw_row(pdf, vals, widths, row_h, align_map=align_map, fill_rgb=fill, wrap_idx_set=wrap_idx_set)

        sub_vals = []
        for c in present:
            if c == "sku":
                sub_vals.append("Subtotal")
            elif c == "quantity":
                sub_vals.append(f"{cat_qty:,.2f}")
            elif c == "value":
                sub_vals.append(f"{cat_value:,.2f}")
            else:
                sub_vals.append("")
        _ensure_page_space(pdf, 7, display_cols, widths, 9, brand_rgb)
        _draw_row(pdf, sub_vals, widths, 7, align_map=align_map, fill_rgb=light_brand, bold=True, wrap_idx_set=wrap_idx_set)

    total_vals = []
    for c in present:
        if c == "sku":
            total_vals.append("TOTAL")
        elif c == "quantity":
            total_vals.append(f"{grand_qty:,.2f}")
        elif c == "value":
            total_vals.append(f"{grand_value:,.2f}")
        else:
            total_vals.append("")
    _ensure_page_space(pdf, 8, display_cols, widths, 9, brand_rgb)
    _draw_row(pdf, total_vals, widths, 8, align_map=align_map, fill_rgb=lighten(brand_rgb, 0.88), bold=True, wrap_idx_set=wrap_idx_set)

    data = pdf.output(dest="S")
    return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")

# ---------- Views ----------
def view_dashboard():
    st.title("🏠 Dashboard")
    st.caption("Quick overview of your stock status.")
    items = get_items()
    df = pd.DataFrame(items)

    col1, col2, col3, col4 = st.columns(4)
    total_items = len(items)
    total_qty = sum((i.get("quantity") or 0) for i in items)
    low_stock = sum(1 for i in items if (i.get("min_qty") or 0) > (i.get("quantity") or 0))
    total_value = sum((i.get("quantity") or 0) * (i.get("unit_cost") or 0) for i in items)

    col1.metric("Distinct SKUs", total_items)
    col2.metric("Total Quantity", f"{total_qty:.2f}")
    col3.metric("Low-Stock Items", low_stock)
    col4.metric("Stock Value", f"R {total_value:,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Low-stock items")
        if df.empty:
            st.info("No items yet.")
        else:
            low_df = df[df["quantity"] < df["min_qty"]].copy()
            if low_df.empty:
                st.success("Nothing low on stock 🎉")
            else:
                st.dataframe(low_df[["sku", "name", "category", "location", "quantity", "min_qty"]], use_container_width=True, height=260)

    with c2:
        st.subheader("Category totals")
        if df.empty:
            st.info("No data.")
        else:
            cg = df.groupby(df["category"].fillna("(Unspecified)")).agg(
                qty=("quantity", "sum"),
                value=("unit_cost", lambda s: float((df.loc[s.index, "quantity"] * df.loc[s.index, "unit_cost"]).sum()))
            )
            cg = cg.sort_index()
            st.dataframe(cg, use_container_width=True, height=260)

def view_inventory():
    st.title("📦 Inventory")
    st.caption("Add, edit, or delete items.")

    items = get_items()
    df = pd.DataFrame(items)
    filt = st.text_input("Filter (SKU/Name/Category/Location contains…)")
    fdf = df.copy()
    if filt:
        f = filt.lower()
        mask = (
            fdf["sku"].astype(str).str.lower().str.contains(f) |
            fdf["name"].astype(str).str.lower().str.contains(f) |
            fdf["category"].astype(str).str.lower().str.contains(f) |
            fdf["location"].astype(str).str.lower().str.contains(f)
        )
        fdf = fdf[mask]

    with st.form("add_item"):
        cols = st.columns(4)
        sku = cols[0].text_input("SKU *")
        name = cols[1].text_input("Name *")
        category = cols[2].text_input("Category")
        location = cols[3].text_input("Location")

        cols2 = st.columns(4)
        unit = cols2[0].text_input("Unit (e.g., pcs, m, kg)")
        quantity = cols2[1].number_input("Quantity", value=0.0, step=1.0, format="%.3f")
        min_qty = cols2[2].number_input("Min Qty (alert level)", value=0.0, step=1.0, format="%.3f")
        unit_cost = cols2[3].number_input("Unit Cost (R)", value=0.0, step=1.0, format="%.2f")

        notes = st.text_area("Notes")

        submitted = st.form_submit_button("Save Item")
        if submitted:
            if sku and name:
                add_or_update_item({
                    "sku": sku.strip(),
                    "name": name.strip(),
                    "category": category.strip() if category else None,
                    "location": location.strip() if location else None,
                    "unit": unit.strip() if unit else None,
                    "quantity": float(quantity),
                    "min_qty": float(min_qty),
                    "unit_cost": float(unit_cost),
                    "notes": notes.strip() if notes else None,
                    "image_path": None,
                })
                st.success(f"Saved item '{sku}'")
            else:
                st.error("SKU and Name are required.")

    st.subheader("Inventory List (editable)")
    if not fdf.empty:
        show_cols = ["sku","name","category","location","unit","quantity","min_qty","unit_cost","notes","updated_at"]
        show_cols = [c for c in show_cols if c in fdf.columns]
        edited = st.data_editor(
            fdf[show_cols],
            use_container_width=True,
            num_rows="dynamic",
            key="inv_editor",
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.3f"),
                "min_qty": st.column_config.NumberColumn(format="%.3f"),
                "unit_cost": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        if st.button("Save Edits (Upsert visible rows)"):
            for _, r in edited.iterrows():
                add_or_update_item({
                    "sku": r.get("sku"),
                    "name": r.get("name"),
                    "category": r.get("category"),
                    "location": r.get("location"),
                    "unit": r.get("unit"),
                    "quantity": float(r.get("quantity") or 0),
                    "min_qty": float(r.get("min_qty") or 0),
                    "unit_cost": float(r.get("unit_cost") or 0),
                    "notes": r.get("notes"),
                    "image_path": None,
                })
            st.success("Edits saved.")
    else:
        st.info("No items yet. Add your first item above.")

    st.divider()
    colA, colB = st.columns(2)
    to_delete = colA.text_input("Delete by SKU")
    if colB.button("Delete Item"):
        if to_delete:
            delete_item(to_delete.strip())
            st.success(f"Deleted '{to_delete}'")
        else:
            st.error("Enter a SKU to delete.")

def view_transactions():
    st.title("🔁 Transactions")
    st.caption("Record stock movement in/out and maintain an audit trail.")

    items = get_items()
    sku_list = [i["sku"] for i in items]

    with st.form("tx_form"):
        cols = st.columns(5)
        sku = cols[0].selectbox("SKU", options=sku_list)
        qty_change = cols[1].number_input("Qty Change (+ in / - out)", value=0.0, step=1.0, format="%.3f")
        reason = cols[2].selectbox(
            "Reason",
            options=["receipt", "issue", "adjustment", "return", "count_correction", "maintenance", "other"],
        )
        project = cols[3].text_input("Project / Job")
        reference = cols[4].text_input("Reference (PO / DO / WO)")

        cols2 = st.columns(2)
        user = cols2[0].text_input("User")
        notes = cols2[1].text_input("Notes")

        submit = st.form_submit_button("Add Transaction")
        if submit:
            if sku and qty_change != 0:
                add_transaction(sku, qty_change, reason, project, reference, user, notes)
                st.success("Transaction recorded.")
            else:
                st.error("SKU and non-zero quantity are required.")

    st.subheader("Transaction Log (filtered)")
    tx = get_transactions(limit=100_000)
    df_tx = pd.DataFrame(tx)
    if df_tx.empty:
        st.info("No transactions yet.")
        return

    c1, c2, c3 = st.columns(3)
    sku_filter = c1.multiselect("SKU filter", sorted(df_tx["sku"].unique().tolist()))
    reason_filter = c2.multiselect("Reason filter", sorted(df_tx["reason"].unique().tolist()))
    search = c3.text_input("Text search")

    f = df_tx.copy()
    if sku_filter:
        f = f[f["sku"].isin(sku_filter)]
    if reason_filter:
        f = f[f["reason"].isin(reason_filter)]
    if search:
        s = search.lower()
        f = f[
            f["sku"].astype(str).str.lower().str.contains(s) |
            f["project"].astype(str).str.lower().str.contains(s) |
            f["reference"].astype(str).str.lower().str.contains(s) |
            f["user"].astype(str).str.lower().str.contains(s) |
            f["notes"].astype(str).str.lower().str.contains(s)
        ]
    st.dataframe(f, use_container_width=True)

def view_versions():
    st.title("🕒 Versions & Snapshots")
    st.caption("Create timestamped ZIP archives of your data for traceability. You can also restore from a snapshot.")

    # --- Create snapshot ---
    tag = st.text_input("Version tag (e.g., V0.1, 'after_stock_count')")
    note = st.text_area("Note")

    if st.button("Create Snapshot ZIP"):
        items = get_items()
        tx = get_transactions(limit=1_000_000)
        zip_path = export_snapshot(items, tx, tag=tag, note=note)
        save_version_record(tag or "", note or "", zip_path)
        st.success(f"Snapshot created: {zip_path}")
        with open(zip_path, "rb") as f:
            st.download_button("Download ZIP", data=f.read(), file_name=os.path.basename(zip_path))

    # --- Restore snapshot ---
    st.divider()
    st.subheader("Restore from Snapshot ZIP")
    st.caption("Use this to rebuild your DB from a previously saved ZIP (e.g., after a rebuild or when switching sites).")
    up = st.file_uploader("Upload snapshot ZIP", type=["zip"])
    colr1, colr2, colr3 = st.columns(3)
    replace_items = colr1.checkbox("Replace items (clear & load)", value=True)
    append_tx = colr2.checkbox("Append transactions", value=False)
    skip_dup = colr3.checkbox("Skip duplicate transactions", value=True)

    if up and st.button("Restore now"):
        try:
            zf = zipfile.ZipFile(io.BytesIO(up.read()))
            all_names = zf.namelist()

            # Flexible finders
            def find_csv(candidates):
                for n in all_names:
                    ln = n.lower()
                    if ln.endswith(".csv") and any(k in ln for k in candidates):
                        return n
                return None

            inv_name = find_csv(["inventory", "items"])
            tx_name  = find_csv(["transactions", "transaction", "trans"])

            if not inv_name:
                st.error("inventory.csv (or items*.csv) not found in ZIP.\n\nFiles in ZIP:\n" + "\n".join(all_names))
                return

            # Load inventory CSV
            inv_df = pd.read_csv(io.BytesIO(zf.read(inv_name)))

            if replace_items:
                existing = get_items()
                for it in existing:
                    try:
                        delete_item(it["sku"])
                    except Exception:
                        pass

            added_items = 0
            for _, r in inv_df.iterrows():
                try:
                    add_or_update_item({
                        "sku": str(r.get("sku") or "").strip(),
                        "name": str(r.get("name") or "").strip(),
                        "category": (r.get("category") if pd.notna(r.get("category")) else None),
                        "location": (r.get("location") if pd.notna(r.get("location")) else None),
                        "unit": (r.get("unit") if pd.notna(r.get("unit")) else None),
                        "quantity": float(r.get("quantity") or 0),
                        "min_qty": float(r.get("min_qty") or 0),
                        "unit_cost": float(r.get("unit_cost") or 0),
                        "notes": (r.get("notes") if pd.notna(r.get("notes")) else None),
                        "image_path": None,
                    })
                    added_items += 1
                except Exception as e:
                    st.warning(f"Item restore skipped for SKU={r.get('sku')}: {e}")

            added_tx = 0
            if append_tx and tx_name:
                tx_df = pd.read_csv(io.BytesIO(zf.read(tx_name)))
                existing_tx = get_transactions(limit=1_000_000) if skip_dup else []
                existing_keys = set()
                if skip_dup:
                    for t in existing_tx:
                        key = (
                            str(t.get("sku") or ""),
                            float(t.get("qty_change") or 0.0),
                            str(t.get("reason") or ""),
                            str(t.get("project") or ""),
                            str(t.get("reference") or ""),
                            str(t.get("user") or ""),
                            str(t.get("notes") or ""),
                        )
                        existing_keys.add(key)

                for _, r in tx_df.iterrows():
                    key = (
                        str(r.get("sku") or ""),
                        float(r.get("qty_change") or 0.0),
                        str(r.get("reason") or ""),
                        str(r.get("project") or ""),
                        str(r.get("reference") or ""),
                        str(r.get("user") or ""),
                        str(r.get("notes") or ""),
                    )
                    if skip_dup and key in existing_keys:
                        continue
                    try:
                        add_transaction(
                            key[0], key[1], key[2], key[3], key[4], key[5], key[6]
                        )
                        added_tx += 1
                    except Exception as e:
                        st.warning(f"Transaction restore skipped for SKU={key[0]}: {e}")

            st.success(f"Restore complete. Items loaded: {added_items}" + (f" | Transactions added: {added_tx}" if append_tx and tx_name else ""))
            st.info("Tip: Save a new snapshot after restoring and updating.")
        except Exception as e:
            st.error(f"Restore failed: {e}")

    st.subheader("History")
    versions = get_versions()
    if versions:
        st.dataframe(pd.DataFrame(versions), use_container_width=True)
    else:
        st.info("No versions yet.")

def view_reports():
    st.title("🧾 Reports & Export (PDF)")
    st.caption("A3 landscape • grouped by category • subtotals • low-stock highlight • totals row • Notes column")

    if not FPDF_AVAILABLE:
        st.error("PDF engine not available. Add `fpdf2==2.7.9` to requirements.txt.")
        return

    st.subheader("Inventory Report")
    items = get_items()
    df_items = pd.DataFrame(items)

    categories = sorted([c for c in df_items.get("category", pd.Series(dtype=str)).dropna().unique().tolist()])
    cat_select = st.multiselect("Categories to include", options=categories, default=categories)
    only_available = st.checkbox("Only available items (> 0 qty)", value=False)
    only_low = st.checkbox("Only low-stock rows", value=False)
    sort_by = st.selectbox("Sort by", options=["category","sku","name"], index=0)

    if not df_items.empty:
        use_cats = cat_select if cat_select else None
        pdf_bytes = _inventory_pdf_bytes_grouped(
            df_items, brand_name, brand_rgb, logo_path, revision_tag,
            prepared_by=prepared_by, only_low=only_low, sort_by=sort_by,
            categories=use_cats, only_available=only_available
        )
        st.download_button(
            "Download Inventory PDF",
            data=pdf_bytes,
            file_name=f"inventory_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No items to include in the report.")

    st.subheader("Transaction Log (PDF)")
    tx = get_transactions(limit=100_000)
    df_tx = pd.DataFrame(tx)
    if df_tx.empty:
        st.info("No transactions to include in the report.")
        return

    def df_to_pdf_bytes_pro(title: str, df: pd.DataFrame, meta_lines, brand_name: str, brand_rgb: tuple, logo_path: str, revision_tag: str) -> bytes:
        pdf = BrandedPDF(
            brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path, revision_tag=revision_tag,
            orientation="L", unit="mm", format="A3"
        )
        pdf.add_page()
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, to_latin1(title), ln=1)
        pdf.set_font("Helvetica", "", 10)
        for line in meta_lines:
            pdf.cell(0, 6, to_latin1(str(line)), ln=1)
        pdf.ln(2)

        def draw_table(pdf, df: pd.DataFrame, header_fill_rgb: tuple, font_size=9):
            pdf.set_font("Helvetica", "B", font_size)
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            measure_rows = min(len(df), 200)
            widths = []
            for col in df.columns:
                max_w = pdf.get_string_width(to_latin1(str(col))) + 6
                pdf.set_font("Helvetica", "", font_size)
                for i in range(measure_rows):
                    w = pdf.get_string_width(to_latin1(str(df.iloc[i][col]))) + 6
                    if w > max_w:
                        max_w = w
                max_w = max(18, min(max_w, 100))
                widths.append(max_w)
            total_w = sum(widths)
            if total_w > page_w:
                scale = page_w / total_w
                widths = [w * scale for w in widths]
            else:
                extra = page_w - total_w
                if extra > 0 and len(widths) > 0:
                    bump = extra / len(widths)
                    widths = [w + bump for w in widths]

            pdf.set_font("Helvetica", "B", font_size)
            pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
            pdf.set_text_color(20, 20, 20)
            row_h = 7
            y0 = pdf.get_y()
            for w, col in zip(widths, df.columns):
                x = pdf.get_x()
                pdf.multi_cell(w, row_h, to_latin1(str(col)), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
                pdf.set_xy(x + w, y0)
            pdf.ln(row_h)

            pdf.set_font("Helvetica", "", font_size)
            pdf.set_text_color(15, 15, 15)
            for idx in range(len(df)):
                y_start = pdf.get_y()
                max_h = row_h
                if y_start + max_h > pdf.h - pdf.b_margin:
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", font_size)
                    pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
                    pdf.set_text_color(20, 20, 20)
                    y0 = pdf.get_y()
                    for w, col in zip(widths, df.columns):
                        x = pdf.get_x()
                        pdf.multi_cell(w, row_h, to_latin1(str(col)), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
                        pdf.set_xy(x + w, y0)
                    pdf.ln(row_h)
                    pdf.set_font("Helvetica", "", font_size)
                    pdf.set_text_color(15, 15, 15)
                    y_start = pdf.get_y()

                x_left = pdf.get_x()
                for w, val in zip(widths, df.iloc[idx].tolist()):
                    txt = to_latin1(str(val))
                    x = pdf.get_x()
                    pdf.multi_cell(w, row_h, txt, border=1, align="L", new_x="RIGHT", new_y="TOP")
                    pdf.set_xy(x + w, y_start)
                pdf.set_xy(x_left, y_start + row_h)

        draw_table(pdf, df, brand_rgb, font_size=9)
        data = pdf.output(dest="S")
        return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")

    col_order = ["ts", "sku", "qty_change", "reason", "project", "reference", "user", "notes"]
    col_names = {"ts":"Timestamp","sku":"SKU","qty_change":"Δ Qty","reason":"Reason","project":"Project/Job","reference":"Reference","user":"User","notes":"Notes"}
    df_tx = pd.DataFrame(tx)
    df_tx = df_tx.loc[:, [c for c in col_order if c in df_tx.columns]].rename(columns=col_names)
    meta = [f"Rows: {len(df_tx)}"]
    if prepared_by:
        meta.append(f"Prepared by: {prepared_by}")
    pdf_bytes_tx = df_to_pdf_bytes_pro("Transaction Log", df_tx, meta, brand_name, brand_rgb, logo_path, revision_tag)
    st.download_button("Download Transactions PDF", data=pdf_bytes_tx, file_name=f"transactions_{timestamp()}.pdf", mime="application/pdf")

def view_maintenance():
    st.title("🛠️ Maintenance (Lightweight)")
    st.caption("Log maintenance-related stock usage (consumables/spares).")
    items = get_items()
    if not items:
        st.info("Add items first in the Inventory page.")
        return

    sku_list = [i["sku"] for i in items]
    with st.form("maint_form"):
        cols = st.columns(4)
        sku = cols[0].selectbox("Item SKU", options=sku_list)
        qty_used = cols[1].number_input("Qty Used (negative)", value=-1.0, step=1.0, format="%.3f")
        project = cols[2].text_input("Home/Workshop area (e.g., Bathroom Reno)")
        user = cols[3].text_input("Person")
        notes = st.text_area("Notes (what/where/why)")

        submitted = st.form_submit_button("Log Maintenance Usage")
        if submitted:
            if not sku or qty_used >= 0:
                st.error("Choose a SKU and enter a negative quantity to deduct.")
            else:
                add_transaction(sku, qty_used, reason="maintenance", project=project, reference="", user=user, notes=notes)
                st.success("Maintenance usage logged (stock deducted).")

def view_settings():
    st.title("⚙️ Settings")
    st.caption("Branding and display options.")

    current_brand = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
    current_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
    current_logo = get_setting("logo_path", CONFIG.get("logo_path", ""))
    current_rev  = get_setting("revision_tag", CONFIG.get("revision_tag", "Rev0.1"))
    current_prepared = get_setting("prepared_by", "")

    col1, col2 = st.columns(2)
    brand_name_in = col1.text_input("Brand Name", value=current_brand)
    brand_color_in = col2.color_picker("Brand Color", value=current_color)
    rev = st.text_input("Revision tag to show on PDFs (e.g., Rev0.1)", value=current_rev)
    prepared_by_in = st.text_input("Prepared by (appears on PDFs)", value=current_prepared)

    st.subheader("Logo")
    upload = st.file_uploader("Upload a PNG/JPG logo", type=["png", "jpg", "jpeg"])
    bundled = ["assets/logo_OMEC.png", "assets/logo_PG_Bison.png"]
    if os.path.exists(_norm_path("assets/logo_custom.png")):
        bundled.append("assets/logo_custom.png")
    selected = st.selectbox("Or choose a bundled logo", options=bundled, index=0)

    if upload:
        save_dir = _norm_path("assets")
        os.makedirs(save_dir, exist_ok=True)
        upath = os.path.join(save_dir, "logo_custom.png")
        with open(upath, "wb") as f:
            f.write(upload.read())
        selected = "assets/logo_custom.png"

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Save Settings"):
        upsert_setting("brand_name", brand_name_in)
        upsert_setting("brand_color", brand_color_in)
        upsert_setting("logo_path", selected)
        upsert_setting("revision_tag", rev)
        upsert_setting("prepared_by", prepared_by_in)
        st.success("Settings saved. Refresh to apply.")
    if c2.button("Clear Logo"):
        upsert_setting("logo_path", "")
        st.success("Logo cleared. Refresh to apply.")
    if c3.button("Use Default OMEC"):
        upsert_setting("brand_name", "OMEC")
        upsert_setting("brand_color", "#0ea5e9")
        upsert_setting("logo_path", "assets/logo_OMEC.png")
        upsert_setting("revision_tag", "Rev0.1")
        upsert_setting("prepared_by", "")
        st.success("Default branding applied. Refresh to see it.")
    if c4.button("Rev++"):
        upsert_setting("revision_tag", rev_bump(rev))
        st.success("Revision bumped. Refresh Reports to see it.")

# ---------- Router ----------
if menu == "Dashboard":
    view_dashboard()
elif menu == "Inventory":
    view_inventory()
elif menu == "Transactions":
    view_transactions()
elif menu == "Versions & Snapshots":
    view_versions()
elif menu == "Reports & Export (PDF)":
    view_reports()
elif menu == "Maintenance":
    view_maintenance()
else:
    view_settings()
