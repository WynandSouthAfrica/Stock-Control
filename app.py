# app.py — OMEC Stock Take (single-file)
# Safe logo + Professional PDF exports (A3 Landscape) with:
# - totals row (sum qty + total stock value)
# - low-stock rows highlighted in red
# - grouped inventory report by Category with subtotals
import os, json, math
import streamlit as st
import pandas as pd

from db import (
    init_db, get_items, add_or_update_item, delete_item,
    get_transactions, add_transaction,
    get_versions, save_version_record,
    upsert_setting, get_setting
)
from utils import export_snapshot, timestamp

# PDF export (fpdf2) — safe import so app still runs if the package is missing
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
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": ""}

init_db()  # ensure SQLite schema exists

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
        return (14, 165, 233)  # fallback cyan-ish

def lighten(rgb, factor=0.85):
    r, g, b = rgb
    return (int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))

if FPDF_AVAILABLE:
    class BrandedPDF(FPDF):
        def __init__(self, brand_name: str, brand_rgb: tuple, logo_path: str = "", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.brand_name = brand_name
            self.brand_rgb = brand_rgb
            self.logo_path = _norm_path(logo_path) if logo_path else ""
            self.set_auto_page_break(auto=True, margin=12)

        def header(self):
            # Top brand bar
            self.set_fill_color(*lighten(self.brand_rgb, 0.75))
            self.rect(x=0, y=0, w=self.w, h=14, style="F")

            # Logo (if any)
            x = 10
            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    self.image(self.logo_path, x=x, y=3, h=8)
                    x += 26
                except Exception:
                    pass

            # Brand name
            self.set_xy(x, 4)
            self.set_text_color(25, 25, 25)
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 6, txt=self.brand_name, ln=0)

            # Date on right
            self.set_xy(-80, 4)
            self.set_font("Helvetica", "", 9)
            self.cell(70, 6, txt=f"Generated: {timestamp()}", ln=0, align="R")

            # Second divider line
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
            self.cell(0, 6, f"Page {self.page_no()}", align="R")

# --- PDF utils (inventory grouped/subtotals/low-stock highlighting) ----------
def _text_lines_needed(pdf, w, txt, line_h):
    if txt is None:
        return 1
    s = str(txt)
    if not s:
        return 1
    # crude wrap estimate
    char_w = max(pdf.get_string_width("M"), 1e-6)
    max_chars = max(int((w - 2) / (char_w if char_w else 1)), 1)
    return max(1, math.ceil(len(s) / max(1, max_chars)))

def _compute_col_widths(pdf, columns, rows, page_w, font_size):
    pdf.set_font("Helvetica", "B", font_size)
    widths = []
    for c in columns:
        max_w = pdf.get_string_width(str(c)) + 6
        pdf.set_font("Helvetica", "", font_size)
        for r in rows:
            w = pdf.get_string_width(str(r.get(c, ""))) + 6
            if w > max_w:
                max_w = w
        max_w = max(18, min(max_w, 80))
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
        pdf.multi_cell(w, row_h, str(col), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_xy(x + w, y0)
    pdf.ln(row_h)
    return row_h

def _ensure_page_space(pdf, needed_h, columns, widths, font_size, brand_rgb):
    if pdf.get_y() + needed_h <= pdf.h - pdf.b_margin:
        return
    pdf.add_page()
    _draw_header_row(pdf, columns, widths, font_size, brand_rgb)

def _draw_row(pdf, values, widths, row_h, align_map=None, fill_rgb=None, bold=False, text_rgb=(15,15,15)):
    if align_map is None:
        align_map = {}
    pdf.set_font("Helvetica", "B" if bold else "", 9)
    if fill_rgb:
        pdf.set_fill_color(*fill_rgb)
    pdf.set_text_color(*text_rgb)

    y_start = pdf.get_y()
    heights = []
    for w, v in zip(widths, values):
        lines = _text_lines_needed(pdf, w, v, row_h)
        heights.append(lines * row_h)
    max_h = max(heights) if heights else row_h

    # draw cells
    x_left = pdf.get_x()
    for idx, (w, v) in enumerate(zip(widths, values)):
        x = pdf.get_x()
        align = align_map.get(idx, "L")
        if fill_rgb:
            pdf.set_fill_color(*fill_rgb)
            pdf.multi_cell(w, row_h, str(v if v is not None else ""), border=1, align=align,
                           new_x="RIGHT", new_y="TOP", fill=True)
        else:
            pdf.multi_cell(w, row_h, str(v if v is not None else ""), border=1, align=align,
                           new_x="RIGHT", new_y="TOP")
        pdf.set_xy(x + w, y_start)
    pdf.set_xy(x_left, y_start + max_h)

def _inventory_pdf_bytes_grouped(df: pd.DataFrame, brand_name, brand_rgb, logo_path) -> bytes:
    # Derive value column
    df = df.copy()
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
    df["min_qty"] = pd.to_numeric(df.get("min_qty"), errors="coerce").fillna(0.0)
    df["unit_cost"] = pd.to_numeric(df.get("unit_cost"), errors="coerce").fillna(0.0)
    df["value"] = (df["quantity"] * df["unit_cost"]).round(2)

    # Order & rename for display
    key_cols = ["sku", "name", "category", "location", "unit", "quantity", "min_qty", "unit_cost", "value", "updated_at"]
    present = [c for c in key_cols if c in df.columns]
    df = df[present]

    col_names = {
        "sku": "SKU", "name": "Name", "category": "Category", "location": "Location",
        "unit": "Unit", "quantity": "Qty", "min_qty": "Min", "unit_cost": "Unit Cost (R)",
        "value": "Value (R)", "updated_at": "Updated"
    }
    display_cols = [col_names.get(c, c) for c in present]

    # Format rows for width estimation
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

    # Build PDF
    pdf = BrandedPDF(brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path,
                     orientation="L", unit="mm", format="A3")
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)

    # Title + meta
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Inventory Report", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Rows: {len(df)}", ln=1)
    pdf.ln(2)

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    widths = _compute_col_widths(pdf, display_cols, rows_for_width, page_w, font_size=9)
    header_h = _draw_header_row(pdf, display_cols, widths, 9, brand_rgb)

    # Alignment (numbers right aligned)
    align_map = {present.index(c): "R" for c in present if c in ("quantity", "min_qty", "unit_cost", "value")}

    # Group by category
    if "category" in df.columns:
        df_sorted = df.sort_values(by=["category", "sku", "name"], kind="stable")
        categories = df_sorted["category"].fillna("(Unspecified)").unique().tolist()
    else:
        df_sorted = df.copy()
        df_sorted["category"] = "(All)"
        categories = ["(All)"]

    light_brand = lighten(brand_rgb, 0.92)
    cat_bar = lighten(brand_rgb, 0.80)
    low_stock_fill = (255, 235, 235)

    grand_qty = 0.0
    grand_value = 0.0

    for cat in categories:
        block = df_sorted[df_sorted["category"].fillna("(Unspecified)") == cat]

        # Category bar
        span_h = 7
        _ensure_page_space(pdf, span_h + header_h, display_cols, widths, 9, brand_rgb)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(*cat_bar)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(sum(widths), span_h, f"Category: {cat}", border=1, ln=1, fill=True)

        cat_qty = float(block["quantity"].sum())
        cat_value = float((block["quantity"] * block["unit_cost"]).sum())
        grand_qty += cat_qty
        grand_value += cat_value

        # Rows
        for _, r in block.iterrows():
            vals = []
            for c in present:
                v = r[c]
                if c in ("quantity", "min_qty"):
                    v = fmt_num(float(v), 2)
                elif c in ("unit_cost", "value"):
                    v = fmt_num(float(v), 2)
                vals.append(v if v is not None else "")
            # low-stock highlight
            fill = low_stock_fill if float(r["quantity"]) < float(r["min_qty"]) else None
            row_h = 7
            _ensure_page_space(pdf, row_h + 2, display_cols, widths, 9, brand_rgb)
            _draw_row(pdf, vals, widths, row_h, align_map=align_map, fill_rgb=fill)

        # Category subtotal
        sub_vals = []
        for c in present:
            if c == "sku":
                sub_vals.append("Subtotal")
            elif c == "quantity":
                sub_vals.append(fmt_num(cat_qty, 2))
            elif c == "value":
                sub_vals.append(fmt_num(cat_value, 2))
            else:
                sub_vals.append("")
        _ensure_page_space(pdf, 7, display_cols, widths, 9, brand_rgb)
        _draw_row(pdf, sub_vals, widths, 7, align_map=align_map, fill_rgb=light_brand, bold=True)

        pdf.ln(1)

    # Grand total row
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
    _draw_row(pdf, total_vals, widths, 8, align_map=align_map, fill_rgb=lighten(brand_rgb, 0.88), bold=True)

    data = pdf.output(dest="S")
    return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")

# Generic table (used for Transactions PDF)
def _text_lines_needed_table(pdf, w, txt, line_h):
    if not txt:
        return 1
    char_w = max(pdf.get_string_width("M"), 1e-6)
    max_chars = max(int((w - 2) / (char_w if char_w else 1)), 1)
    return max(1, math.ceil(len(str(txt)) / max(1, max_chars)))

def draw_table(pdf, df: pd.DataFrame, header_fill_rgb: tuple, col_order=None, col_rename=None, font_size=9):
    if col_order is not None:
        df = df.loc[:, [c for c in col_order if c in df.columns]]
    if col_rename:
        df = df.rename(columns=col_rename)

    pdf.set_font("Helvetica", "B", font_size)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    measure_rows = min(len(df), 200)

    widths = []
    for col in df.columns:
        max_w = pdf.get_string_width(str(col)) + 6
        pdf.set_font("Helvetica", "", font_size)
        for i in range(measure_rows):
            w = pdf.get_string_width(str(df.iloc[i][col])) + 6
            if w > max_w:
                max_w = w
        max_w = max(18, min(max_w, 80))
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

    # Header
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
    pdf.set_text_color(20, 20, 20)
    row_h = 7
    y0 = pdf.get_y()
    for w, col in zip(widths, df.columns):
        x = pdf.get_x()
        pdf.multi_cell(w, row_h, str(col), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_xy(x + w, y0)
    pdf.ln(row_h)

    # Rows
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_text_color(15, 15, 15)
    for idx in range(len(df)):
        y_start = pdf.get_y()
        heights = []
        for w, val in zip(widths, df.iloc[idx].tolist()):
            lines = _text_lines_needed_table(pdf, w, str(val), row_h)
            heights.append(lines * row_h)
        max_h = max(heights) if heights else row_h

        if y_start + max_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            # redraw header
            pdf.set_font("Helvetica", "B", font_size)
            pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
            pdf.set_text_color(20, 20, 20)
            y0 = pdf.get_y()
            for w, col in zip(widths, df.columns):
                x = pdf.get_x()
                pdf.multi_cell(w, row_h, str(col), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
                pdf.set_xy(x + w, y0)
            pdf.ln(row_h)
            pdf.set_font("Helvetica", "", font_size)
            pdf.set_text_color(15, 15, 15)
            y_start = pdf.get_y()

        x_left = pdf.get_x()
        for w, val in zip(widths, df.iloc[idx].tolist()):
            txt = str(val)
            x = pdf.get_x()
            pdf.multi_cell(w, row_h, txt, border=1, align="L", new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x + w, y_start)
        pdf.set_xy(x_left, y_start + max_h)

def df_to_pdf_bytes_pro(title: str, df: pd.DataFrame, meta_lines, brand_name: str, brand_rgb: tuple, logo_path: str) -> bytes:
    pdf = BrandedPDF(brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path,
                     orientation="L", unit="mm", format="A3")
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, txt=title, ln=1)
    pdf.set_font("Helvetica", "", 10)
    for line in meta_lines:
        pdf.cell(0, 6, txt=str(line), ln=1)
    pdf.ln(2)
    draw_table(pdf, df, header_fill_rgb=brand_rgb, font_size=9)

    data = pdf.output(dest="S")
    return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")

# ---------- Settings (with config defaults) ----------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
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

# ---------- Views ----------
def view_dashboard():
    st.title("🏠 Dashboard")
    st.caption("Quick overview of your stock status.")
    items = get_items()
    tx = get_transactions(limit=10)

    col1, col2, col3, col4 = st.columns(4)
    total_items = len(items)
    total_qty = sum((i.get("quantity") or 0) for i in items)
    low_stock = sum(1 for i in items if (i.get("min_qty") or 0) > (i.get("quantity") or 0))
    total_value = sum((i.get("quantity") or 0) * (i.get("unit_cost") or 0) for i in items)

    col1.metric("Distinct SKUs", total_items)
    col2.metric("Total Quantity", f"{total_qty:.2f}")
    col3.metric("Low-Stock Items", low_stock)
    col4.metric("Stock Value", f"R {total_value:,.2f}")

    st.subheader("Recent Transactions")
    if tx:
        st.dataframe(pd.DataFrame(tx), use_container_width=True)
    else:
        st.info("No transactions yet.")

def view_inventory():
    st.title("📦 Inventory")
    st.caption("Add, edit, or delete items.")

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

    st.subheader("Inventory List")
    items = get_items()
    df = pd.DataFrame(items)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        to_delete = st.text_input("Delete by SKU")
        if st.button("Delete Item"):
            if to_delete:
                delete_item(to_delete.strip())
                st.success(f"Deleted '{to_delete}'")
            else:
                st.error("Enter a SKU to delete.")
    else:
        st.info("No items yet. Add your first item above.")

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

    st.subheader("Recent Transactions")
    tx = get_transactions(limit=500)
    if tx:
        st.dataframe(pd.DataFrame(tx), use_container_width=True)
    else:
        st.info("No transactions yet.")

def view_versions():
    st.title("🕒 Versions & Snapshots")
    st.caption("Create timestamped ZIP archives of your data for traceability.")
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

    st.subheader("History")
    versions = get_versions()
    if versions:
        st.dataframe(pd.DataFrame(versions), use_container_width=True)
    else:
        st.info("No versions yet.")

def view_reports():
    st.title("🧾 Reports & Export (PDF)")
    st.caption("A3 landscape • group by category • subtotals • low-stock highlight • totals row")

    if not FPDF_AVAILABLE:
        st.error("PDF engine not available. Add `fpdf2==2.7.9` to requirements.txt.")
        return

    # ---------- Inventory PDF (grouped) ----------
    st.subheader("Inventory Report")
    items = get_items()
    df_items = pd.DataFrame(items)

    if not df_items.empty:
        pdf_bytes = _inventory_pdf_bytes_grouped(df_items, brand_name, brand_rgb, logo_path)
        st.download_button(
            "Download Inventory PDF",
            data=pdf_bytes,
            file_name=f"inventory_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No items to include in the report.")

    # ---------- Transactions PDF (simple table) ----------
    st.subheader("Transaction Log")
    tx = get_transactions(limit=100_000)
    df_tx = pd.DataFrame(tx)

    if not df_tx.empty:
        col_order = ["ts", "sku", "qty_change", "reason", "project", "reference", "user", "notes"]
        col_names = {
            "ts": "Timestamp", "sku": "SKU", "qty_change": "Δ Qty", "reason": "Reason",
            "project": "Project/Job", "reference": "Reference", "user": "User", "notes": "Notes",
        }
        df_tx = df_tx.loc[:, [c for c in col_order if c in df_tx.columns]].rename(columns=col_names)
        pdf_bytes_tx = df_to_pdf_bytes_pro(
            "Transaction Log",
            df_tx,
            [f"Rows: {len(df_tx)}"],
            brand_name, brand_rgb, logo_path
        )
        st.download_button(
            "Download Transactions PDF",
            data=pdf_bytes_tx,
            file_name=f"transactions_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No transactions to include in the report.")

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
                add_transaction(
                    sku, qty_used, reason="maintenance", project=project, reference="", user=user, notes=notes
                )
                st.success("Maintenance usage logged (stock deducted).")

def view_settings():
    st.title("⚙️ Settings")
    st.caption("Branding and display options.")

    current_brand = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
    current_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
    current_logo = get_setting("logo_path", CONFIG.get("logo_path", ""))

    col1, col2 = st.columns(2)
    brand_name_in = col1.text_input("Brand Name", value=current_brand)
    brand_color_in = col2.color_picker("Brand Color", value=current_color)

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

    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("Save Settings"):
        upsert_setting("brand_name", brand_name_in)
        upsert_setting("brand_color", brand_color_in)
        upsert_setting("logo_path", selected)
        st.success("Settings saved. Refresh to apply.")
    if c2.button("Clear Logo"):
        upsert_setting("logo_path", "")
        st.success("Logo cleared. Refresh to apply.")
    if c3.button("Use Default OMEC"):
        upsert_setting("brand_name", "OMEC")
        upsert_setting("brand_color", "#0ea5e9")
        upsert_setting("logo_path", "assets/logo_OMEC.png")
        st.success("Default branding applied. Refresh to see it.")

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
