# app.py — OMEC Stock Take (single-file) — Safe logo + Professional PDF exports (no CSV)
import os, json, html, math
import streamlit as st
import pandas as pd

from db import (
    init_db, get_items, add_or_update_item, delete_item,
    get_transactions, add_transaction,
    get_versions, save_version_record,
    upsert_setting, get_setting
)
from utils import export_snapshot, timestamp

# PDF export (fpdf2)
from fpdf import FPDF

st.set_page_config(page_title="OMEC Stock Take", page_icon="🗃️", layout="wide")

# ---------- Config ------------------------------------------------------------
ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": ""}

init_db()  # ensure SQLite schema exists

# ---------- Helpers -----------------------------------------------------------
def _norm_path(p: str) -> str:
    """Absolute path from project root; '' if invalid."""
    if not isinstance(p, str) or not p.strip():
        return ""
    p = os.path.normpath(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def safe_show_logo(path: str):
    """Show logo if file exists; never crash."""
    try:
        apath = _norm_path(path)
        if apath and os.path.exists(apath):
            st.sidebar.image(apath, use_container_width=True)
    except Exception:
        pass

def hex_to_rgb(h: str):
    """#RRGGBB -> (r,g,b)."""
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (14, 165, 233)  # fallback cyan-ish

def lighten(rgb, factor=0.85):
    """Lighten color toward white by factor (0-1)."""
    r, g, b = rgb
    return (int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))

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

def draw_table(pdf: BrandedPDF, df: pd.DataFrame, header_fill_rgb: tuple, col_order=None, col_rename=None, font_size=9):
    """Draw a table across pages with auto column widths and wrapped cells."""
    if col_order is not None:
        df = df.loc[:, [c for c in col_order if c in df.columns]]
    if col_rename:
        df = df.rename(columns=col_rename)

    # Calculate column widths based on content
    pdf.set_font("Helvetica", "B", font_size)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    # max width in mm for each column based on header and first N rows
    measure_rows = min(len(df), 200)
    widths = []
    for col in df.columns:
        max_w = pdf.get_string_width(str(col)) + 4
        pdf.set_font("Helvetica", "", font_size)
        for i in range(measure_rows):
            w = pdf.get_string_width(str(df.iloc[i][col])) + 4
            if w > max_w:
                max_w = w
        # clamp per column
        max_w = max(18, min(max_w, 70))
        widths.append(max_w)

    # scale widths to fit page if necessary
    total_w = sum(widths)
    if total_w > page_w:
        scale = page_w / total_w
        widths = [w * scale for w in widths]
    else:
        # distribute remaining space to wider columns
        extra = page_w - total_w
        if extra > 0 and len(widths) > 0:
            bump = extra / len(widths)
            widths = [w + bump for w in widths]

    # Header row
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
    pdf.set_text_color(20, 20, 20)
    row_h = 7
    x0 = pdf.get_x()
    y0 = pdf.get_y()
    for w, col in zip(widths, df.columns):
        pdf.multi_cell(w, row_h, str(col), border=1, align="L", fill=True, ln=3, max_line_height=pdf.font_size)
        pdf.set_xy(pdf.get_x() + w, y0)
    pdf.ln(row_h)

    # Body rows
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_text_color(15, 15, 15)
    for idx in range(len(df)):
        y_before = pdf.get_y()
        max_cell_h = row_h
        # measure wrapped height per cell
        cell_texts = [str(v) for v in df.iloc[idx].tolist()]
        for w, txt in zip(widths, cell_texts):
            h = pdf.get_string_width(txt)  # proxy; we will use multi_cell anyway
            lines = max(1, math.ceil((pdf.get_string_width(txt) + 1) / (w - 2)))
            h = lines * row_h
            if h > max_cell_h:
                max_cell_h = h

        # page break check
        if pdf.get_y() + max_cell_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            # redraw header
            pdf.set_font("Helvetica", "B", font_size)
            pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
            pdf.set_text_color(20, 20, 20)
            y0 = pdf.get_y()
            for w, col in zip(widths, df.columns):
                pdf.multi_cell(w, row_h, str(col), border=1, align="L", fill=True, ln=3, max_line_height=pdf.font_size)
                pdf.set_xy(pdf.get_x() + w, y0)
            pdf.ln(row_h)
            pdf.set_font("Helvetica", "", font_size)
            pdf.set_text_color(15, 15, 15)

        x = pdf.get_x()
        y = pdf.get_y()
        for w, txt in zip(widths, cell_texts):
            pdf.multi_cell(w, row_h, txt, border=1, align="L", ln=3, max_line_height=pdf.font_size)
            pdf.set_xy(pdf.get_x() + w, y)
        pdf.ln(max_cell_h if max_cell_h > row_h else row_h)

def df_to_pdf_bytes_pro(
    title: str,
    df: pd.DataFrame,
    meta_lines,
    brand_name: str,
    brand_rgb: tuple,
    logo_path: str
) -> bytes:
    pdf = BrandedPDF(brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path, orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 15)
    pdf.ln(2)
    pdf.cell(0, 8, txt=title, ln=1)
    pdf.set_font("Helvetica", "", 10)
    for line in meta_lines:
        pdf.cell(0, 6, txt=str(line), ln=1)
    pdf.ln(2)

    draw_table(pdf, df, header_fill_rgb=brand_rgb, font_size=9)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# ---------- Settings (with config defaults) -----------------------------------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
brand_rgb = hex_to_rgb(brand_color)

# ---------- Sidebar -----------------------------------------------------------
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

# ---------- Views -------------------------------------------------------------
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
    st.caption("Generate branded PDF reports for inventory and transactions.")

    # ---------- Inventory PDF ----------
    st.subheader("Inventory Report")
    items = get_items()
    df_items = pd.DataFrame(items)

    if not df_items.empty:
        # Choose columns & order for a clean report
        col_order = ["sku", "name", "category", "location", "unit", "quantity", "min_qty", "unit_cost", "updated_at"]
        col_names = {
            "sku": "SKU",
            "name": "Name",
            "category": "Category",
            "location": "Location",
            "unit": "Unit",
            "quantity": "Qty",
            "min_qty": "Min",
            "unit_cost": "Unit Cost (R)",
            "updated_at": "Updated",
        }
        pdf_bytes = df_to_pdf_bytes_pro(
            "Inventory Report",
            df_items.loc[:, [c for c in col_order if c in df_items.columns]].rename(columns=col_names),
            [f"Rows: {len(df_items)}"],
            brand_name,
            brand_rgb,
            logo_path,
        )
        st.download_button(
            "Download Inventory PDF",
            data=pdf_bytes,
            file_name=f"inventory_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No items to include in the report.")

    # ---------- Transactions PDF ----------
    st.subheader("Transaction Log")
    tx = get_transactions(limit=100_000)
    df_tx = pd.DataFrame(tx)

    if not df_tx.empty:
        col_order = ["ts", "sku", "qty_change", "reason", "project", "reference", "user", "notes"]
        col_names = {
            "ts": "Timestamp",
            "sku": "SKU",
            "qty_change": "Δ Qty",
            "reason": "Reason",
            "project": "Project/Job",
            "reference": "Reference",
            "user": "User",
            "notes": "Notes",
        }
        pdf_bytes_tx = df_to_pdf_bytes_pro(
            "Transaction Log",
            df_tx.loc[:, [c for c in col_order if c in df_tx.columns]].rename(columns=col_names),
            [f"Rows: {len(df_tx)}"],
            brand_name,
            brand_rgb,
            logo_path,
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

# ---------- Router ------------------------------------------------------------
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
