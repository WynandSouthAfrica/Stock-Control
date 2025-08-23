# app.py — OMEC Stock Take (single-file app) — SAFE LOGO + PDF (fpdf2)
import os, json, io
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
from fpdf import FPDF, HTMLMixin

st.set_page_config(page_title="OMEC Stock Take", page_icon="🗃️", layout="wide")

# ------------- Config ---------------------------------------------------------
ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": ""}

init_db()

# ------------- Helpers --------------------------------------------------------
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

class PDF(FPDF, HTMLMixin):
    pass

def df_to_pdf_bytes(title: str, df: pd.DataFrame, meta=None) -> bytes:
    """Render a DataFrame as a simple landscape A4 PDF using fpdf2's HTML table."""
    if meta is None:
        meta = []
    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, txt=title, ln=1)
    pdf.set_font("Helvetica", "", 10)
    for line in meta:
        pdf.cell(0, 6, txt=line, ln=1)

    # Build HTML table
    html = ['<table border="1" width="100%"><thead><tr>']
    for col in df.columns:
        html.append(f"<th>{str(col)}</th>")
    html.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        html.extend(f"<td>{str(v)}</td>" for v in row)
        html.append("</tr>")
    html.append("</tbody></table>")
    pdf.write_html("".join(html))

    # Return PDF bytes
    return pdf.output(dest="S").encode("latin-1")

# ------------- Settings -------------------------------------------------------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))

# ------------- Sidebar --------------------------------------------------------
st.sidebar.markdown(
    f"<h2 style='color:{brand_color}; margin-bottom:0'>{brand_name}</h2>",
    unsafe_allow_html=True
)
safe_show_logo(logo_path)

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Inventory", "Transactions", "Versions & Snapshots", "Reports & Export", "Maintenance", "Settings"],
    index=0
)

# ------------- Views ----------------------------------------------------------
def view_dashboard():
    st.title("🏠 Dashboard")
    st.caption("Quick overview of your stock status.")
    items = get_items()
    tx = get_transactions(limit=10)

    col1, col2, col3, col4 = st.columns(4)
    total_items = len(items)
    total_qty = sum([(i.get("quantity") or 0) for i in items])
    low_stock = sum([1 for i in items if (i.get("min_qty") or 0) > (i.get("quantity") or 0)])
    total_value = sum([(i.get("quantity") or 0) * (i.get("unit_cost") or 0) for i in items])

    col1.metric("Distinct SKUs", total_items)
    col2.metric("Total Quantity", f"{total_qty:.2f}")
    col3.metric("Low-Stock Items", low_stock)
    col4.metric("Stock Value", f"R {total_value:,.2f}")

    st.subheader("Recent Transactions")
    if tx:
        st.dataframe(pd.DataFrame(tx), use_container_width=True)
    else:
        st.info("No transactions yet.")

def vie
