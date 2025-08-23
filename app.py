# app.py — OMEC Stock Take (single-file) — Safe logo + CSV/PDF exports
import os, json, html
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

# ---------- Config ------------------------------------------------------------
ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": ""}

init_db()  # ensures SQLite schema exists

# ---------- Helpers -----------------------------------------------------------
def _norm_path(p: str) -> str:
    """Make absolute path from project root; return '' if invalid/empty."""
    if not isinstance(p, str) or not p.strip():
        return ""
    p = os.path.normpath(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def safe_show_logo(path: str):
    """Show logo if file exists; never crash if invalid."""
    try:
        apath = _norm_path(path)
        if apath and os.path.exists(apath):
            st.sidebar.image(apath, use_container_width=True)
    except Exception:
        pass

class PDF(FPDF, HTMLMixin):
    """Simple PDF with HTML table support via fpdf2."""
    pass

def df_to_pdf_bytes(title: str, df: pd.DataFrame, meta=None) -> bytes:
    """Render DataFrame to a landscape A4 PDF and return bytes."""
    if meta is None:
        meta = []
    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, txt=title, ln=1)

    pdf.set_font("Helvetica", "", 10)
    for line in meta:
        pdf.cell(0, 6, txt=str(line), ln=1)

    # Build HTML table
    parts = ['<table border="1" width="100%"><thead><tr>']
    for col in df.columns:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for v in row:
            parts.append(f"<td>{html.escape(str(v))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    pdf.write_html("".join(parts))

    return pdf.output(dest="S").encode("latin-1")

# ---------- Settings (with config defaults) -----------------------------------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))

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
        "Reports & Export",
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
    st.title("🧾 Reports & Export")
    st.caption("Filter and export inventory and transaction data (CSV/PDF).")

    items = get_items()
    tx = get_transactions(limit=100_000)

    # ----- Inventory ----------------------------------------------------------
    st.subheader("Inventory")
    df_items = pd.DataFrame(items)
    if not df_items.empty:
        cat = st.multiselect("Filter by Category", sorted({i.get("category") for i in items if i.get("category")}))
        loc = st.multiselect("Filter by Location", sorted({i.get("location") for i in items if i.get("location")}))
        filtered = df_items.copy()
        if cat:
            filtered = filtered[filtered["category"].isin(cat)]
        if loc:
            filtered = filtered[filtered["location"].isin(loc)]

        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Export Inventory CSV (comma)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"inventory_{timestamp()}.csv",
        )
        st.download_button(
            "Export Inventory CSV (semicolon • Excel)",
            data=filtered.to_csv(index=False, sep=";").encode("utf-8"),
            file_name=f"inventory_{timestamp()}_semicolon.csv",
        )

        inv_pdf = df_to_pdf_bytes("Inventory Report", filtered, [f"Generated: {timestamp()}", f"Rows: {len(filtered)}"])
        st.download_button(
            "Download Inventory PDF",
            data=inv_pdf,
            file_name=f"inventory_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No items to show.")

    # ----- Transactions -------------------------------------------------------
    st.subheader("Transactions")
    df_tx = pd.DataFrame(tx)
    if not df_tx.empty:
        st.dataframe(df_tx, use_container_width=True)

        st.download_button(
            "Export Transactions CSV (comma)",
            data=df_tx.to_csv(index=False).encode("utf-8"),
            file_name=f"transactions_{timestamp()}.csv",
        )
        st.download_button(
            "Export Transactions CSV (semicolon • Excel)",
            data=df_tx.to_csv(index=False, sep=";").encode("utf-8"),
            file_name=f"transactions_{timestamp()}_semicolon.csv",
        )

        tx_pdf = df_to_pdf_bytes("Transaction Log", df_tx, [f"Generated: {timestamp()}", f"Rows: {len(df_tx)}"])
        st.download_button(
            "Download Transactions PDF",
            data=tx_pdf,
            file_name=f"transactions_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No transactions found.")

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
elif menu == "Reports & Export":
    view_reports()
elif menu == "Maintenance":
    view_maintenance()
else:
    view_settings()
