
import os, json
import streamlit as st
from PIL import Image
import pandas as pd

from db import init_db, get_items, add_or_update_item, delete_item, get_transactions, add_transaction, get_versions, save_version_record, upsert_setting, get_setting
from utils import export_snapshot

st.set_page_config(page_title="OMEC Stock Take", page_icon="🗃️", layout="wide")

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": "assets/logo_OMEC.png"}

init_db()

# Sidebar branding
logo_path = get_setting("logo_path", CONFIG.get("logo_path"))
brand_name = get_setting("brand_name", CONFIG.get("brand_name"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color"))

st.sidebar.markdown(f"<h2 style='color:{brand_color}; margin-bottom:0'>{brand_name}</h2>", unsafe_allow_html=True)
if logo_path and os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Navigation")
st.sidebar.page_link("app.py", label="🏠 Dashboard")
st.sidebar.page_link("pages/1_📦_Inventory.py", label="📦 Inventory")
st.sidebar.page_link("pages/2_🔁_Transactions.py", label="🔁 Transactions")
st.sidebar.page_link("pages/3_🕒_Versions.py", label="🕒 Versions & Snapshots")
st.sidebar.page_link("pages/4_🧾_Reports.py", label="🧾 Reports & Export")
st.sidebar.page_link("pages/5_🛠️_Maintenance.py", label="🛠️ Maintenance")
st.sidebar.page_link("pages/6_⚙️_Settings.py", label="⚙️ Settings")

st.title("🏠 Dashboard")
st.caption("Quick overview of your stock status.")

items = get_items()
tx = get_transactions(limit=10)

col1, col2, col3, col4 = st.columns(4)
total_items = len(items)
total_qty = sum([i['quantity'] or 0 for i in items])
low_stock = sum([1 for i in items if (i['min_qty'] or 0) > (i['quantity'] or 0)])
total_value = sum([(i['quantity'] or 0) * (i['unit_cost'] or 0) for i in items])

col1.metric("Distinct SKUs", total_items)
col2.metric("Total Quantity", f"{total_qty:.2f}")
col3.metric("Low-Stock Items", low_stock)
col4.metric("Stock Value", f"R {total_value:,.2f}")

st.subheader("Recent Transactions")
if tx:
    st.dataframe(pd.DataFrame(tx))
else:
    st.info("No transactions yet. Head to **Transactions** to log movement.")
