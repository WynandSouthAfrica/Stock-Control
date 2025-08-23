
import streamlit as st
import pandas as pd
from db import get_items, add_transaction

st.set_page_config(page_title="Maintenance", page_icon="🛠️", layout="wide")

st.title("🛠️ Maintenance (Lightweight)")
st.caption("Log maintenance-related stock usage (consumables/spares against an item).")

items = get_items()
if not items:
    st.info("Add items first in the Inventory page.")
else:
    sku_list = [i['sku'] for i in items]

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
