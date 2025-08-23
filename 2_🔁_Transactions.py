
import streamlit as st
import pandas as pd
from db import get_items, add_transaction, get_transactions

st.set_page_config(page_title="Transactions", page_icon="🔁", layout="wide")

st.title("🔁 Transactions")
st.caption("Record stock movement in/out and maintain an audit trail.")

items = get_items()
sku_list = [i['sku'] for i in items]

with st.form("tx_form"):
    cols = st.columns(5)
    sku = cols[0].selectbox("SKU", options=sku_list)
    qty_change = cols[1].number_input("Qty Change (+ in / - out)", value=0.0, step=1.0, format="%.3f")
    reason = cols[2].selectbox("Reason", options=["receipt", "issue", "adjustment", "return", "count_correction", "other"])
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
