
import streamlit as st
import pandas as pd
from db import get_items, get_transactions

st.set_page_config(page_title="Reports", page_icon="🧾", layout="wide")

st.title("🧾 Reports & Export")
st.caption("Filter and export inventory and transaction data (CSV).")

items = get_items()
tx = get_transactions(limit=100000)

st.subheader("Inventory")
df_items = pd.DataFrame(items)
if not df_items.empty:
    cat = st.multiselect("Filter by Category", sorted({i['category'] for i in items if i['category']}))
    loc = st.multiselect("Filter by Location", sorted({i['location'] for i in items if i['location']}))

    filtered = df_items.copy()
    if cat:
        filtered = filtered[filtered["category"].isin(cat)]
    if loc:
        filtered = filtered[filtered["location"].isin(loc)]

    st.dataframe(filtered, use_container_width=True)
    st.download_button("Export Inventory CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="inventory_export.csv")
else:
    st.info("No items to show.")

st.subheader("Transactions")
df_tx = pd.DataFrame(tx)
if not df_tx.empty:
    st.dataframe(df_tx, use_container_width=True)
    st.download_button("Export Transactions CSV", data=df_tx.to_csv(index=False).encode("utf-8"), file_name="transactions_export.csv")
else:
    st.info("No transactions found.")
