
import os
import streamlit as st
import pandas as pd
from db import get_items, get_transactions, save_version_record, get_versions
from utils import export_snapshot

st.set_page_config(page_title="Versions", page_icon="🕒", layout="wide")

st.title("🕒 Versions & Snapshots")
st.caption("Create timestamped ZIP archives of your data for traceability.")

tag = st.text_input("Version tag (e.g., V0.1, 'after_stock_count')")
note = st.text_area("Note")

if st.button("Create Snapshot ZIP"):
    items = get_items()
    tx = get_transactions(limit=1000000)
    zip_path = export_snapshot(items, tx, tag=tag, note=note)
    save_version_record(tag or "", note or "", zip_path)
    st.success(f"Snapshot created: {zip_path}")
    st.download_button("Download ZIP", data=open(zip_path, "rb").read(), file_name=os.path.basename(zip_path))

st.subheader("History")
versions = get_versions()
if versions:
    df = pd.DataFrame(versions)
    st.dataframe(df, use_container_width=True)
    # Allow download of selected row
    if "file_path" in df.columns:
        selected = st.selectbox("Select snapshot to download", options=df["file_path"].tolist())
        if selected and os.path.exists(selected):
            st.download_button("Download selected ZIP", data=open(selected, "rb").read(), file_name=os.path.basename(selected))
else:
    st.info("No versions yet. Create your first snapshot above.")
