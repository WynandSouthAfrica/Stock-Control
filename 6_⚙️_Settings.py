
import os, json
import streamlit as st
from db import upsert_setting, get_setting

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.caption("Branding and display options.")

default_logo = get_setting("logo_path", "assets/logo_OMEC.png")
default_brand = get_setting("brand_name", "OMEC")
default_color = get_setting("brand_color", "#0ea5e9")

col1, col2 = st.columns(2)
brand_name = col1.text_input("Brand Name", value=default_brand)
brand_color = col2.color_picker("Brand Color", value=default_color)

st.subheader("Logo")
upload = st.file_uploader("Upload a PNG/JPG logo", type=["png","jpg","jpeg"])
selected_logo = st.selectbox("Or choose a bundled logo", options=[
    "assets/logo_OMEC.png",
    "assets/logo_PG_Bison.png"
], index=0 if str(default_logo).endswith("OMEC.png") else 1)

if upload:
    save_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(save_dir, exist_ok=True)
    upload_path = os.path.join(save_dir, "logo_custom.png")
    with open(upload_path, "wb") as f:
        f.write(upload.read())
    # store relative path from project root
    selected_logo = os.path.relpath(upload_path, os.path.dirname(__file__)).replace(".."+os.sep, "")

if st.button("Save Settings"):
    upsert_setting("brand_name", brand_name)
    upsert_setting("brand_color", brand_color)
    upsert_setting("logo_path", selected_logo)
    st.success("Settings saved. Refresh the page to apply.")
