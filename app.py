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
                    "image_path": None
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
