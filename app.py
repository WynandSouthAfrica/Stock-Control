# app.py — OMEC Stock Take (Streamlit)
# Adds Return Sheet section to Issue Sheets PDF (ruled lines). Everything else preserved.

import os, json, re, io, zipfile, glob, math
import datetime as dt
import urllib.parse
import streamlit as st
import pandas as pd

from db import (
    init_db, get_items, add_or_update_item, delete_item,
    get_transactions, add_transaction,
    get_versions, save_version_record,
    upsert_setting, get_setting
)
from utils import export_snapshot, timestamp


# ---------- PDF engine ----------
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False


st.set_page_config(page_title="OMEC Stock Take", page_icon="🗃️", layout="wide")


# ---------- Config ----------
ROOT = os.path.dirname(__file__)
SNAP_DIR = os.path.join(ROOT, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": "", "revision_tag": "Rev0.1"}

init_db()


# ---------- Helpers ----------
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

def hex_to_rgb(h: str):
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (14, 165, 233)

def lighten(rgb, factor=0.85):
    r, g, b = rgb
    return (int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))

def to_latin1(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.encode("latin-1", "replace").decode("latin-1")

def normalize_category(cat):
    if cat is None: return None
    s = str(cat).strip()
    if not s: return None
    s = re.sub(r"\s+", " ", s)
    return s.title()


# ---------- Settings ----------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OMEC"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
revision_tag = get_setting("revision_tag", CONFIG.get("revision_tag", "Rev0.1"))
prepared_by = get_setting("prepared_by", "")
checked_by  = get_setting("checked_by", "")
approved_by = get_setting("approved_by", "")
email_recipients = get_setting("email_recipients", "")
auto_backup_enabled = str(get_setting("auto_backup_enabled", "true")).lower() in {"1","true","yes","on"}
brand_rgb = hex_to_rgb(brand_color)


# ---------- Sidebar ----------
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
        "Issue Sheets",
        "Versions & Snapshots",
        "Reports & Export (PDF)",
        "Maintenance",
    ],
    index=0,
)


# ---------- Auto-backup (daily) ----------
def _has_snapshot_for_today():
    patt = os.path.join(SNAP_DIR, f"*{dt.date.today().strftime('%Y%m%d')}*.zip")
    return bool(glob.glob(patt))

if auto_backup_enabled and not _has_snapshot_for_today():
    try:
        items = get_items()
        tx = get_transactions(limit=1_000_000)
        path = export_snapshot(items, tx, tag=f"Auto_{dt.date.today().isoformat()}", note="Auto-backup on app open")
        save_version_record(f"Auto_{dt.date.today().isoformat()}", "Auto-backup", path)
        st.sidebar.success("Auto-backup snapshot created for today.")
    except Exception:
        st.sidebar.warning("Auto-backup attempt failed (non-blocking).")


# ---------- PDF base ----------
if FPDF_AVAILABLE:
    class BrandedPDF(FPDF):
        def __init__(self, brand_name: str, brand_rgb: tuple, logo_path: str = "", revision_tag: str = "Rev0.1", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.brand_name = brand_name
            self.brand_rgb = brand_rgb
            self.revision_tag = revision_tag
            self.logo_path = _norm_path(logo_path) if logo_path else ""
            self.set_auto_page_break(auto=True, margin=12)

        def header(self):
            self.set_fill_color(*lighten(self.brand_rgb, 0.75))
            self.rect(x=0, y=0, w=self.w, h=14, style="F")
            x = 10
            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    self.image(self.logo_path, x=x, y=3, h=8)
                    x += 26
                except Exception:
                    pass
            self.set_xy(x, 4)
            self.set_text_color(25, 25, 25)
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 6, txt=to_latin1(self.brand_name), ln=0)
            self.set_xy(-80, 4)
            self.set_font("Helvetica", "", 9)
            self.cell(70, 6, txt=to_latin1(f"Revision: {self.revision_tag}"), ln=0, align="R")
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
            self.cell(0, 6, to_latin1(f"Page {self.page_no()}"), align="R")


def _compute_col_widths(pdf, columns, rows, page_w, font_size):
    pdf.set_font("Helvetica", "B", font_size)
    widths = []
    for c in columns:
        max_w = pdf.get_string_width(to_latin1(str(c))) + 6
        pdf.set_font("Helvetica", "", font_size)
        for r in rows[:200]:
            w = pdf.get_string_width(to_latin1(str(r.get(c, "")))) + 6
            if w > max_w:
                max_w = w
        max_w = max(18, min(max_w, 100))
        widths.append(max_w)
    total_w = sum(widths)
    if total_w > page_w:
        scale = page_w / total_w
        widths = [w * scale for w in widths]
    else:
        extra = page_w - total_w
        if extra > 0 and len(widths) > 0:
            bump = extra / len(widths)
            widths = [w + bump for w in widths]
    return widths

def _draw_header_row(pdf, columns, widths, font_size, brand_rgb):
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_fill_color(*lighten(brand_rgb, 0.85))
    pdf.set_text_color(20, 20, 20)
    row_h = 7
    y0 = pdf.get_y()
    for w, col in zip(widths, columns):
        x = pdf.get_x()
        pdf.multi_cell(w, row_h, to_latin1(str(col)), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_xy(x + w, y0)
    pdf.ln(row_h)
    return row_h

def _ensure_page_space(pdf, needed_h, columns, widths, font_size, brand_rgb):
    if pdf.get_y() + needed_h <= pdf.h - pdf.b_margin:
        return
    pdf.add_page()
    _draw_header_row(pdf, columns, widths, font_size, brand_rgb)

def _calc_row_height_exact(pdf, values, widths, row_h, wrap_idx_set):
    heights = []
    for idx, (w, v) in enumerate(zip(widths, values)):
        txt = "" if v is None else to_latin1(str(v))
        if wrap_idx_set and (idx in wrap_idx_set):
            try:
                lines = pdf.multi_cell(w, row_h, txt, new_x="RIGHT", new_y="TOP", split_only=True)
                n = max(1, len(lines))
            except Exception:
                n = 1
        else:
            n = 1
        heights.append(n * row_h)
    return max(heights) if heights else row_h

def _draw_row(pdf, values, widths, row_h, align_map=None, fill_rgb=None, bold=False, text_rgb=(15,15,15), wrap_idx_set=None, border="1"):
    if align_map is None:
        align_map = {}
    if wrap_idx_set is None:
        wrap_idx_set = set()
    max_h = _calc_row_height_exact(pdf, values, widths, row_h, wrap_idx_set)
    pdf.set_font("Helvetica", "B" if bold else "", 9)
    if fill_rgb:
        pdf.set_fill_color(*fill_rgb)
    pdf.set_text_color(*text_rgb)
    y_start = pdf.get_y()
    x_left = pdf.get_x()
    for idx, (w, v) in enumerate(zip(widths, values)):
        x = pdf.get_x()
        align = align_map.get(idx, "L")
        txt = "" if v is None else to_latin1(str(v))
        if fill_rgb:
            pdf.set_fill_color(*fill_rgb)
        pdf.multi_cell(w, row_h, txt, border=border, align=align, new_x="RIGHT", new_y="TOP", fill=bool(fill_rgb))
        pdf.set_xy(x + w, y_start)
    pdf.set_xy(x_left, y_start + max_h)


# ---------- Inventory PDF (grouped; unchanged) ----------
def _inventory_pdf_bytes_grouped(
    df: pd.DataFrame,
    brand_name, brand_rgb, logo_path, revision_tag,
    prepared_by: str, checked_by: str, approved_by: str,
    only_low: bool, sort_by: str, categories=None, only_available: bool=False
) -> bytes:
    df = df.copy()
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
    df["min_qty"] = pd.to_numeric(df.get("min_qty"), errors="coerce").fillna(0.0)
    df["unit_cost"] = pd.to_numeric(df.get("unit_cost"), errors="coerce").fillna(0.0)
    df["value"] = (df["quantity"] * df["unit_cost"]).round(2)

    if "convert_to" in df.columns and "convert_factor" in df.columns:
        df["convert_factor"] = pd.to_numeric(df.get("convert_factor"), errors="coerce").fillna(0.0)
        df["converted_qty"] = (df["quantity"] * df["convert_factor"]).round(3)
    else:
        df["converted_qty"] = None

    if categories and "category" in df.columns:
        df = df[df["category"].isin(categories)]
    if only_available:
        df = df[df["quantity"] > 0]
    if only_low:
        df = df[df["quantity"] < df["min_qty"]]

    if sort_by in {"sku", "name", "category"} and sort_by in df.columns:
        df = df.sort_values(by=[sort_by, "sku", "name"], kind="stable")

    key_cols = [
        "sku", "name", "category", "location", "unit",
        "quantity", "min_qty", "unit_cost", "value",
        "convert_to", "converted_qty", "notes", "updated_at"
    ]
    present = [c for c in key_cols if c in df.columns]
    if not present:
        present = list(df.columns)
    df = df[present]

    col_names = {
        "sku": "SKU", "name": "Name", "category": "Category", "location": "Location",
        "unit": "Unit", "quantity": "Qty", "min_qty": "Min",
        "unit_cost": "Unit Cost (R)", "value": "Value (R)",
        "convert_to": "Conv Unit", "converted_qty": "Conv Qty",
        "notes": "Notes", "updated_at": "Updated"
    }
    display_cols = [col_names.get(c, c) for c in present]
    def fmt_num(x, places=2): return f"{x:,.{places}f}"

    rows_for_width = []
    for _, r in df.iterrows():
        row = {}
        for c in present:
            v = r[c]
            if c in ("quantity", "min_qty"):
                v = fmt_num(float(v), 2)
            elif c in ("unit_cost", "value"):
                v = fmt_num(float(v), 2)
            elif c == "converted_qty" and pd.notna(v):
                v = fmt_num(float(v), 3)
            row[col_names.get(c, c)] = "" if v is None else str(v)
        rows_for_width.append(row)

    pdf = BrandedPDF(
        brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path, revision_tag=revision_tag,
        orientation="L", unit="mm", format="A3"
    )
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, to_latin1("Inventory Report"), ln=1)
    pdf.set_font("Helvetica", "", 10)

    meta = [f"Rows: {len(df)}"]
    if categories:
        meta.append("Categories: " + ", ".join([to_latin1(c) for c in categories]))
    if only_available:
        meta.append("Only available (>0 qty)")
    if only_low:
        meta.append("Low-stock only")
    if prepared_by:
        meta.append(f"Prepared by: {prepared_by}")
    pdf.cell(0, 6, to_latin1(" | ".join(meta)), ln=1)
    pdf.ln(2)

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    widths = _compute_col_widths(pdf, display_cols, rows_for_width, page_w, font_size=9)
    _draw_header_row(pdf, display_cols, widths, 9, brand_rgb)

    align_map = {}
    for c in ("quantity", "min_qty", "unit_cost", "value", "converted_qty"):
        if c in present:
            align_map[present.index(c)] = "R"
    wrap_idx_set = set()
    if "notes" in present:
        wrap_idx_set.add(display_cols.index("Notes"))

    if "category" in df.columns:
        df_sorted = df.sort_values(by=["category", "sku", "name"], kind="stable")
        categories_iter = df_sorted["category"].fillna("(Unspecified)").unique().tolist()
    else:
        df_sorted = df.copy()
        df_sorted["category"] = "(All)"
        categories_iter = ["(All)"]

    light_brand = lighten(brand_rgb, 0.92)
    cat_bar = lighten(brand_rgb, 0.80)
    low_stock_fill = (255, 235, 235)

    grand_qty = 0.0
    grand_value = 0.0

    for cat in categories_iter:
        block = df_sorted[df_sorted["category"].fillna("(Unspecified)") == cat]

        span_h = 7
        _ensure_page_space(pdf, span_h + 7, display_cols, widths, 9, brand_rgb)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(*cat_bar)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(sum(widths), span_h, to_latin1(f"Category: {cat}"), border=1, ln=1, fill=True)

        cat_qty = float(block["quantity"].sum())
        cat_value = float((block["quantity"] * block["unit_cost"]).sum())
        grand_qty += cat_qty
        grand_value += cat_value

        for _, r in block.iterrows():
            vals = []
            for c in present:
                v = r[c]
                if c in ("quantity", "min_qty"):
                    v = f"{float(v):,.2f}"
                elif c in ("unit_cost", "value"):
                    v = f"{float(v):,.2f}"
                elif c == "converted_qty" and pd.notna(v):
                    v = f"{float(v):,.3f}"
                vals.append(v if v is not None else "")
            fill = low_stock_fill if float(r["quantity"]) < float(r["min_qty"]) else None
            _ensure_page_space(pdf, 9, display_cols, widths, 9, brand_rgb)
            _draw_row(pdf, vals, widths, 7, align_map=align_map, fill_rgb=fill, wrap_idx_set=set())

        sub_vals = []
        for c in present:
            if c == "sku":
                sub_vals.append("Subtotal")
            elif c == "quantity":
                sub_vals.append(f"{cat_qty:,.2f}")
            elif c == "value":
                sub_vals.append(f"{cat_value:,.2f}")
            else:
                sub_vals.append("")
        _ensure_page_space(pdf, 7, display_cols, widths, 9, brand_rgb)
        _draw_row(pdf, sub_vals, widths, 7, align_map=align_map, fill_rgb=light_brand, bold=True, wrap_idx_set=set())

    total_vals = []
    for c in present:
        if c == "sku":
            total_vals.append("TOTAL")
        elif c == "quantity":
            total_vals.append(f"{grand_qty:,.2f}")
        elif c == "value":
            total_vals.append(f"{grand_value:,.2f}")
        else:
            total_vals.append("")
    _ensure_page_space(pdf, 8, display_cols, widths, 9, brand_rgb)
    _draw_row(pdf, total_vals, widths, 8, align_map=align_map, fill_rgb=lighten(brand_rgb, 0.88), bold=True, wrap_idx_set=set())

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, to_latin1("Sign-off"), ln=1)
    pdf.set_font("Helvetica", "", 9)
    w = (pdf.w - pdf.l_margin - pdf.r_margin)
    colw = w / 3.0
    y0 = pdf.get_y()
    for i, label in enumerate(["Prepared by", "Checked by", "Approved by"]):
        x = pdf.l_margin + i * colw
        pdf.set_xy(x, y0 + 10)
        pdf.line(x, y0 + 9, x + colw - 6, y0 + 9)
        name = [prepared_by, checked_by, approved_by][i]
        pdf.set_xy(x, y0 + 11)
        pdf.cell(colw - 6, 5, to_latin1(name or ""), ln=0)
        pdf.set_xy(x, y0)
        pdf.cell(colw - 6, 5, to_latin1(label), ln=0)

    data = pdf.output(dest="S")
    return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")


# ---------- Issue Sheet PDF (A4) + Return Sheet section ----------
def _issue_sheet_pdf_bytes(
    df: pd.DataFrame,
    brand_name, brand_rgb, logo_path, revision_tag,
    manager: str, project: str, notes: str,
    categories=None, only_available: bool=True,
    blanks_cap: int = 12, fixed_blanks: int | None = None,
    include_returns: bool = False,
    return_blanks_cap: int = 12, fixed_return_blanks: int | None = None
) -> bytes:
    """
    Issue section: summary row per item, then ruled blank rows for write-ins.
    Return section (optional): per item descriptor row, then ruled blank rows with return-specific columns.
    """
    df = df.copy()
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
    df["min_qty"] = pd.to_numeric(df.get("min_qty"), errors="coerce").fillna(0.0)

    if categories and "category" in df.columns:
        df = df[df["category"].isin(categories)]
    if only_available:
        df = df[df["quantity"] > 0]

    present = ["sku","name","unit","quantity","min_qty"]
    col_map = {"sku":"SKU","name":"Item","unit":"Unit","quantity":"On-hand","min_qty":"Min"}
    display_cols = [col_map[c] for c in present] + ["Qty Issued", "To (Person)", "Signature", "Date"]

    rows_for_width = []
    for _, r in df.iterrows():
        rows_for_width.append({
            "SKU": str(r.get("sku","")),
            "Item": str(r.get("name","")),
            "Unit": str(r.get("unit","") or ""),
            "On-hand": f"{float(r.get('quantity') or 0):,.2f}",
            "Min": f"{float(r.get('min_qty') or 0):,.2f}",
            "Qty Issued": "",
            "To (Person)": "",
            "Signature": "",
            "Date": "",
        })

    pdf = BrandedPDF(
        brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path, revision_tag=revision_tag,
        orientation="P", unit="mm", format="A4"
    )
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)

    def draw_ruled_blank_row(pdf, widths, row_h=7, line_rgb=(170,170,170)):
        y = pdf.get_y()
        x = pdf.l_margin
        pdf.set_x(x)
        pdf.set_draw_color(*line_rgb)
        pdf.set_line_width(0.2)
        for w in widths:
            pdf.rect(x, y, w, row_h)
            x += w
        pdf.set_y(y + row_h)

    # ---- Header / Meta
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, to_latin1("Stock Issue Sheet"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    meta = [
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Manager: {manager or '-'}",
        f"Project/Job: {project or '-'}",
    ]
    pdf.cell(0, 6, to_latin1(" | ".join(meta)), ln=1)
    if notes:
        pdf.multi_cell(0, 6, to_latin1(f"Notes: {notes}"), ln=1)
    pdf.ln(2)

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    widths = _compute_col_widths(pdf, display_cols, rows_for_width, page_w, font_size=9)

    for i, name in enumerate(display_cols):
        if name in {"Qty Issued"}:            widths[i] = max(widths[i], 20)
        if name in {"To (Person)"}:           widths[i] = max(widths[i], 35)
        if name in {"Signature"}:             widths[i] = max(widths[i], 32)
        if name in {"Date"}:                  widths[i] = max(widths[i], 22)

    _draw_header_row(pdf, display_cols, widths, 9, brand_rgb)
    align_map = {display_cols.index("On-hand"): "R", display_cols.index("Min"): "R"}
    cat_bar = lighten(brand_rgb, 0.80)

    if "category" in df.columns:
        df = df.sort_values(by=["category","sku","name"], kind="stable")
        groups = df["category"].fillna("(Unspecified)").unique().tolist()
    else:
        groups = ["(All)"]
        df["category"] = "(All)"

    # ---- Issue rows
    for cat in groups:
        block = df[df["category"].fillna("(Unspecified)") == cat]
        if block.empty:
            continue

        _ensure_page_space(pdf, 8, display_cols, widths, 9, brand_rgb)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(*cat_bar)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(sum(widths), 7, to_latin1(f"Category: {cat}"), border=1, ln=1, fill=True)

        pdf.set_font("Helvetica", "", 9)
        for _, r in block.iterrows():
            onhand = float(r.get("quantity") or 0.0)
            minq   = float(r.get("min_qty") or 0.0)

            values = [
                r.get("sku",""),
                r.get("name",""),
                r.get("unit","") or "",
                f"{onhand:,.2f}",
                f"{minq:,.2f}",
                "", "", "", ""
            ]
            _ensure_page_space(pdf, 8, display_cols, widths, 9, brand_rgb)
            _draw_row(pdf, values, widths, 7, align_map=align_map, border="1")

            if fixed_blanks is not None and fixed_blanks > 0:
                blanks = fixed_blanks
            else:
                blanks = min(max(0, int(math.floor(onhand))) + 1, max(1, int(blanks_cap)))

            for _ in range(blanks):
                _ensure_page_space(pdf, 7, display_cols, widths, 9, brand_rgb)
                draw_ruled_blank_row(pdf, widths, row_h=7, line_rgb=(170,170,170))

    # ---- Return section (optional)
    if include_returns:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 9, to_latin1("Stock Return Sheet"), ln=1)
        pdf.set_font("Helvetica", "", 10)
        meta2 = [
            f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Manager: {manager or '-'}",
            f"Project/Job: {project or '-'}",
        ]
        pdf.cell(0, 6, to_latin1(" | ".join(meta2)), ln=1)
        pdf.ln(2)

        ret_cols = ["SKU","Item","Unit","Qty Returned","From (Person)","Signature","Date","Condition / Notes"]
        ret_rows_for_width = [{"SKU":"","Item":"","Unit":"","Qty Returned":"","From (Person)":"","Signature":"","Date":"","Condition / Notes":""}]
        ret_widths = _compute_col_widths(pdf, ret_cols, ret_rows_for_width, page_w, font_size=9)

        # widen writing columns
        for i, name in enumerate(ret_cols):
            if name in {"Qty Returned"}:           ret_widths[i] = max(ret_widths[i], 24)
            if name in {"From (Person)"}:          ret_widths[i] = max(ret_widths[i], 38)
            if name in {"Signature"}:              ret_widths[i] = max(ret_widths[i], 32)
            if name in {"Date"}:                   ret_widths[i] = max(ret_widths[i], 22)
            if name in {"Condition / Notes"}:      ret_widths[i] = max(ret_widths[i], 48)

        _draw_header_row(pdf, ret_cols, ret_widths, 9, brand_rgb)

        for cat in groups:
            block = df[df["category"].fillna("(Unspecified)") == cat]
            if block.empty:
                continue

            _ensure_page_space(pdf, 8, ret_cols, ret_widths, 9, brand_rgb)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(*cat_bar)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(sum(ret_widths), 7, to_latin1(f"Category: {cat}"), border=1, ln=1, fill=True)

            pdf.set_font("Helvetica", "", 9)
            for _, r in block.iterrows():
                # descriptor row (SKU/Item/Unit shown; rest blank)
                desc_vals = [r.get("sku",""), r.get("name",""), r.get("unit","") or "", "", "", "", "", ""]
                _ensure_page_space(pdf, 8, ret_cols, ret_widths, 9, brand_rgb)
                _draw_row(pdf, desc_vals, ret_widths, 7, border="1")

                if fixed_return_blanks is not None and fixed_return_blanks > 0:
                    r_blanks = fixed_return_blanks
                else:
                    onhand = float(r.get("quantity") or 0.0)
                    r_blanks = min(max(0, int(math.floor(onhand))) + 1, max(1, int(return_blanks_cap)))

                for _ in range(r_blanks):
                    _ensure_page_space(pdf, 7, ret_cols, ret_widths, 9, brand_rgb)
                    draw_ruled_blank_row(pdf, ret_widths, row_h=7, line_rgb=(170,170,170))

        # footer lines
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 9)
        w = (pdf.w - pdf.l_margin - pdf.r_margin)
        colw = w / 2.0
        y0 = pdf.get_y()
        pdf.set_xy(pdf.l_margin, y0 + 10)
        pdf.line(pdf.l_margin, y0 + 9, pdf.l_margin + colw - 6, y0 + 9)
        pdf.set_xy(pdf.l_margin, y0)
        pdf.cell(colw - 6, 5, to_latin1("Received by (Stores)"), ln=0)
        pdf.set_xy(pdf.l_margin + colw, y0 + 10)
        pdf.line(pdf.l_margin + colw, y0 + 9, pdf.l_margin + w - 6, y0 + 9)
        pdf.set_xy(pdf.l_margin + colw, y0)
        pdf.cell(colw - 6, 5, to_latin1("Returned by (Workshop)"), ln=0)

    data = pdf.output(dest="S")
    return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")


# ---------- Views ----------
def view_dashboard():
    st.title("🏠 Dashboard")
    st.caption("Quick overview + low-stock email helper.")

    items = get_items()
    df = pd.DataFrame(items)
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    col1, col2, col3, col4 = st.columns(4)
    total_items = len(items)
    total_qty = sum((i.get("quantity") or 0) for i in items)
    low_stock = sum(1 for i in items if (i.get("min_qty") or 0) > (i.get("quantity") or 0))
    total_value = sum((i.get("quantity") or 0) * (i.get("unit_cost") or 0) for i in items)

    col1.metric("Distinct SKUs", total_items)
    col2.metric("Total Quantity", f"{total_qty:.2f}")
    col3.metric("Low-Stock Items", low_stock)
    col4.metric("Stock Value", f"R {total_value:,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Low-stock items")
        if df.empty:
            st.info("No items yet.")
        else:
            low_df = df[df["quantity"] < df["min_qty"]].copy()
            if low_df.empty:
                st.success("Nothing low on stock 🎉")
            else:
                st.dataframe(low_df[["sku", "name", "category", "location", "quantity", "min_qty"]],
                             use_container_width=True, height=260)

    with c2:
        st.subheader("Category totals")
        if df.empty:
            st.info("No data.")
        else:
            qty = df.groupby(df["category"].fillna("(Unspecified)"))["quantity"].sum().rename("qty")
            val = (df["quantity"] * df["unit_cost"]).groupby(df["category"].fillna("(Unspecified)")).sum().rename("value")
            cg = pd.concat([qty, val], axis=1)
            st.dataframe(cg, use_container_width=True, height=260)

    st.divider()
    st.subheader("Low-stock email (quick draft)")
    if df.empty:
        st.info("Add inventory first.")
    else:
        low = df[df["quantity"] < df["min_qty"]].copy()
        if low.empty:
            st.info("No low stock to email.")
        else:
            body_lines = ["Low-stock report:", ""]
            for _, r in low.iterrows():
                body_lines.append(f"- {r.get('sku')} | {r.get('name')} | Qty {r.get('quantity')} < Min {r.get('min_qty')}")
            body = "\n".join(body_lines)
            encoded_subject = urllib.parse.quote("Low-Stock Alert")
            encoded_body = urllib.parse.quote(body)
            mailto = f"mailto:{email_recipients}?subject={encoded_subject}&body={encoded_body}"
            st.text_area("Email body (copy/paste)", value=body, height=120)
            st.markdown(f"[Open email draft]({mailto})")


def view_inventory():
    st.title("📦 Inventory")
    st.caption("Add, edit, or delete items. Quick Edit helps fix mistakes (e.g., wrong category).")

    items = get_items()
    df = pd.DataFrame(items)
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    filt = st.text_input("Filter (SKU/Name/Category/Location contains…)")
    fdf = df.copy()
    if filt:
        f = filt.lower()
        mask = (
            fdf["sku"].astype(str).str.lower().str.contains(f) |
            fdf["name"].astype(str).str.lower().str.contains(f) |
            fdf["category"].astype(str).str.lower().str.contains(f) |
            fdf["location"].astype(str).str.lower().str.contains(f)
        )
        fdf = fdf[mask]

    with st.form("add_item"):
        cols = st.columns(4)
        sku = cols[0].text_input("SKU *")
        name = cols[1].text_input("Name *")
        category = cols[2].text_input("Category")
        location = cols[3].text_input("Location")

        cols2 = st.columns(5)
        unit = cols2[0].text_input("Unit (e.g., pcs, m, kg)")
        quantity = cols2[1].number_input("Quantity", value=0.0, step=1.0, format="%.3f")
        min_qty = cols2[2].number_input("Min Qty (alert level)", value=0.0, step=1.0, format="%.3f")
        unit_cost = cols2[3].number_input("Unit Cost (R)", value=0.0, step=1.0, format="%.2f")
        notes = cols2[4].text_input("Notes")

        cols3 = st.columns(2)
        convert_to = cols3[0].text_input("Convert to (optional, e.g., m², m)")
        convert_factor = cols3[1].number_input("Conversion factor (base→convert)", value=0.0, step=0.001, format="%.3f")

        submitted = st.form_submit_button("Save Item")
        if submitted:
            if sku and name:
                add_or_update_item({
                    "sku": sku.strip(),
                    "name": name.strip(),
                    "category": normalize_category(category),
                    "location": location.strip() if location else None,
                    "unit": unit.strip() if unit else None,
                    "quantity": float(quantity),
                    "min_qty": float(min_qty),
                    "unit_cost": float(unit_cost),
                    "notes": notes.strip() if notes else None,
                    "convert_to": convert_to.strip() if convert_to else None,
                    "convert_factor": float(convert_factor or 0.0),
                    "image_path": None,
                })
                st.success(f"Saved item '{sku}'")
                st.rerun()
            else:
                st.error("SKU and Name are required.")

    st.subheader("Inventory List (editable)")
    if not fdf.empty:
        show_cols = ["sku","name","category","location","unit","quantity","min_qty","unit_cost",
                     "convert_to","convert_factor","notes","updated_at"]
        show_cols = [c for c in show_cols if c in fdf.columns]
        edited = st.data_editor(
            fdf[show_cols],
            use_container_width=True,
            num_rows="dynamic",
            key="inv_editor",
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.3f"),
                "min_qty": st.column_config.NumberColumn(format="%.3f"),
                "unit_cost": st.column_config.NumberColumn(format="%.2f"),
                "convert_factor": st.column_config.NumberColumn(format="%.3f"),
            }
        )
        if st.button("Save Edits (Upsert visible rows)"):
            for _, r in edited.iterrows():
                add_or_update_item({
                    "sku": r.get("sku"),
                    "name": r.get("name"),
                    "category": normalize_category(r.get("category")),
                    "location": r.get("location"),
                    "unit": r.get("unit"),
                    "quantity": float(r.get("quantity") or 0),
                    "min_qty": float(r.get("min_qty") or 0),
                    "unit_cost": float(r.get("unit_cost") or 0),
                    "notes": r.get("notes"),
                    "convert_to": r.get("convert_to"),
                    "convert_factor": float(r.get("convert_factor") or 0),
                    "image_path": None,
                })
            st.success("Edits saved.")
            st.rerun()
    else:
        st.info("No items yet. Add your first item above.")

    st.divider()
    st.subheader("Quick Edit (fix a captured mistake)")
    if not df.empty:
        sku_pick = st.selectbox("Select SKU to edit", options=sorted(df["sku"].tolist()))
        rec = df[df["sku"] == sku_pick].iloc[0]
        with st.form("quick_edit"):
            cols = st.columns(4)
            q_name = cols[0].text_input("Name", value=rec.get("name",""))
            q_cat  = cols[1].text_input("Category", value=str(rec.get("category") or ""))
            q_loc  = cols[2].text_input("Location", value=str(rec.get("location") or ""))
            q_unit = cols[3].text_input("Unit", value=str(rec.get("unit") or ""))

            cols2 = st.columns(4)
            q_qty  = cols2[0].number_input("Quantity", value=float(rec.get("quantity") or 0.0), step=1.0, format="%.3f")
            q_min  = cols2[1].number_input("Min Qty", value=float(rec.get("min_qty") or 0.0), step=1.0, format="%.3f")
            q_cost = cols2[2].number_input("Unit Cost (R)", value=float(rec.get("unit_cost") or 0.0), step=1.0, format="%.2f")
            q_notes= cols2[3].text_input("Notes", value=str(rec.get("notes") or ""))

            cols3 = st.columns(2)
            q_conv_to = cols3[0].text_input("Convert to", value=str(rec.get("convert_to") or ""))
            q_conv_fac= cols3[1].number_input("Conversion factor", value=float(rec.get("convert_factor") or 0.0), step=0.001, format="%.3f")

            submit = st.form_submit_button("Apply Changes")
            if submit:
                add_or_update_item({
                    "sku": sku_pick,
                    "name": q_name,
                    "category": normalize_category(q_cat),
                    "location": q_loc or None,
                    "unit": q_unit or None,
                    "quantity": float(q_qty),
                    "min_qty": float(q_min),
                    "unit_cost": float(q_cost),
                    "notes": q_notes or None,
                    "convert_to": q_conv_to or None,
                    "convert_factor": float(q_conv_fac or 0.0),
                    "image_path": None,
                })
                st.success("Item updated.")
                st.rerun()

    st.divider()
    colA, colB = st.columns(2)
    to_delete = colA.text_input("Delete by SKU")
    if colB.button("Delete Item"):
        if to_delete:
            delete_item(to_delete.strip())
            st.success(f"Deleted '{to_delete}'")
            st.rerun()
        else:
            st.error("Enter a SKU to delete.")


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
                st.rerun()
            else:
                st.error("SKU and non-zero quantity are required.")

    st.subheader("Transaction Log (filtered)")
    tx = get_transactions(limit=100_000)
    df_tx = pd.DataFrame(tx)
    if df_tx.empty:
        st.info("No transactions yet.")
        return

    c1, c2, c3 = st.columns(3)
    sku_filter = c1.multiselect("SKU filter", sorted(df_tx["sku"].unique().tolist()))
    reason_filter = c2.multiselect("Reason filter", sorted(df_tx["reason"].unique().tolist()))
    search = c3.text_input("Text search")

    f = df_tx.copy()
    if sku_filter:
        f = f[f["sku"].isin(sku_filter)]
    if reason_filter:
        f = f[f["reason"].isin(reason_filter)]
    if search:
        s = search.lower()
        f = f[
            f["sku"].astype(str).str.lower().str.contains(s) |
            f["project"].astype(str).str.lower().str.contains(s) |
            f["reference"].astype(str).str.lower().str.contains(s) |
            f["user"].astype(str).str.lower().str.contains(s) |
            f["notes"].astype(str).str.lower().str.contains(s)
        ]
    st.dataframe(f, use_container_width=True)


def view_issue_sheets():
    st.title("📝 Issue Sheets")
    st.caption("Create a printable Stock Issue Sheet (PPE & consumables) with ruled blank sign-off rows, and (optional) Return Sheet.")

    if not FPDF_AVAILABLE:
        st.error("PDF engine not available. Add `fpdf2==2.7.9` to requirements.txt.")
        return

    items = get_items()
    df = pd.DataFrame(items)
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    if df.empty:
        st.info("No inventory yet. Add items first.")
        return

    all_cats = sorted([c for c in df.get("category", pd.Series(dtype=str)).dropna().unique().tolist()])

    def looks_consumable(x: str) -> bool:
        s = x.lower()
        keys = ["disc", "cutting", "grinding", "weld", "wire", "ppe", "glove", "mask", "goggle", "paint", "oxygen", "gas"]
        return any(k in s for k in keys)

    default_cats = [c for c in all_cats if looks_consumable(c)] or all_cats
    cat_select = st.multiselect("Categories to include", options=all_cats, default=default_cats)

    c1, c2, c3 = st.columns(3)
    only_avail = c1.checkbox("Only show items with qty > 0", value=True)
    mgr = c2.text_input("Manager (issued by)")
    project = c3.text_input("Project / Job")
    notes = st.text_input("Notes (optional)")

    with st.expander("Sheet layout options", expanded=False):
        cap = st.slider("Cap blank lines per item (Issue sheet; On-hand + 1, capped)", min_value=1, max_value=40, value=12)
        fixed_n = st.number_input("Or use a fixed number per item (Issue; 0 = disabled)", min_value=0, max_value=50, value=0)
        fixed_blanks = fixed_n if fixed_n > 0 else None

        st.markdown("---")
        include_returns = st.checkbox("Include Return Sheet section", value=True)
        ret_cap = st.slider("Cap blank lines per item (Return sheet; On-hand + 1, capped)", min_value=1, max_value=40, value=12)
        ret_fixed_n = st.number_input("Or use a fixed number per item (Return; 0 = disabled)", min_value=0, max_value=50, value=0)
        fixed_return_blanks = ret_fixed_n if ret_fixed_n > 0 else None

    build = st.button("Generate Issue PDF (with optional Return section)")
    if build:
        pdf_bytes = _issue_sheet_pdf_bytes(
            df, brand_name, brand_rgb, logo_path, revision_tag,
            manager=mgr, project=project, notes=notes,
            categories=cat_select if cat_select else None,
            only_available=only_avail,
            blanks_cap=cap, fixed_blanks=fixed_blanks,
            include_returns=include_returns,
            return_blanks_cap=ret_cap, fixed_return_blanks=fixed_return_blanks
        )
        st.download_button(
            "Download Issue/Return PDF",
            data=pdf_bytes,
            file_name=f"Issue_Return_{timestamp()}.pdf",
            mime="application/pdf",
        )

    st.divider()
    st.subheader("Quick Issue (log immediately)")
    filt_df = df[df["category"].isin(cat_select)] if cat_select else df
    sku_list = sorted(filt_df["sku"].unique().tolist())

    with st.form("quick_issue_form"):
        cols = st.columns(4)
        q_sku = cols[0].selectbox("SKU", options=sku_list)
        q_qty = cols[1].number_input("Qty issued", value=1.0, min_value=0.0, step=1.0, format="%.3f")
        q_to  = cols[2].text_input("To (person)")
        q_proj= cols[3].text_input("Project/Job", value=project)

        notes2 = st.text_input("Notes")
        submit = st.form_submit_button("Log Issue")
        if submit:
            if not q_sku or q_qty <= 0:
                st.error("Choose a SKU and a positive quantity.")
            else:
                add_transaction(q_sku, -abs(q_qty), reason="issue", project=q_proj, reference="", user=q_to, notes=notes2)
                st.success(f"Issue logged for {q_sku} → {q_to} (-{q_qty})")
                st.rerun()


# ---------- Versions / Snapshots ----------
def _zip_list():
    return sorted(glob.glob(os.path.join(SNAP_DIR, "*.zip")), reverse=True)

def _find_latest_for_site(site: str):
    pats = [
        os.path.join(SNAP_DIR, f"*{site}*.zip"),
        os.path.join(SNAP_DIR, f"*{site.replace(' ', '_')}*.zip"),
    ]
    files = []
    for p in pats:
        files += glob.glob(p)
    files = sorted(files, reverse=True)
    return files[0] if files else None

def view_versions():
    st.title("🕒 Versions & Snapshots")
    st.caption("Create timestamped ZIP archives of your data for traceability. Also supports restore and site profiles.")

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

    st.divider()
    st.subheader("Site Profiles (snapshot-based)")
    cols = st.columns(3)
    site = cols[0].text_input("Site name", value="Main Workshop")
    if cols[1].button("Create snapshot for site"):
        items = get_items()
        tx = get_transactions(limit=1_000_000)
        site_tag = f"{site.replace(' ', '_')}_{timestamp()}"
        zip_path = export_snapshot(items, tx, tag=site_tag, note=f"Site: {site}")
        save_version_record(site_tag, f"Site {site}", zip_path)
        st.success(f"Snapshot for '{site}' saved: {os.path.basename(zip_path)}")
    if cols[2].button("Restore latest for site"):
        latest = _find_latest_for_site(site)
        if not latest:
            st.error("No snapshot found for that site name.")
        else:
            try:
                with open(latest, "rb") as f:
                    data = f.read()
                _restore_from_bytes(data, replace_items=True, append_tx=True, skip_dup=True, preserve_ts=True)
            except Exception as e:
                st.error(f"Restore failed: {e}")

    st.divider()
    st.subheader("Restore from Snapshot ZIP")
    st.caption("Rebuild DB from a saved ZIP (e.g., after rebuild or when switching sites).")
    up = st.file_uploader("Upload snapshot ZIP", type=["zip"])
    colr1, colr2, colr3, colr4 = st.columns(4)
    replace_items = colr1.checkbox("Replace items (clear & load)", value=True)
    append_tx = colr2.checkbox("Append transactions", value=False)
    skip_dup = colr3.checkbox("Skip duplicate transactions", value=True)
    preserve_ts = colr4.checkbox("Try to keep original tx timestamps", value=True)

    if up and st.button("Restore now"):
        try:
            _restore_from_bytes(up.read(), replace_items, append_tx, skip_dup, preserve_ts)
        except Exception as e:
            st.error(f"Restore failed: {e}")

    st.subheader("History")
    versions = get_versions()
    if versions:
        st.dataframe(pd.DataFrame(versions), use_container_width=True, height=300)
    else:
        st.info("No versions yet.")
    st.caption("Snapshot files in /snapshots:")
    st.write([os.path.basename(p) for p in _zip_list()])


def _restore_from_bytes(zip_bytes: bytes, replace_items: bool, append_tx: bool, skip_dup: bool, preserve_ts: bool):
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    all_names = zf.namelist()

    def find_csv(candidates):
        for n in all_names:
            ln = n.lower()
            if ln.endswith(".csv") and any(k in ln for k in candidates):
                return n
        return None

    inv_name = find_csv(["inventory", "items"])
    tx_name  = find_csv(["transactions", "transaction", "trans"])

    if not inv_name:
        st.error("inventory.csv (or items*.csv) not found in ZIP.\n\nFiles:\n" + "\n".join(all_names))
        return

    inv_df = pd.read_csv(io.BytesIO(zf.read(inv_name)))

    if replace_items:
        existing = get_items()
        for it in existing:
            try:
                delete_item(it["sku"])
            except Exception:
                pass

    added_items = 0
    for _, r in inv_df.iterrows():
        try:
            add_or_update_item({
                "sku": str(r.get("sku") or "").strip(),
                "name": str(r.get("name") or "").strip(),
                "category": normalize_category(r.get("category")),
                "location": (r.get("location") if pd.notna(r.get("location")) else None),
                "unit": (r.get("unit") if pd.notna(r.get("unit")) else None),
                "quantity": float(r.get("quantity") or 0),
                "min_qty": float(r.get("min_qty") or 0),
                "unit_cost": float(r.get("unit_cost") or 0),
                "notes": (r.get("notes") if pd.notna(r.get("notes")) else None),
                "convert_to": (r.get("convert_to") if "convert_to" in inv_df.columns and pd.notna(r.get("convert_to")) else None),
                "convert_factor": float(r.get("convert_factor") or 0) if "convert_factor" in inv_df.columns else 0.0,
                "image_path": None,
            })
            added_items += 1
        except Exception as e:
            st.warning(f"Item restore skipped for SKU={r.get('sku')}: {e}")

    added_tx = 0
    if append_tx and tx_name:
        tx_df = pd.read_csv(io.BytesIO(zf.read(tx_name)))
        existing_tx = get_transactions(limit=1_000_000) if skip_dup else []
        existing_keys = set()
        if skip_dup:
            for t in existing_tx:
                key = (
                    str(t.get("sku") or ""),
                    float(t.get("qty_change") or 0.0),
                    str(t.get("reason") or ""),
                    str(t.get("project") or ""),
                    str(t.get("reference") or ""),
                    str(t.get("user") or ""),
                    str(t.get("notes") or ""),
                    to_latin1(str(t.get("ts") or "")),
                )
                existing_keys.add(key)

        for _, r in tx_df.iterrows():
            key = (
                str(r.get("sku") or ""),
                float(r.get("qty_change") or 0.0),
                str(r.get("reason") or ""),
                str(r.get("project") or ""),
                str(r.get("reference") or ""),
                str(r.get("user") or ""),
                str(r.get("notes") or ""),
                to_latin1(str(r.get("ts") or "")),
            )
            if skip_dup and key in existing_keys:
                continue
            try:
                if preserve_ts:
                    try:
                        add_transaction(key[0], key[1], key[2], key[3], key[4], key[5], key[6], ts=key[7])
                    except TypeError:
                        add_transaction(key[0], key[1], key[2], key[3], key[4], key[5], key[6])
                else:
                    add_transaction(key[0], key[1], key[2], key[3], key[4], key[5], key[6])
                added_tx += 1
            except Exception as e:
                st.warning(f"Transaction restore skipped for SKU={key[0]}: {e}")

    st.success(f"Restore complete. Items loaded: {added_items}" + (f" | Transactions added: {added_tx}" if append_tx and tx_name else ""))


def view_reports():
    st.title("🧾 Reports & Export (PDF)")
    st.caption("A3 landscape • grouped by category • subtotals • totals row • notes • unit conversion • sign-off block")

    if not FPDF_AVAILABLE:
        st.error("PDF engine not available. Add `fpdf2==2.7.9` to requirements.txt.")
        return

    st.subheader("Inventory Report")
    items = get_items()
    df_items = pd.DataFrame(items)
    if "category" in df_items.columns:
        df_items["category"] = df_items["category"].apply(normalize_category)

    categories = sorted([c for c in df_items.get("category", pd.Series(dtype=str)).dropna().unique().tolist()])
    cat_select = st.multiselect("Categories to include", options=categories, default=categories)
    only_available = st.checkbox("Only available items (> 0 qty)", value=False)
    only_low = st.checkbox("Only low-stock rows", value=False)
    sort_by = st.selectbox("Sort by", options=["category","sku","name"], index=0)

    if not df_items.empty:
        use_cats = cat_select if cat_select else None
        pdf_bytes = _inventory_pdf_bytes_grouped(
            df_items, brand_name, brand_rgb, logo_path, revision_tag,
            prepared_by=prepared_by, checked_by=checked_by, approved_by=approved_by,
            only_low=only_low, sort_by=sort_by, categories=use_cats, only_available=only_available
        )
        st.download_button(
            "Download Inventory PDF",
            data=pdf_bytes,
            file_name=f"inventory_{timestamp()}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No items to include in the report.")

    st.subheader("Transaction Log (PDF)")
    tx = get_transactions(limit=100_000)
    df_tx = pd.DataFrame(tx)
    if df_tx.empty:
        st.info("No transactions to include in the report.")
        return

    def df_to_pdf_bytes_pro(title: str, df: pd.DataFrame, meta_lines, brand_name: str, brand_rgb: tuple, logo_path: str, revision_tag: str) -> bytes:
        pdf = BrandedPDF(
            brand_name=brand_name, brand_rgb=brand_rgb, logo_path=logo_path, revision_tag=revision_tag,
            orientation="L", unit="mm", format="A3"
        )
        pdf.add_page()
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, to_latin1(title), ln=1)
        pdf.set_font("Helvetica", "", 10)
        for line in meta_lines:
            pdf.cell(0, 6, to_latin1(str(line)), ln=1)
        pdf.ln(2)

        def draw_table(pdf, df: pd.DataFrame, header_fill_rgb: tuple, font_size=9):
            pdf.set_font("Helvetica", "B", font_size)
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            measure_rows = min(len(df), 200)
            widths = []
            for col in df.columns:
                max_w = pdf.get_string_width(to_latin1(str(col))) + 6
                pdf.set_font("Helvetica", "", font_size)
                for i in range(measure_rows):
                    w = pdf.get_string_width(to_latin1(str(df.iloc[i][col]))) + 6
                    if w > max_w:
                        max_w = w
                max_w = max(18, min(max_w, 100))
                widths.append(max_w)
            total_w = sum(widths)
            if total_w > page_w:
                scale = page_w / total_w
                widths = [w * scale for w in widths]
            else:
                extra = page_w - total_w
                if extra > 0 and len(widths) > 0:
                    bump = extra / len(widths)
                    widths = [w + bump for w in widths]

            pdf.set_font("Helvetica", "B", font_size)
            pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
            pdf.set_text_color(20, 20, 20)
            row_h = 7
            y0 = pdf.get_y()
            for w, col in zip(widths, df.columns):
                x = pdf.get_x()
                pdf.multi_cell(w, row_h, to_latin1(str(col)), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
                pdf.set_xy(x + w, y0)
            pdf.ln(row_h)

            pdf.set_font("Helvetica", "", font_size)
            pdf.set_text_color(15, 15, 15)
            for idx in range(len(df)):
                y_start = pdf.get_y()
                if y_start + row_h > pdf.h - pdf.b_margin:
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", font_size)
                    pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
                    pdf.set_text_color(20, 20, 20)
                    y0 = pdf.get_y()
                    for w, col in zip(widths, df.columns):
                        x = pdf.get_x()
                        pdf.multi_cell(w, row_h, to_latin1(str(col)), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
                        pdf.set_xy(x + w, y0)
                    pdf.ln(row_h)
                    pdf.set_font("Helvetica", "", font_size)
                    pdf.set_text_color(15, 15, 15)
                    y_start = pdf.get_y()

                x_left = pdf.get_x()
                for w, val in zip(widths, df.iloc[idx].tolist()):
                    txt = to_latin1(str(val))
                    x = pdf.get_x()
                    pdf.multi_cell(w, row_h, txt, border=1, align="L", new_x="RIGHT", new_y="TOP")
                    pdf.set_xy(x + w, y_start)
                pdf.set_xy(x_left, y_start + row_h)

        draw_table(pdf, df, brand_rgb, font_size=9)
        data = pdf.output(dest="S")
        return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("latin-1", errors="ignore")

    col_order = ["ts", "sku", "qty_change", "reason", "project", "reference", "user", "notes"]
    col_names = {
        "ts":"Timestamp","sku":"SKU","qty_change":"Δ Qty","reason":"Reason",
        "project":"Project/Job","reference":"Reference","user":"User","notes":"Notes"
    }
    df_tx = df_tx.loc[:, [c for c in col_order if c in df_tx.columns]].rename(columns=col_names)
    meta = [f"Rows: {len(df_tx)}"]
    if prepared_by:
        meta.append(f"Prepared by: {prepared_by}")
    pdf_bytes_tx = df_to_pdf_bytes_pro("Transaction Log", df_tx, meta, brand_name, brand_rgb, logo_path, revision_tag)
    st.download_button("Download Transactions PDF", data=pdf_bytes_tx, file_name=f"transactions_{timestamp()}.pdf", mime="application/pdf")


def view_maintenance():
    st.title("🛠️ Maintenance")
    st.caption("Maintenance usage + Category Manager / cleanup tools.")

    items = get_items()
    if not items:
        st.info("Add items first in the Inventory page.")
    else:
        sku_list = [i["sku"] for i in items]
        with st.form("maint_form"):
            st.subheader("Log maintenance usage")
            cols = st.columns(4)
            sku = cols[0].selectbox("Item SKU", options=sku_list)
            qty_used = cols[1].number_input("Qty Used (negative)", value=-1.0, step=1.0, format="%.3f")
            project = cols[2].text_input("Home/Workshop area")
            user = cols[3].text_input("Person")
            notes = st.text_area("Notes (what/where/why)")

            submitted = st.form_submit_button("Log Maintenance Usage")
            if submitted:
                if not sku or qty_used >= 0:
                    st.error("Choose a SKU and enter a negative quantity to deduct.")
                else:
                    add_transaction(sku, qty_used, reason="maintenance", project=project, reference="", user=user, notes=notes)
                    st.success("Maintenance usage logged (stock deducted).")
                    st.rerun()

    st.divider()
    st.subheader("Category Manager")
    df_all = pd.DataFrame(get_items())
    if "category" in df_all.columns:
        df_all["category"] = df_all["category"].apply(normalize_category)
    distinct = sorted([c for c in df_all.get("category", pd.Series(dtype=str)).dropna().unique().tolist()])
    c1, c2 = st.columns(2)
    old_cat = c1.selectbox("Existing category to rename/merge", options=["—"] + distinct, index=0)
    new_cat = c2.text_input("Replace with (target category, e.g., 'Laser Shop')")
    c3, c4 = st.columns(2)
    if c3.button("Rename/Merge"):
        if old_cat == "—" or not new_cat.strip():
            st.error("Select an existing category and enter a new name.")
        else:
            target = normalize_category(new_cat)
            count = 0
            items = get_items()
            for it in items:
                cur = normalize_category(it.get("category"))
                if cur == old_cat:
                    it["category"] = target
                    add_or_update_item(it)
                    count += 1
            st.success(f"Updated {count} item(s) from '{old_cat}' → '{target}'.")
            st.rerun()
    if c4.button("Normalize categories (trim/collapse Title-Case)"):
        items = get_items()
        fixed = 0
        for it in items:
            cur = it.get("category")
            norm = normalize_category(cur)
            if cur != norm:
                it["category"] = norm
                add_or_update_item(it)
                fixed += 1
        st.success(f"Normalized {fixed} item(s).")
        st.rerun()


# ---------- Router ----------
if menu == "Dashboard":
    view_dashboard()
elif menu == "Inventory":
    view_inventory()
elif menu == "Transactions":
    view_transactions()
elif menu == "Issue Sheets":
    view_issue_sheets()
elif menu == "Versions & Snapshots":
    view_versions()
elif menu == "Reports & Export (PDF)":
    view_reports()
else:
    view_maintenance()
