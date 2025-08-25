# app.py — OMEC Stock Take (Streamlit)
# Update: PDF header shows ONLY the logo on the left (no brand text) and adds "Issued" date-time on the right.
# Rest of the layout/logic unchanged.

import os, json, re, io, zipfile, glob, math, shutil
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
ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

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

# If no logo saved yet, auto-adopt a known on-disk PG Bison logo if available (one-time).
try:
    if not logo_path:
        pg_bison_src = "/mnt/data/PG Bison.jpg"
        if os.path.exists(pg_bison_src):
            dst = os.path.join(ASSETS_DIR, "brand_logo.jpg")
            if not os.path.exists(dst):
                shutil.copyfile(pg_bison_src, dst)
            upsert_setting("logo_path", dst)
            logo_path = dst
except Exception:
    pass


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

# Branding control (logo only)
with st.sidebar.expander("Branding (Logo)", expanded=False):
    st.caption("Upload a PNG/JPG or type a server path, then **Save Logo**.")
    up = st.file_uploader("Upload logo", type=["png", "jpg", "jpeg"])
    col_a, col_b = st.columns([3, 1])
    logo_input = col_a.text_input("Or path on server", value=logo_path or "")
    if col_b.button("Save Logo"):
        try:
            new_path = logo_input.strip()
            if up is not None:
                ext = os.path.splitext(up.name)[1].lower() or ".png"
                if ext not in [".png", ".jpg", ".jpeg"]:
                    ext = ".png"
                new_path = os.path.join(ASSETS_DIR, f"brand_logo{ext}")
                with open(new_path, "wb") as f:
                    f.write(up.read())
            if new_path:
                upsert_setting("logo_path", new_path)
                st.success("Logo saved. Reloading…")
                st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")
    clear_logo = st.button("Clear Logo")
    if clear_logo:
        upsert_setting("logo_path", "")
        st.success("Logo cleared. Reloading…")
        st.rerun()
    if logo_path:
        try:
            st.image(_norm_path(logo_path), caption="Current logo", use_container_width=True)
        except Exception:
            st.caption("Logo path set, but preview failed.")

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
            # Top band
            self.set_fill_color(*lighten(self.brand_rgb, 0.75))
            self.rect(x=0, y=0, w=self.w, h=14, style="F")

            # Left: LOGO ONLY
            x = 10
            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    self.image(self.logo_path, x=x, y=3, h=8)
                except Exception:
                    pass

            # Right: Revision + Issued date-time
            self.set_xy(-100, 4)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(25, 25, 25)
            issued = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.cell(95, 6, txt=to_latin1(f"Revision: {self.revision_tag}   |   Issued: {issued}"), ln=0, align="R")

            # Separator line
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

def _truncate_to_fit(pdf, txt: str, w: float, margin: float = 2.0) -> str:
    """Truncate text with ASCII ellipsis so it fits inside width w (Latin-1 safe)."""
    if txt is None:
        return ""
    s = to_latin1(str(txt))
    maxw = max(1.0, w - margin)
    if pdf.get_string_width(s) <= maxw:
        return s
    ell = "..."
    if pdf.get_string_width(ell) > maxw:
        return ""
    while s and pdf.get_string_width(s + ell) > maxw:
        s = s[:-1]
    return (s + ell) if s else ell

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
        txt = "" if v is None else str(v)

        if idx in wrap_idx_set:
            pdf.multi_cell(w, row_h, to_latin1(txt), border=border, align=align, new_x="RIGHT", new_y="TOP", fill=bool(fill_rgb))
            pdf.set_xy(x + w, y_start)
        else:
            t = _truncate_to_fit(pdf, txt, w)
            pdf.cell(w, max_h, t, align=align, border=border)
    pdf.set_xy(x_left, y_start + max_h)


# ---------- Inventory PDF (grouped; unchanged logic) ----------
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
            _draw_row(pdf, vals, widths, 7, align_map=align_map, fill_rgb=fill, wrap_idx_set=wrap_idx_set)

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


# ---------- Issue Sheet PDF (A4) + Return Sheet uses same blank counts ----------
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
    Return section now reuses EXACTLY the same number of blank lines per item as Issue section.
    Also, non-wrapping cells are drawn as single-line cells (with ellipsis) so text never spills into blank rows.
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

    # widen important handwriting columns; also widen Item to reduce chance of ellipsis
    for i, name in enumerate(display_cols):
        if name in {"Item"}:                 widths[i] = max(widths[i], 55)
        if name in {"Qty Issued"}:           widths[i] = max(widths[i], 20)
        if name in {"To (Person)"}:          widths[i] = max(widths[i], 35)
        if name in {"Signature"}:            widths[i] = max(widths[i], 32)
        if name in {"Date"}:                 widths[i] = max(widths[i], 22)

    _draw_header_row(pdf, display_cols, widths, 9, brand_rgb)
    align_map = {display_cols.index("On-hand"): "R", display_cols.index("Min"): "R"}
    cat_bar = lighten(brand_rgb, 0.80)

    if "category" in df.columns:
        df = df.sort_values(by=["category","sku","name"], kind="stable")
        groups = df["category"].fillna("(Unspecified)").unique().tolist()
    else:
        groups = ["(All)"]
        df["category"] = "(All)"

    # capture per-item blank counts to reuse for Return sheet
    per_item_blanks = []  # list of dicts: {cat, sku, name, unit, blanks}

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
            # wrap_idx_set is empty: draw single-line cells (with truncation)
            _draw_row(pdf, values, widths, 7, align_map=align_map, border="1", wrap_idx_set=set())

            if fixed_blanks is not None and fixed_blanks > 0:
                blanks = fixed_blanks
            else:
                blanks = min(max(0, int(math.floor(onhand))) + 1, max(1, int(blanks_cap)))

            per_item_blanks.append({
                "cat": cat,
                "sku": r.get("sku",""),
                "name": r.get("name",""),
                "unit": r.get("unit","") or "",
                "blanks": blanks
            })

            for _ in range(blanks):
                _ensure_page_space(pdf, 7, display_cols, widths, 9, brand_rgb)
                draw_ruled_blank_row(pdf, widths, row_h=7, line_rgb=(170,170,170))

    # ---- Return section (optional) — uses SAME blank counts per item
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

        for i, name in enumerate(ret_cols):
            if name in {"Item"}:                   ret_widths[i] = max(ret_widths[i], 55)
            if name in {"Qty Returned"}:           ret_widths[i] = max(ret_widths[i], 24)
            if name in {"From (Person)"}:          ret_widths[i] = max(ret_widths[i], 38)
            if name in {"Signature"}:              ret_widths[i] = max(ret_widths[i], 32)
            if name in {"Date"}:                   ret_widths[i] = max(ret_widths[i], 22)
            if name in {"Condition / Notes"}:      ret_widths[i] = max(ret_widths[i], 48)

        _draw_header_row(pdf, ret_cols, ret_widths, 9, brand_rgb)

        # iterate by categories for nice grouping
        for cat in groups:
            items_cat = [x for x in per_item_blanks if x["cat"] == cat]
            if not items_cat:
                continue

            _ensure_page_space(pdf, 8, ret_cols, ret_widths, 9, brand_rgb)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(*cat_bar)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(sum(ret_widths), 7, to_latin1(f"Category: {cat}"), border=1, ln=1, fill=True)

            pdf.set_font("Helvetica", "", 9)
            for it in items_cat:
                # descriptor row (show SKU/Item/Unit; others blank)
                desc_vals = [it["sku"], it["name"], it["unit"], "", "", "", "", ""]
                _ensure_page_space(pdf, 8, ret_cols, ret_widths, 9, brand_rgb)
                _draw_row(pdf, desc_vals, ret_widths, 7, border="1", wrap_idx_set=set())

                r_blanks = it["blanks"] if (fixed_return_blanks is None or fixed_return_blanks <= 0) else fixed_return_blanks
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
        min_qty = cols2[2].number_input("Min
