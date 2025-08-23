# app.py — OMEC Stock Take (single-file) — Safe logo + Professional PDF exports (no CSV)
import os, json, html, math
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
from fpdf import FPDF

st.set_page_config(page_title="OMEC Stock Take", page_icon="🗃️", layout="wide")

# ---------- Config ------------------------------------------------------------
ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OMEC", "brand_color": "#0ea5e9", "logo_path": ""}

init_db()  # ensure SQLite schema exists

# ---------- Helpers -----------------------------------------------------------
def _norm_path(p: str) -> str:
    """Absolute path from project root; '' if invalid."""
    if not isinstance(p, str) or not p.strip():
        return ""
    p = os.path.normpath(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def safe_show_logo(path: str):
    """Show logo if file exists; never crash."""
    try:
        apath = _norm_path(path)
        if apath and os.path.exists(apath):
            st.sidebar.image(apath, use_container_width=True)
    except Exception:
        pass

def hex_to_rgb(h: str):
    """#RRGGBB -> (r,g,b)."""
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (14, 165, 233)  # fallback cyan-ish

def lighten(rgb, factor=0.85):
    """Lighten color toward white by factor (0-1)."""
    r, g, b = rgb
    return (int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))

class BrandedPDF(FPDF):
    def __init__(self, brand_name: str, brand_rgb: tuple, logo_path: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brand_name = brand_name
        self.brand_rgb = brand_rgb
        self.logo_path = _norm_path(logo_path) if logo_path else ""
        self.set_auto_page_break(auto=True, margin=12)

    def header(self):
        # Top brand bar
        self.set_fill_color(*lighten(self.brand_rgb, 0.75))
        self.rect(x=0, y=0, w=self.w, h=14, style="F")

        # Logo (if any)
        x = 10
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.image(self.logo_path, x=x, y=3, h=8)
                x += 26
            except Exception:
                pass

        # Brand name
        self.set_xy(x, 4)
        self.set_text_color(25, 25, 25)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 6, txt=self.brand_name, ln=0)

        # Date on right
        self.set_xy(-80, 4)
        self.set_font("Helvetica", "", 9)
        self.cell(70, 6, txt=f"Generated: {timestamp()}", ln=0, align="R")

        # Second divider line
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
        self.cell(0, 6, f"Page {self.page_no()}", align="R")

def draw_table(pdf: BrandedPDF, df: pd.DataFrame, header_fill_rgb: tuple, col_order=None, col_rename=None, font_size=9):
    """Draw a table across pages with auto column widths and wrapped cells."""
    if col_order is not None:
        df = df.loc[:, [c for c in col_order if c in df.columns]]
    if col_rename:
        df = df.rename(columns=col_rename)

    # Calculate column widths based on content
    pdf.set_font("Helvetica", "B", font_size)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    # max width in mm for each column based on header and first N rows
    measure_rows = min(len(df), 200)
    widths = []
    for col in df.columns:
        max_w = pdf.get_string_width(str(col)) + 4
        pdf.set_font("Helvetica", "", font_size)
        for i in range(measure_rows):
            w = pdf.get_string_width(str(df.iloc[i][col])) + 4
            if w > max_w:
                max_w = w
        # clamp per column
        max_w = max(18, min(max_w, 70))
        widths.append(max_w)

    # scale widths to fit page if necessary
    total_w = sum(widths)
    if total_w > page_w:
        scale = page_w / total_w
        widths = [w * scale for w in widths]
    else:
        # distribute remaining space to wider columns
        extra = page_w - total_w
        if extra > 0 and len(widths) > 0:
            bump = extra / len(widths)
            widths = [w + bump for w in widths]

    # Header row
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_fill_color(*lighten(header_fill_rgb, 0.85))
    pdf.set_text_color(20, 20, 20)
    row_h = 7
    x0 = pdf.get_x()
    y0 = pdf.get_y()
