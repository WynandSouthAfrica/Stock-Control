# app.py — OpperWorks Stock Take (Streamlit)
# === What’s new in this build ===
# - Sidebar header now shows your OpperWorks logo at the very top (replaces the old “OMEC” text).
# - Auto-detects the bundled logo “/mnt/data/OpperWorks Logo.png” on first run and saves it.
# - Branding panel adds a “Use OpperWorks logo” quick button.
# - All previous features preserved: Serial # column, add-as-you-go rows, rename/delete on save,
#   PDF wrapping for Name/Notes, Rand formatting, snapshots, etc.

import os, json, re, io, zipfile, glob, math, shutil, sqlite3
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

st.set_page_config(page_title="OpperWorks Stock Take", page_icon="🗃️", layout="wide")

# ---------- Paths / helpers ----------
ROOT = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def _coerce_path(user_path: str) -> str:
    if not isinstance(user_path, str) or not user_path.strip():
        return ""
    p = user_path.strip()
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m and os.name != "nt":
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return os.path.normpath(f"/mnt/{drive}/{rest}")
    if p.startswith("\\\\") and os.name != "nt":
        return p
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(ROOT, p))

def _norm_path(p: str) -> str:
    return _coerce_path(p)

def safe_show_logo(path: str, height: int | None = None):
    """Best-effort image render (sidebar)."""
    try:
        apath = _norm_path(path)
        if apath and os.path.exists(apath):
            st.sidebar.image(apath, use_container_width=(height is None), clamp=False, output_format="auto")
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

# ---- Rand currency formatting ----
def fmt_rands(x, with_symbol=True) -> str:
    try:
        val = float(x)
    except Exception:
        val = 0.0
    s = f"{val:,.2f}"
    return f"R {s}" if with_symbol else s

def _can_write_here(path_dir: str) -> tuple[bool, str]:
    try:
        os.makedirs(path_dir, exist_ok=True)
        testfile = os.path.join(path_dir, ".write_test.tmp")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return True, os.path.abspath(path_dir)
    except Exception:
        return False, os.path.abspath(path_dir)

def _move_to_snap_dir(src_path: str, snap_dir: str) -> str:
    try:
        if not src_path:
            return src_path
        os.makedirs(snap_dir, exist_ok=True)
        dst_path = os.path.join(snap_dir, os.path.basename(src_path))
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            try:
                shutil.move(src_path, dst_path)
            except Exception:
                shutil.copyfile(src_path, dst_path)
        return dst_path
    except Exception:
        return src_path

# ---------- Init DB / Config ----------
init_db()

CONFIG_PATH = os.path.join(ROOT, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"brand_name": "OpperWorks", "brand_color": "#0ea5e9", "logo_path": "", "revision_tag": "Rev0.1"}

DEFAULT_SNAP_DIR = os.path.join(ROOT, "snapshots")
_saved_snap = get_setting("snap_dir", DEFAULT_SNAP_DIR) or DEFAULT_SNAP_DIR
SNAP_DIR = _coerce_path(_saved_snap)
os.makedirs(SNAP_DIR, exist_ok=True)

# ---------- Settings ----------
logo_path = get_setting("logo_path", CONFIG.get("logo_path", ""))
brand_name = get_setting("brand_name", CONFIG.get("brand_name", "OpperWorks"))
brand_color = get_setting("brand_color", CONFIG.get("brand_color", "#0ea5e9"))
revision_tag = get_setting("revision_tag", CONFIG.get("revision_tag", "Rev0.1"))
prepared_by = get_setting("prepared_by", "")
checked_by  = get_setting("checked_by", "")
approved_by = get_setting("approved_by", "")
email_recipients = get_setting("email_recipients", "")
auto_backup_enabled = str(get_setting("auto_backup_enabled", "true")).lower() in {"1","true","yes","on"}
brand_rgb = hex_to_rgb(brand_color)

# ---------- Serial Number: schema + compatibility layer ----------
def _db_candidates_from_module():
    paths = []
    try:
        import db as _db
        for attr in ("DB_PATH", "DB_FILE", "PATH", "db_path"):
            p = getattr(_db, attr, None)
            if isinstance(p, str):
                paths.append(_coerce_path(p))
    except Exception:
        pass
    for name in ("data.db", "db.sqlite3", "app.db", "stock.db", "inventory.db", "omec.db"):
        paths.append(os.path.join(ROOT, name))
    uniq = []
    for p in paths:
        if p and p not in uniq:
            uniq.append(p)
    return [p for p in uniq if os.path.exists(p)]

def _ensure_serialno_column(show_toast=False) -> bool:
    try:
        sample = get_items() or []
        if sample and "serial_no" in sample[0]:
            return True
    except Exception:
        pass
    for path in _db_candidates_from_module():
        try:
            with sqlite3.connect(path) as con:
                cur = con.cursor()
                cur.execute("PRAGMA table_info(items)")
                cols = [r[1] for r in cur.fetchall()]
                if "serial_no" in cols:
                    return True
                cur.execute("ALTER TABLE items ADD COLUMN serial_no TEXT")
                con.commit()
                if show_toast: st.toast("DB patched: added items.serial_no", icon="✅")
                return True
        except Exception:
            continue
    if show_toast:
        st.toast("Could not auto-add items.serial_no (compat fallback will be used).", icon="⚠️")
    return False

def _load_serial_map() -> dict:
    try:
        raw = get_setting("serial_map", "{}") or "{}"
        m = json.loads(raw)
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}

def _save_serial_map(m: dict):
    try:
        upsert_setting("serial_map", json.dumps(m, ensure_ascii=False))
    except Exception:
        pass

def _overlay_serials(items: list[dict]) -> list[dict]:
    m = _load_serial_map()
    out = []
    for it in items:
        it = dict(it)
        if (not it.get("serial_no")) and it.get("sku") in m:
            it["serial_no"] = m[it["sku"]]
        out.append(it)
    return out

def get_items_with_serial() -> list[dict]:
    return _overlay_serials(get_items())

def _set_or_clear_serial_in_map(sku: str, serial: str | None):
    m = _load_serial_map()
    if sku and (serial or "").strip():
        m[sku] = serial.strip()
    else:
        m.pop(sku, None)
    _save_serial_map(m)

def _rename_serial_in_map(old_sku: str, new_sku: str):
    if not old_sku or not new_sku or old_sku == new_sku: return
    m = _load_serial_map()
    if old_sku in m:
        m[new_sku] = m.pop(old_sku)
        _save_serial_map(m)

_ensure_serialno_column(show_toast=False)

# ---------- Auto-detect bundled OpperWorks logo on first run ----------
if not logo_path:
    try:
        candidates = [
            "/mnt/data/OpperWorks Logo.png",
            "/mnt/data/OpperWorks_Logo.png",
            "/mnt/data/Logo R0.1.png",
            "/mnt/data/PG Bison.jpg",
        ]
        chosen = None
        for src in candidates:
            if os.path.exists(src):
                chosen = src
                break
        if chosen:
            ext = os.path.splitext(chosen)[1] or ".png"
            dst = os.path.join(ASSETS_DIR, f"brand_logo{ext}")
            if not os.path.exists(dst):
                shutil.copyfile(chosen, dst)
            upsert_setting("logo_path", dst)
            logo_path = dst
    except Exception:
        pass

# ---------- Sidebar brand header ----------
def render_brand_header():
    """Show logo at the very top. Falls back to brand text if logo missing."""
    if logo_path and os.path.exists(_norm_path(logo_path)):
        # Logo replaces the old text header
        st.sidebar.image(_norm_path(logo_path), use_container_width=True)
    else:
        st.sidebar.markdown(
            f"<h2 style='color:{brand_color}; margin: 0 0 6px 0'>{brand_name}</h2>",
            unsafe_allow_html=True
        )

render_brand_header()

# ---------- Navigation ----------
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

# ---------- Branding controls ----------
with st.sidebar.expander("Branding (Logo)", expanded=False):
    st.caption("Upload a PNG/JPG or type a server path, then **Save Logo**.")
    up = st.file_uploader("Upload logo", type=["png", "jpg", "jpeg"])
    col_a, col_b, col_c = st.columns([3, 1, 2])
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
    if col_c.button("Use OpperWorks logo"):
        try:
            src = "/mnt/data/OpperWorks Logo.png"
            if not os.path.exists(src):
                st.error("Bundled OpperWorks logo not found on this server.")
            else:
                ext = os.path.splitext(src)[1] or ".png"
                dst = os.path.join(ASSETS_DIR, f"brand_logo{ext}")
                shutil.copyfile(src, dst)
                upsert_setting("logo_path", dst)
                st.success("OpperWorks logo applied. Reloading…")
                st.rerun()
        except Exception as e:
            st.error(f"Could not apply default logo: {e}")

    if st.button("Clear Logo"):
        upsert_setting("logo_path", "")
        st.success("Logo cleared. Reloading…")
        st.rerun()
    if logo_path:
        try:
            st.image(_norm_path(logo_path), caption="Current logo", use_container_width=True)
        except Exception:
            st.caption("Logo path set, but preview failed.")

with st.sidebar.expander("Snapshot folder", expanded=False):
    st.caption("Current location for ZIPs (auto-backups and manual snapshots).")
    ok, resolved = _can_write_here(SNAP_DIR)
    st.code(resolved + ("  ✔" if ok else "  ✖"))
    new_snap_raw = st.text_input("Set snapshot folder", value=get_setting("snap_dir", SNAP_DIR))
    if st.button("Save folder"):
        try:
            new_abs = _coerce_path(new_snap_raw)
            ok2, resolved2 = _can_write_here(new_abs)
            if not ok2:
                st.error(f"That folder isn't accessible from this server: {resolved2}")
            else:
                upsert_setting("snap_dir", resolved2)
                st.success("Snapshot folder updated. Reloading…")
                st.rerun()
        except Exception as e:
            st.error(f"Could not set folder: {e}")

# ---------- Auto-backup (daily) ----------
def _has_snapshot_for_today():
    patt = os.path.join(SNAP_DIR, f"*{dt.date.today().strftime('%Y%m%d')}*.zip")
    return bool(glob.glob(patt))

if auto_backup_enabled and not _has_snapshot_for_today():
    try:
        items = get_items_with_serial()
        tx = get_transactions(limit=1_000_000)
        path = export_snapshot(items, tx, tag=f"Auto_{dt.date.today().isoformat()}", note="Auto-backup on app open")
        path = _move_to_snap_dir(path, SNAP_DIR)
        save_version_record(f"Auto_{dt.date.today().isoformat()}", "Auto-backup", path)
        st.sidebar.success("Auto-backup snapshot created for today.")
    except Exception:
        st.sidebar.warning("Auto-backup attempt failed (non-blocking).")

# ---------- PDF helpers/classes ----------
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
            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    self.image(self.logo_path, x=10, y=3, h=8)
                except Exception:
                    pass
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

def _fit_widths(widths, page_w):
    total = sum(widths)
    if total > page_w and total > 0:
        scale = page_w / total
        widths = [w * scale for w in widths]
    elif total < page_w and len(widths) > 0:
        bump = (page_w - total) / len(widths)
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

# ---------- Inventory PDF (grouped) ----------
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
        "sku", "serial_no", "name", "category", "location", "unit",
        "quantity", "min_qty", "unit_cost", "value",
        "convert_to", "converted_qty", "notes", "updated_at"
    ]
    present = [c for c in key_cols if c in df.columns]
    if not present:
        present = list(df.columns)
    df = df[present]

    col_names = {
        "sku": "SKU", "serial_no": "Serial #", "name": "Name", "category": "Category", "location": "Location",
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
            v = r[c] if c in r else None
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
    issued_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 6, to_latin1(f"Date Issued: {issued_ts}"), ln=1)

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

    # Wrap Name and Notes in the PDF
    wrap_idx_set = set()
    if "Name" in display_cols:
        wrap_idx_set.add(display_cols.index("Name"))
    if "Notes" in display_cols:
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

        cat_qty = float(block["quantity"].sum() if "quantity" in block.columns else 0.0)
        cat_value = float((block["quantity"] * block["unit_cost"]).sum() if {"quantity","unit_cost"}.issubset(block.columns) else 0.0)
        grand_qty += cat_qty
        grand_value += cat_value

        for _, r in block.iterrows():
            vals = []
            for c in present:
                v = r[c] if c in r else None
                if c in ("quantity", "min_qty"):
                    v = f"{float(v):,.2f}"
                elif c in ("unit_cost", "value"):
                    v = f"{float(v):,.2f}"
                elif c == "converted_qty" and pd.notna(v):
                    v = f"{float(v):,.3f}"
                vals.append(v if v is not None else "")
            fill = None
            try:
                if float(r.get("quantity", 0)) < float(r.get("min_qty", 0)):
                    fill = low_stock_fill
            except Exception:
                pass
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

# ---------- Issue Sheet PDF (A4 Landscape) + Return Sheet ----------
def _issue_sheet_pdf_bytes(
    df: pd.DataFrame,
    brand_name, brand_rgb, logo_path, revision_tag,
    manager: str, project: str, notes: str,
    categories=None, only_available: bool=True,
    blanks_cap: int = 12, fixed_blanks: int | None = None,
    include_returns: bool = False,
    return_blanks_cap: int = 12, fixed_return_blanks: int | None = None
) -> bytes:
    df = df.copy()
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
    df["min_qty"] = pd.to_numeric(df.get("min_qty"), errors="coerce").fillna(0.0)

    if categories and "category" in df.columns:
        df = df[df["category"].isin(categories)]
    if only_available:
        df = df[df["quantity"] > 0]

    present = ["sku","serial_no","name","unit","quantity","min_qty"]
    col_map = {"sku":"SKU","serial_no":"Serial #","name":"Item","unit":"Unit","quantity":"On-hand","min_qty":"Min"}
    display_cols = [col_map[c] for c in present if c in df.columns or c in present] + ["Qty Issued", "To (Person)", "Signature", "Date"]

    rows_for_width = []
    for _, r in df.iterrows():
        rows_for_width.append({
            "SKU": str(r.get("sku","")),
            "Serial #": str(r.get("serial_no","") or ""),
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
        orientation="L", unit="mm", format="A4"
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
        if name in {"Item"}:                 widths[i] = max(widths[i], 80)
        if name in {"Serial #"}:             widths[i] = max(widths[i], 35)
        if name in {"Qty Issued"}:           widths[i] = max(widths[i], 26)
        if name in {"To (Person)"}:          widths[i] = max(widths[i], 45)
        if name in {"Signature"}:            widths[i] = max(widths[i], 38)
        if name in {"Date"}:                 widths[i] = max(widths[i], 26)
    widths = _fit_widths(widths, page_w)

    _draw_header_row(pdf, display_cols, widths, 9, brand_rgb)
    align_map = {}
    if "On-hand" in display_cols: align_map[display_cols.index("On-hand")] = "R"
    if "Min" in display_cols:     align_map[display_cols.index("Min")]     = "R"
    cat_bar = lighten(brand_rgb, 0.80)

    wrap_issue_idx = set()
    if "Item" in display_cols:
        wrap_issue_idx.add(display_cols.index("Item"))

    if "category" in df.columns:
        df = df.sort_values(by=["category","sku","name"], kind="stable")
        groups = df["category"].fillna("(Unspecified)").unique().tolist()
    else:
        groups = ["(All)"]
        df["category"] = "(All)"

    per_item_blanks = []

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
            values_map = {
                "SKU": r.get("sku",""),
                "Serial #": r.get("serial_no","") or "",
                "Item": r.get("name",""),
                "Unit": r.get("unit","") or "",
                "On-hand": f"{onhand:,.2f}",
                "Min": f"{minq:,.2f}",
                "Qty Issued": "",
                "To (Person)": "",
                "Signature": "",
                "Date": ""
            }
            values = [values_map[col] for col in display_cols]
            _ensure_page_space(pdf, 8, display_cols, widths, 9, brand_rgb)
            _draw_row(pdf, values, widths, 7, align_map=align_map, border="1", wrap_idx_set=wrap_issue_idx)

            blanks = (fixed_blanks if (fixed_blanks is not None and fixed_blanks > 0)
                      else min(max(0, int(math.floor(onhand))) + 1, max(1, int(blanks_cap))))
            per_item_blanks.append({
                "cat": cat,
                "sku": r.get("sku",""),
                "serial_no": r.get("serial_no","") or "",
                "name": r.get("name",""),
                "unit": r.get("unit","") or "",
                "blanks": blanks
            })

            for _ in range(blanks):
                _ensure_page_space(pdf, 7, display_cols, widths, 9, brand_rgb)
                draw_ruled_blank_row(pdf, widths, row_h=7, line_rgb=(170,170,170))

    if include_returns:
        pdf.add_page(orientation="L")
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

        ret_cols = ["SKU","Serial #","Item","Unit","Qty Returned","From (Person)","Signature","Date","Condition / Notes"]
        ret_rows_for_width = [{k:"" for k in ret_cols}]
        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        ret_widths = _compute_col_widths(pdf, ret_cols, ret_rows_for_width, page_w, font_size=9)

        for i, name in enumerate(ret_cols):
            if name in {"Item"}:                   ret_widths[i] = max(ret_widths[i], 90)
            if name in {"Serial #"}:               ret_widths[i] = max(ret_widths[i], 35)
            if name in {"Qty Returned"}:           ret_widths[i] = max(ret_widths[i], 30)
            if name in {"From (Person)"}:          ret_widths[i] = max(ret_widths[i], 50)
            if name in {"Signature"}:              ret_widths[i] = max(ret_widths[i], 40)
            if name in {"Date"}:                   ret_widths[i] = max(ret_widths[i], 26)
            if name in {"Condition / Notes"}:      ret_widths[i] = max(ret_widths[i], 70)
        ret_widths = _fit_widths(ret_widths, page_w)

        _draw_header_row(pdf, ret_cols, ret_widths, 9, brand_rgb)

        wrap_return_idx = set()
        if "Item" in ret_cols: wrap_return_idx.add(ret_cols.index("Item"))
        if "Condition / Notes" in ret_cols: wrap_return_idx.add(ret_cols.index("Condition / Notes"))

        cat_bar = lighten(brand_rgb, 0.80)
        groups2 = [x["cat"] for x in per_item_blanks]
        groups2 = list(dict.fromkeys(groups2))

        for cat in groups2:
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
                desc_vals_map = {
                    "SKU": it["sku"],
                    "Serial #": it.get("serial_no","") or "",
                    "Item": it["name"],
                    "Unit": it["unit"],
                    "Qty Returned": "",
                    "From (Person)": "",
                    "Signature": "",
                    "Date": "",
                    "Condition / Notes": ""
                }
                desc_vals = [desc_vals_map[c] for c in ret_cols]
                _ensure_page_space(pdf, 8, ret_cols, ret_widths, 9, brand_rgb)
                _draw_row(pdf, desc_vals, ret_widths, 7, border="1", wrap_idx_set=wrap_return_idx)

                r_blanks = it["blanks"] if (fixed_return_blanks is None or fixed_return_blanks <= 0) else fixed_return_blanks
                for _ in range(r_blanks):
                    _ensure_page_space(pdf, 7, ret_cols, ret_widths, 9, brand_rgb)
                    draw_ruled_blank_row(pdf, ret_widths, row_h=7, line_rgb=(170,170,170))

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

# ---------- Views (Dashboard / Inventory / Transactions / Versions / Reports / Maintenance)
# ... (UNCHANGED logic below except that all calls to get_items() are replaced by get_items_with_serial())

def view_dashboard():
    st.title("🏠 Dashboard")
    st.caption("Quick overview + low-stock email helper.")

    items = get_items_with_serial()
    df = pd.DataFrame(items)
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    col1, col2, col3, col4 = st.columns(4)
    total_items = len(items)
    total_qty = sum((i.get("quantity") or 0) for i in items)
    low_stock = sum(1 for i in items if (i.get("min_qty") or 0) > (i.get("quantity") or 0))
    total_value = sum((i.get("quantity") or 0) * (i.get("unit_cost") or 0) for i in items)

    col1.metric("Distinct SKUs", total_items)
    col2.metric("Total Quantity", f"{total_qty:,.2f}")
    col3.metric("Low-Stock Items", low_stock)
    col4.metric("Stock Value", fmt_rands(total_value))

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
                cols = [c for c in ["sku","serial_no","name","category","location","quantity","min_qty"] if c in low_df.columns]
                st.dataframe(low_df[cols].fillna(""), use_container_width=True, height=260)

    with c2:
        st.subheader("Category totals")
        if df.empty:
            st.info("No data.")
        else:
            qty = df.groupby(df["category"].fillna("(Unspecified)"))["quantity"].sum().rename("qty")
            val = (df["quantity"] * df["unit_cost"]).groupby(df["category"].fillna("(Unspecified)")).sum().rename("value")
            cg = pd.concat([qty, val], axis=1)
            styler = cg.style.format({"qty": "{:,.2f}".format, "value": lambda v: fmt_rands(v)})
            st.dataframe(styler, use_container_width=True, height=260)

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
    st.caption("Add as you go: insert blank rows, type, and **Save**. Delete rows to remove. Editing SKU performs a rename.")
    _ensure_serialno_column(show_toast=False)

    if "inv_new_rows" not in st.session_state:
        st.session_state.inv_new_rows = 0

    items = get_items_with_serial()
    df = pd.DataFrame(items)
    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    with st.container():
        cA, cB, cC = st.columns([2,4,4])
        serial_find = cA.text_input("🔎 Find by Serial #", placeholder="Type full/part of serial…")
        filt = cB.text_input("Filter (SKU/Serial/Name/Category/Location contains…")
        dup_warn_toggle = cC.toggle("Warn on duplicate serials", value=True, help="Shows a warning if the grid contains duplicate Serial # values (ignores blanks).")

    fdf = df.copy()
    for col in ["sku","serial_no","name","category","location","unit","quantity","min_qty","unit_cost","convert_to","convert_factor","notes","updated_at"]:
        if col not in fdf.columns:
            fdf[col] = None

    if serial_find:
        s = serial_find.lower().strip()
        fdf = fdf[fdf["serial_no"].astype(str).str.lower().str.contains(s, na=False)]

    if filt:
        f = filt.lower()
        mask = (
            fdf["sku"].astype(str).str.lower().str.contains(f, na=False) |
            fdf["serial_no"].astype(str).str.lower().str.contains(f, na=False) |
            fdf["name"].astype(str).str.lower().str.contains(f, na=False) |
            fdf["category"].astype(str).str.lower().str.contains(f, na=False) |
            fdf["location"].astype(str).str.lower().str.contains(f, na=False)
        )
        fdf = fdf[mask]

    sort_keys = [k for k in ["category","sku","name"] if k in fdf.columns]
    if sort_keys:
        fdf = fdf.sort_values(by=sort_keys, kind="stable", na_position="last").reset_index(drop=True)

    st.subheader("Inventory List (editable)")
    fdf = fdf.copy()
    fdf["_orig_sku"] = fdf["sku"].astype(str)

    add_cols = st.columns([1,1,6,2])
    add_n = add_cols[0].number_input("Add rows", min_value=1, max_value=50, value=1, step=1)
    if add_cols[1].button("➕ Add blank row(s)"):
        st.session_state.inv_new_rows += int(add_n)

    def _blank_row_dict():
        return {
            "sku": "", "serial_no": "", "name": "", "category": "", "location": "", "unit": "",
            "quantity": 0.0, "min_qty": 0.0, "unit_cost": 0.0,
            "convert_to": "", "convert_factor": 0.0, "notes": "", "updated_at": "",
            "_orig_sku": ""
        }

    if st.session_state.inv_new_rows > 0:
        blanks = pd.DataFrame([_blank_row_dict() for _ in range(st.session_state.inv_new_rows)])
        fdf = pd.concat([fdf, blanks], ignore_index=True)

    has_textarea = hasattr(st.column_config, "TextAreaColumn")
    if has_textarea:
        name_col = st.column_config.TextAreaColumn("Name", help="Required", rows=2)
        notes_col = st.column_config.TextAreaColumn("Notes", rows=3)
    else:
        name_col = st.column_config.TextColumn("Name", help="Required")
        notes_col = st.column_config.TextColumn("Notes")
        st.markdown(
            """
            <style>
              [data-testid="stDataFrame"] div[role="gridcell"] {
                  white-space: normal !important;
                  line-height: 1.2rem !important;
                  overflow-wrap: anywhere !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

    column_config = {
        "sku": st.column_config.TextColumn("SKU", help="Required"),
        "serial_no": st.column_config.TextColumn("Serial Number"),
        "name": name_col,
        "category": st.column_config.TextColumn("Category"),
        "location": st.column_config.TextColumn("Location"),
        "unit": st.column_config.TextColumn("Unit"),
        "quantity": st.column_config.NumberColumn(format="%.3f"),
        "min_qty": st.column_config.NumberColumn(format="%.3f"),
        "unit_cost": st.column_config.NumberColumn(format="%.2f"),
        "convert_to": st.column_config.TextColumn("Convert To"),
        "convert_factor": st.column_config.NumberColumn(format="%.3f"),
        "notes": notes_col,
        "updated_at": st.column_config.TextColumn("Updated", help="Auto"),
        "_orig_sku": st.column_config.TextColumn("Original SKU", help="Used for renames; read-only", disabled=True, width="small"),
    }

    show_cols = ["sku","serial_no","name","category","location","unit",
                 "quantity","min_qty","unit_cost","convert_to","convert_factor","notes","updated_at","_orig_sku"]

    edited = st.data_editor(
        fdf[show_cols],
        use_container_width=True,
        num_rows="dynamic",
        key="inv_editor",
        height=900,
        column_config=column_config
    )

    try:
        ser = edited["serial_no"].astype(str).str.strip()
        ser = ser[ser != ""]
        dup_vals = ser[ser.duplicated(keep=False)]
        if dup_warn_toggle and len(dup_vals) > 0:
            dups = sorted(set(dup_vals.tolist()))
            st.warning(f"Duplicate Serial # detected in the grid: {', '.join(dups)}")
    except Exception:
        pass

    try:
        missing_mask = (edited["sku"].astype(str).str.strip() == "") | (edited["name"].astype(str).str.strip() == "")
        wont_save = int(missing_mask.sum())
        if wont_save > 0:
            st.info(f"{wont_save} row(s) missing SKU or Name will be skipped on Save.")
    except Exception:
        pass

    block_on_dup = st.checkbox("Block save on duplicates (Serial #)", value=False, help="If enabled, Save will abort when duplicate Serial # values are present (ignores blanks).")

    if add_cols[3].button("💾 Save (Upsert / Rename / Delete removed)"):
        try:
            ser = edited["serial_no"].astype(str).str.strip()
            ser = ser[ser != ""]
            has_dups = ser.duplicated().any()
        except Exception:
            has_dups = False
        if block_on_dup and has_dups:
            st.error("Save blocked: duplicate Serial # values present. Resolve and try again.")
            return

        original_skus = set(df["sku"].astype(str).tolist()) if not df.empty else set()
        edited = edited.fillna({"sku":"", "name":"", "_orig_sku":"", "serial_no":""})

        upserts = 0
        renames = 0
        creates = 0
        seen_targets = set()

        for _, r in edited.iterrows():
            new_sku = (r.get("sku") or "").strip()
            orig_sku = (r.get("_orig_sku") or "").strip()
            name = (r.get("name") or "").strip()

            if new_sku == "" and name == "":
                continue
            if not new_sku or not name:
                continue

            payload = {
                "sku": new_sku,
                "serial_no": (r.get("serial_no") or None),
                "name": name,
                "category": normalize_category(r.get("category")),
                "location": (r.get("location") or None),
                "unit": (r.get("unit") or None),
                "quantity": float(r.get("quantity") or 0),
                "min_qty": float(r.get("min_qty") or 0),
                "unit_cost": float(r.get("unit_cost") or 0),
                "notes": (r.get("notes") or None),
                "convert_to": (r.get("convert_to") or None),
                "convert_factor": float(r.get("convert_factor") or 0),
                "image_path": None,
            }

            if new_sku in seen_targets:
                continue
            seen_targets.add(new_sku)

            if orig_sku and new_sku != orig_sku and orig_sku in original_skus:
                try:
                    delete_item(orig_sku)
                except Exception:
                    pass
                add_or_update_item(payload)
                renames += 1
                _rename_serial_in_map(orig_sku, new_sku)
            else:
                if new_sku not in original_skus:
                    creates += 1
                else:
                    upserts += 1
                add_or_update_item(payload)

            _set_or_clear_serial_in_map(new_sku, payload.get("serial_no"))

        kept_orig_skus = set(edited["_orig_sku"].astype(str).tolist()) if "_orig_sku" in edited.columns else set()
        deleted_skus = list(original_skus - kept_orig_skus)
        deletions = 0
        for sku_del in deleted_skus:
            try:
                delete_item(sku_del)
                deletions += 1
            except Exception:
                pass
            _set_or_clear_serial_in_map(sku_del, None)

        st.session_state.inv_new_rows = 0
        st.success(f"Saved: {creates} new | {upserts} updates | {renames} renames | {deletions} deletions.")
        st.rerun()

    st.divider()
    colA, colB = st.columns(2)
    to_delete = colA.text_input("Delete by SKU")
    if colB.button("Delete Item"):
        if to_delete:
            delete_item(to_delete.strip())
            _set_or_clear_serial_in_map(to_delete.strip(), None)
            st.success(f"Deleted '{to_delete}'")
            st.rerun()
        else:
            st.error("Enter a SKU to delete.")

def view_transactions():
    st.title("🔁 Transactions")
    st.caption("Record stock movement in/out and maintain an audit trail.")

    items = get_items_with_serial()
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
        for col in ["sku","project","reference","user","notes"]:
            if col not in f.columns: f[col] = ""
        f = f[
            f["sku"].astype(str).str.lower().str.contains(s) |
            f["project"].astype(str).str.lower().str.contains(s) |
            f["reference"].astype(str).str.lower().str.contains(s) |
            f["user"].astype(str).str.lower().str.contains(s) |
            f["notes"].astype(str).str.lower().str.contains(s)
        ]
    st.dataframe(f, use_container_width=True)

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
        items = get_items_with_serial()
        tx = get_transactions(limit=1_000_000)
        zip_path = export_snapshot(items, tx, tag=tag, note=note)
        zip_path = _move_to_snap_dir(zip_path, SNAP_DIR)
        save_version_record(tag or "", note or "", zip_path)
        st.success(f"Snapshot created: {zip_path}")
        with open(zip_path, "rb") as f:
            st.download_button("Download ZIP", data=f.read(), file_name=os.path.basename(zip_path))

    st.divider()
    st.subheader("Site Profiles (snapshot-based)")
    cols = st.columns(3)
    site = cols[0].text_input("Site name", value="Main Workshop")
    if cols[1].button("Create snapshot for site"):
        items = get_items_with_serial()
        tx = get_transactions(limit=1_000_000)
        site_tag = f"{site.replace(' ', '_')}_{timestamp()}"
        zip_path = export_snapshot(items, tx, tag=site_tag, note=f"Site: {site}")
        zip_path = _move_to_snap_dir(zip_path, SNAP_DIR)
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
    st.caption("Snapshot files in current folder:")
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
        _save_serial_map({})

    added_items = 0
    for _, r in inv_df.iterrows():
        try:
            serial_val = (str(r.get("serial_no")).strip() if "serial_no" in inv_df.columns and pd.notna(r.get("serial_no")) else None)
            add_or_update_item({
                "sku": str(r.get("sku") or "").strip(),
                "serial_no": serial_val,
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
            _set_or_clear_serial_in_map(str(r.get("sku") or "").strip(), serial_val)
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
    items = get_items_with_serial()
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
        pdf.cell(0, 6, to_latin1(f"Date Issued: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=1)
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

    items = get_items_with_serial()
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
    df_all = pd.DataFrame(get_items_with_serial())
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
            items = get_items_with_serial()
            for it in items:
                cur = normalize_category(it.get("category"))
                if cur == old_cat:
                    it["category"] = target
                    add_or_update_item(it)
                    count += 1
            st.success(f"Updated {count} item(s) from '{old_cat}' → '{target}'.")
            st.rerun()
    if c4.button("Normalize categories (trim/collapse Title-Case)"):
        items = get_items_with_serial()
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

    st.divider()
    st.subheader("DB Repair")
    st.caption("If Serial Numbers aren’t sticking after Save, click this to patch the DB schema.")
    if st.button("Repair DB: add Serial Number column"):
        ok = _ensure_serialno_column(show_toast=True)
        if ok:
            st.success("Serial Number column is present. You can try saving again.")
        else:
            st.error("Could not patch automatically. If the DB file is custom, please share its path/name.")

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
