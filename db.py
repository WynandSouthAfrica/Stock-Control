import sqlite3, os, json, datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "stocktake.db")

# FIX 1: Context manager ensures connections are always closed after use
@contextmanager
def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT,
                location TEXT,
                unit TEXT,
                quantity REAL DEFAULT 0,
                min_qty REAL DEFAULT 0,
                unit_cost REAL DEFAULT 0,
                notes TEXT,
                image_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                sku TEXT NOT NULL,
                qty_change REAL NOT NULL,
                reason TEXT,
                project TEXT,
                reference TEXT,
                user TEXT,
                notes TEXT,
                FOREIGN KEY (sku) REFERENCES items (sku) ON UPDATE CASCADE ON DELETE CASCADE
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                tag TEXT,
                note TEXT,
                file_path TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

def upsert_setting(key, value):
    with _connect() as conn:
        conn.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        conn.commit()

def get_setting(key, default=None):
    with _connect() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return default
        # FIX 3: Catch specific exception instead of bare except
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return default

def add_or_update_item(item):
    with _connect() as conn:
        ts = now_iso()
        cur = conn.execute("""
            UPDATE items SET name=?, category=?, location=?, unit=?, quantity=?, min_qty=?, unit_cost=?,
                notes=?, image_path=?, updated_at=?
            WHERE sku=?
        """, (item['name'], item.get('category'), item.get('location'), item.get('unit'),
              item.get('quantity', 0), item.get('min_qty', 0), item.get('unit_cost', 0),
              item.get('notes'), item.get('image_path'), ts, item['sku']))
        if cur.rowcount == 0:
            conn.execute("""
                INSERT INTO items (sku, name, category, location, unit, quantity, min_qty, unit_cost, notes, image_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (item['sku'], item['name'], item.get('category'), item.get('location'), item.get('unit'),
                  item.get('quantity', 0), item.get('min_qty', 0), item.get('unit_cost', 0),
                  item.get('notes'), item.get('image_path'), ts, ts))
        conn.commit()

def get_items():
    with _connect() as conn:
        cur = conn.execute("SELECT sku, name, category, location, unit, quantity, min_qty, unit_cost, notes, image_path, created_at, updated_at FROM items ORDER BY sku")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def get_item(sku):
    with _connect() as conn:
        cur = conn.execute("SELECT sku, name, category, location, unit, quantity, min_qty, unit_cost, notes, image_path, created_at, updated_at FROM items WHERE sku=?", (sku,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

def delete_item(sku):
    with _connect() as conn:
        conn.execute("DELETE FROM items WHERE sku=?", (sku,))
        conn.commit()

def add_transaction(sku, qty_change, reason="", project="", reference="", user="", notes=""):
    # FIX 2: Explicit rollback if anything fails between the UPDATE and INSERT
    with _connect() as conn:
        try:
            ts = now_iso()
            cur = conn.execute("SELECT quantity FROM items WHERE sku=?", (sku,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"SKU '{sku}' not found")
            new_qty = float(row[0]) + float(qty_change)
            conn.execute("UPDATE items SET quantity=?, updated_at=? WHERE sku=?", (new_qty, ts, sku))
            conn.execute("""
                INSERT INTO transactions (ts, sku, qty_change, reason, project, reference, user, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, sku, qty_change, reason, project, reference, user, notes))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

def get_transactions(limit=500):
    with _connect() as conn:
        cur = conn.execute("""
            SELECT ts, sku, qty_change, reason, project, reference, user, notes
            FROM transactions
            ORDER BY ts DESC
            LIMIT ?
        """, (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def save_version_record(tag, note, file_path):
    with _connect() as conn:
        conn.execute("INSERT INTO versions (ts, tag, note, file_path) VALUES (?, ?, ?, ?)",
                     (now_iso(), tag, note, file_path))
        conn.commit()

def get_versions():
    with _connect() as conn:
        cur = conn.execute("SELECT ts, tag, note, file_path FROM versions ORDER BY ts DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
