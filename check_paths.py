import sys, os
sys.path.insert(0, ".")

# Check what path database.py is actually using
from data.database import DB_PATH
print(f"DB_PATH in code: {DB_PATH}")
print(f"DB_PATH absolute: {os.path.abspath(DB_PATH)}")
print(f"File exists: {os.path.exists(DB_PATH)}")

import sqlite3
conn = sqlite3.connect(str(DB_PATH))
n = conn.execute("SELECT COUNT(*) FROM indicator_results").fetchone()[0]
print(f"indicator_results at DB_PATH: {n} rows")
conn.close()

# Also check the local path
local = "data/scanner.db"
print(f"\nLocal path: {os.path.abspath(local)}")
print(f"File exists: {os.path.exists(local)}")
conn2 = sqlite3.connect(local)
n2 = conn2.execute("SELECT COUNT(*) FROM indicator_results").fetchone()[0]
print(f"indicator_results at local path: {n2} rows")
conn2.close()
