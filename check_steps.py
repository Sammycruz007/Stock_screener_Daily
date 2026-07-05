import sys
sys.path.insert(0, ".")
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/scanner.db")

tables = ["raw_prices", "filtered_universe", "indicator_results", "scan_results"]
for t in tables:
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {n} rows")

print()
# Check if filtered_universe has tickers for today
fu = pd.read_sql("SELECT * FROM filtered_universe ORDER BY scan_date DESC LIMIT 5", conn)
print("filtered_universe sample:")
print(fu)

print()
# Check latest indicator results date
try:
    d = conn.execute("SELECT MAX(date) FROM indicator_results").fetchone()[0]
    print(f"Latest indicator_results date: {d}")
except:
    print("indicator_results empty or missing")

conn.close()
