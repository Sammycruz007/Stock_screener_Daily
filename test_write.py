import sys
sys.path.insert(0, ".")
import pandas as pd
from data.database import initialise_database, write_indicator_results, get_connection

initialise_database()

# Create a minimal test row
test_df = pd.DataFrame([{
    "ticker"          : "AAPL",
    "date"            : "2026-07-03 09:30:00",
    "linreg_value"    : 195.0,
    "linreg_slope"    : 0.001,
    "linreg_slope_up" : 1,
    "std_dev"         : 2.5,
    "sd1_upper"       : 197.5,
    "sd1_lower"       : 192.5,
    "sd2_upper"       : 200.0,
    "sd2_lower"       : 190.0,
    "sd3_upper"       : 202.5,
    "sd3_lower"       : 187.5,
    "price_sd_position": -1.8,
    "smc_structure"   : "bullish",
    "choch_detected"  : 0,
    "has_valid_zone"  : 1,
    "volume_signal"   : "accumulation",
}])

print("Writing test row...")
result = write_indicator_results(test_df)
print(f"write_indicator_results returned: {result}")

# Verify it landed
import sqlite3
conn = sqlite3.connect("data/scanner.db")
n = conn.execute("SELECT COUNT(*) FROM indicator_results").fetchone()[0]
rows = conn.execute("SELECT ticker, date FROM indicator_results").fetchall()
print(f"indicator_results now has: {n} rows")
print(f"Rows: {rows}")
conn.close()
