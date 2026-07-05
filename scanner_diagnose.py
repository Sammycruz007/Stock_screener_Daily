
import sys
sys.path.insert(0, ".")
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/scanner.db")

# Get latest indicator results
df = pd.read_sql("""
    SELECT * FROM indicator_results
    WHERE date = (SELECT MAX(date) FROM indicator_results)
""", conn)

print(f"Total tickers in indicator_results: {len(df)}")
print(f"Date: {df['date'].iloc[0] if len(df) > 0 else 'NONE'}")
print()

if len(df) == 0:
    print("NO INDICATOR RESULTS — engines never wrote output")
    conn.close()
    exit()

# Check each condition independently
slope_up     = (df["linreg_slope_up"] == 1)
in_sd_zone   = (df["price_sd_position"].between(-3, -1))
accum_vol    = (df["volume_signal"] == "accumulation")
has_zone     = (df["has_valid_zone"] == 1)

print("=== CONDITION BREAKDOWN (LONG) ===")
print(f"1. Slope UP:              {slope_up.sum()} / {len(df)}")
print(f"2. SD position -1 to -3:  {in_sd_zone.sum()} / {len(df)}")
print(f"3. Accumulation volume:   {accum_vol.sum()} / {len(df)}")
print(f"4. Has valid zone:        {has_zone.sum()} / {len(df)}")
print()
print(f"Passes 1+2:               {(slope_up & in_sd_zone).sum()}")
print(f"Passes 1+2+3:             {(slope_up & in_sd_zone & accum_vol).sum()}")
print(f"Passes 1+2+3+4:           {(slope_up & in_sd_zone & accum_vol & has_zone).sum()}")
print()
print("=== SD POSITION DISTRIBUTION ===")
print(df["price_sd_position"].describe())
print()
print("=== VOLUME SIGNAL COUNTS ===")
print(df["volume_signal"].value_counts())
print()
print("=== HAS_VALID_ZONE COUNTS ===")
print(df["has_valid_zone"].value_counts())
print()
print("=== SAMPLE OF ROWS CLOSEST TO -1 TO -3 SD ===")
close_to_zone = df[df["price_sd_position"].between(-4, 0)].sort_values("price_sd_position", ascending=False)
print(close_to_zone[["ticker", "price_sd_position", "linreg_slope_up", "volume_signal", "has_valid_zone"]].head(20).to_string(index=False))
