import sys
sys.path.insert(0, ".")
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/scanner.db")
df = pd.read_sql("SELECT * FROM indicator_results WHERE date = (SELECT MAX(date) FROM indicator_results)", conn)
print(f"Rows: {len(df)}")
print(f"Date: {df['date'].iloc[0] if len(df) > 0 else 'NONE'}")

if len(df) == 0:
    print("EMPTY")
    conn.close()
    exit()

slope_up   = df["linreg_slope_up"] == 1
slope_down = df["linreg_slope_up"] == 0
in_long_sd = df["price_sd_position"].between(-3, -1)
in_short_sd = df["price_sd_position"].between(1, 3)
accum      = df["volume_signal"] == "accumulation"
distrib    = df["volume_signal"] == "distribution"
has_zone   = df["has_valid_zone"] == 1

print("\n=== LONG CONDITIONS ===")
print(f"1. Slope UP:           {slope_up.sum()} / {len(df)}")
print(f"2. SD -1 to -3:        {in_long_sd.sum()} / {len(df)}")
print(f"3. Accumulation vol:   {accum.sum()} / {len(df)}")
print(f"4. Has valid zone:      {has_zone.sum()} / {len(df)}")
print(f"Passes 1+2:            {(slope_up & in_long_sd).sum()}")
print(f"Passes 1+2+3:          {(slope_up & in_long_sd & accum).sum()}")
print(f"Passes 1+2+3+4:        {(slope_up & in_long_sd & accum & has_zone).sum()}")

print("\n=== SHORT CONDITIONS ===")
print(f"1. Slope DOWN:         {slope_down.sum()} / {len(df)}")
print(f"2. SD +1 to +3:        {in_short_sd.sum()} / {len(df)}")
print(f"3. Distribution vol:   {distrib.sum()} / {len(df)}")
print(f"4. Has valid zone:      {has_zone.sum()} / {len(df)}")
print(f"Passes 1+2:            {(slope_down & in_short_sd).sum()}")
print(f"Passes 1+2+3:          {(slope_down & in_short_sd & distrib).sum()}")
print(f"Passes 1+2+3+4:        {(slope_down & in_short_sd & distrib & has_zone).sum()}")

print("\n=== SD POSITION DISTRIBUTION ===")
print(df["price_sd_position"].describe())

print("\n=== VOLUME SIGNAL COUNTS ===")
print(df["volume_signal"].value_counts())

print("\n=== HAS_VALID_ZONE COUNTS ===")
print(df["has_valid_zone"].value_counts())

conn.close()
