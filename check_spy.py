import sqlite3
conn = sqlite3.connect("data/scanner.db")
rows = conn.execute(
    "SELECT ticker, date, linreg_slope, linreg_slope_up, price_sd_position "
    "FROM indicator_results "
    "WHERE ticker IN ('SPY','QQQ','DIA') "
    "ORDER BY date DESC LIMIT 10"
).fetchall()
for row in rows:
    print(row)
conn.close()