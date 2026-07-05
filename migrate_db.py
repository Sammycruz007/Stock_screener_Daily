import sqlite3
conn = sqlite3.connect("data/scanner.db")

# Check existing columns
cols = [row[1] for row in conn.execute("PRAGMA table_info(indicator_results)").fetchall()]
print("Existing columns:", cols)

# Add missing columns if not present
if "has_valid_zone" not in cols:
    conn.execute("ALTER TABLE indicator_results ADD COLUMN has_valid_zone INTEGER DEFAULT 0")
    print("Added: has_valid_zone")

if "choch_detected" not in cols:
    conn.execute("ALTER TABLE indicator_results ADD COLUMN choch_detected INTEGER DEFAULT 0")
    print("Added: choch_detected")

conn.commit()
conn.close()
print("Migration complete")
