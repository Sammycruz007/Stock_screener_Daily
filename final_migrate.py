import sqlite3
conn = sqlite3.connect('data/scanner.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(indicator_results)').fetchall()]
print('Current columns:', cols)
if 'has_valid_zone' not in cols:
    conn.execute('ALTER TABLE indicator_results ADD COLUMN has_valid_zone INTEGER DEFAULT 0')
    print('Added has_valid_zone')
conn.commit()
conn.close()
print('Done')
