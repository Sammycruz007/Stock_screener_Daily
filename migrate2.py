import sqlite3
conn = sqlite3.connect('data/scanner.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(scan_results)').fetchall()]
print('scan_results columns:', cols)
if 'has_valid_zone' not in cols:
    conn.execute('ALTER TABLE scan_results ADD COLUMN has_valid_zone INTEGER DEFAULT 0')
    print('Added has_valid_zone')
if 'volume_signal' not in cols:
    conn.execute('ALTER TABLE scan_results ADD COLUMN volume_signal TEXT')
    print('Added volume_signal')
conn.commit()
conn.close()
print('Done')
