import sqlite3
conn = sqlite3.connect('data/scanner.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(model_metrics)').fetchall()]
print('model_metrics columns:', cols)
if 'recall_score' not in cols:
    conn.execute('ALTER TABLE model_metrics ADD COLUMN recall_score REAL')
    print('Added recall_score')
if 'pr_auc_score' not in cols:
    conn.execute('ALTER TABLE model_metrics ADD COLUMN pr_auc_score REAL')
    print('Added pr_auc_score')
conn.commit()
conn.close()
print('Done')
