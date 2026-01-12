import os, sqlite3

os.makedirs("static", exist_ok=True)
db_path = os.path.join("static", "database.db")

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Simple Users table (prototype). Store hashed passwords in real apps.
c.execute("""
CREATE TABLE IF NOT EXISTS Users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  gender   TEXT
)
""")

conn.commit()
conn.close()
print(f"Initialized {db_path}")
