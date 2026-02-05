from datetime import datetime, date, time
import sqlite3
import os
DB_PATH = os.path.join("static", "database.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
def delete_gpt_scenarios_before_today():
    """
    Permanently delete ALL GPT scenarios created before today (local date).
    This removes runs AND all dependent records.
    """

    # ---- define today's start (local date) ----
    today_start = datetime.combine(date.today(), time.min)
    today_iso = today_start.isoformat(timespec="seconds")

    conn = get_db()
    c = conn.cursor()

    # ---- find GPT run_ids to delete ----
    c.execute("""
        SELECT id
        FROM Runs
        WHERE run_type = 'gpt'
          AND started_at < ?
    """, (today_iso,))

    run_ids = [r["id"] for r in c.fetchall()]

    if not run_ids:
        conn.close()
        print("[library-cleanup] No GPT scenarios to delete.")
        return 0

    placeholders = ",".join(["?"] * len(run_ids))

    # ---- delete children first (FK safe) ----
    c.execute(f"DELETE FROM RunImages WHERE run_id IN ({placeholders})", run_ids)
    c.execute(f"DELETE FROM RunDecisions WHERE run_id IN ({placeholders})", run_ids)
    c.execute(f"DELETE FROM RunReflections WHERE run_id IN ({placeholders})", run_ids)
    c.execute(f"DELETE FROM RunJournals WHERE run_id IN ({placeholders})", run_ids)
    c.execute(f"DELETE FROM RunSequences WHERE run_id IN ({placeholders})", run_ids)

    # ---- delete runs ----
    c.execute(f"DELETE FROM Runs WHERE id IN ({placeholders})", run_ids)

    conn.commit()
    conn.close()

    print(f"[library-cleanup] Deleted {len(run_ids)} GPT scenarios created before today.")
    return len(run_ids)


delete_gpt_scenarios_before_today()