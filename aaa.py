# seed_mock_profile_data.py
# Seeds rich mock data for username 'testtest' so ALL profile charts (including Ethics & Strategy radar)
# render with a meaningful, non-flat shape.
#
# How it works (matches your backend):
# - Ethics & Strategy radar is keyword-based: counts substrings in decision label/consequence/title.
# - So we insert decisions containing those exact keywords: civilian/combatant/verify (Distinction),
#   collateral/excessive/harm/casualties (Proportionality), necessary/objective/mission/only way (Necessity),
#   warning/evacuate/delay/minimize/avoid/safe route/protect/confirm (Precaution).
# - Sentiment + Performance Delta need RunReflections with:
#   phase='pre' questions containing 'confiden' and phase='post' questions containing 'satisf'
#   plus numeric response_text and sentiment_score.

import sqlite3
import json
from datetime import datetime, timedelta
import random
import os

# IMPORTANT: set this to the SAME DB file your Flask app uses.
# If you are unsure, open app.py and find DB_PATH / sqlite connect path.
DB_PATH = "static/database.db"

USERNAME = "testtest"
RUN_PREFIX = "Mock Research Run "


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def connect():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"DB not found at {DB_PATH}. "
            f"Check DB_PATH in this script matches the DB your Flask app uses."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_user(conn, username: str):
    c = conn.cursor()
    c.execute("SELECT username FROM Users WHERE username=?", (username,))
    if c.fetchone():
        return

    # Minimal insert aligned to your Users schema used in profile.html
    c.execute(
        """
        INSERT INTO Users(username, password, gender, age_group, preferred_war, interest, created_at)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            username,
            "testtest",
            "Prefer not to say",
            "18-24",
            "Research",
            "Ethics & Strategy",
            now_iso(),
        ),
    )


def clear_persona_cache(conn, username: str):
    c = conn.cursor()
    # Cache makes old “superhero/captain” persona stick around; clear it.
    c.execute("DELETE FROM UserPersonaCache WHERE username=?", (username,))


def delete_old_mock_runs(conn, username: str) -> int:
    """
    Delete previously seeded runs (and their decisions/reflections) for a clean re-seed.
    Safe to re-run.
    """
    c = conn.cursor()
    c.execute(
        "SELECT id FROM Runs WHERE username=? AND title LIKE ?",
        (username, f"{RUN_PREFIX}%"),
    )
    run_ids = [row["id"] for row in c.fetchall()]
    if not run_ids:
        return 0

    qmarks = ",".join(["?"] * len(run_ids))
    c.execute(f"DELETE FROM RunReflections WHERE run_id IN ({qmarks})", run_ids)
    c.execute(f"DELETE FROM RunDecisions   WHERE run_id IN ({qmarks})", run_ids)
    c.execute(f"DELETE FROM Runs          WHERE id IN ({qmarks})", run_ids)
    return len(run_ids)


def insert_run(conn, username: str, sid: str, title: str, total_steps: int, prefs: dict) -> int:
    c = conn.cursor()
    started = datetime.utcnow() - timedelta(days=random.randint(1, 21))
    finished = started + timedelta(minutes=random.randint(10, 28))
    c.execute(
        """
        INSERT INTO Runs(username, run_type, sid, title, total_steps, started_at, finished_at, prefs_json)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            username,
            "research",
            sid,
            title,
            total_steps,
            started.isoformat(timespec="seconds"),
            finished.isoformat(timespec="seconds"),
            json.dumps(prefs),
        ),
    )
    return c.lastrowid


def insert_decision(conn, run_id: int, step_index: int, value: str, label: str, consequence: str):
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO RunDecisions(run_id, step_index, option_value, option_label, option_consequence)
        VALUES(?,?,?,?,?)
        """,
        (run_id, step_index, value, label, consequence),
    )


def insert_reflection(
    conn,
    run_id: int,
    step_index: int,
    phase: str,
    q: str,
    resp: str,
    sentiment_score: float,
    sentiment_label: str,
    choice_value: str,
    choice_label: str,
):
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO RunReflections(
            run_id, step_index, phase, question_text, response_text,
            sentiment_score, sentiment_label, choice_value, choice_label
        )
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (run_id, step_index, phase, q, resp, float(sentiment_score), sentiment_label, choice_value, choice_label),
    )


def sentiment_label(score: float) -> str:
    if score > 0.2:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


# ===== Keyword-heavy decision templates (to light up radar axes) =====
# These strings intentionally contain substrings used by backend scoring.
DISTINCTION = [
    (
        "A",
        "Verify target identity; distinguish combatant from civilian and non-combatant presence.",
        "Distinction: verify and identify target; confirm uniform, weapon, armed indicators; avoid civilian harm.",
    ),
    (
        "A",
        "Confirm whether the individual is an armed combatant before engagement.",
        "Verify combatant status; identify weapon and uniform signals; protect civilians and non-combatants.",
    ),
]

PROPORTIONALITY = [
    (
        "C",
        "Choose a proportional response to reduce collateral damage and avoid excessive harm.",
        "Proportionality: reduce collateral damage; avoid excessive casualties and damage; manage risk trade-off.",
    ),
    (
        "C",
        "Limit strike radius to minimize casualties and excessive collateral damage.",
        "Balance military advantage against harm risk; avoid excessive damage and casualties (proportionality).",
    ),
]

NECESSITY = [
    (
        "D",
        "Proceed only if necessary to achieve the mission objective and military advantage.",
        "Necessity: action is necessary and critical to the objective; proceed only if the only way to meet mission goals.",
    ),
    (
        "D",
        "Select the only necessary action aligned to the objective under mission constraints.",
        "Necessary to achieve objective; critical mission requirement; avoid actions not essential to military advantage.",
    ),
]

PRECAUTION = [
    (
        "B",
        "Issue warning; delay action; establish a safe route to evacuate civilians.",
        "Precaution: warning and evacuate plan; delay to confirm conditions; minimize and avoid risk; protect civilians.",
    ),
    (
        "B",
        "Delay and confirm location; protect evacuation corridor; warn civilians to evacuate.",
        "Precaution: confirm, warning, evacuate, safe route; minimize harm and avoid risk to civilians.",
    ),
]

MIXED = [
    (
        "B",
        "Minimize collateral damage while issuing warning messages.",
        "Warning issued; minimize harm; reduce collateral damage; avoid excessive risk and protect civilians.",
    ),
    (
        "A",
        "Verify identity and confirm civilian proximity before action.",
        "Verify target; confirm civilians nearby; minimize harm and avoid risk with precautionary checks.",
    ),
]

# Weighted pools by step to create a “meaningful” non-uniform radar shape
def pick_decision_for_step(step_index: int):
    if step_index in (1, 2):
        pool = DISTINCTION + PRECAUTION + MIXED
        weights = [5] * len(DISTINCTION) + [5] * len(PRECAUTION) + [2] * len(MIXED)
    elif step_index in (3, 4):
        pool = PROPORTIONALITY + PRECAUTION + MIXED
        weights = [5] * len(PROPORTIONALITY) + [3] * len(PRECAUTION) + [2] * len(MIXED)
    else:
        pool = NECESSITY + PROPORTIONALITY + MIXED
        weights = [5] * len(NECESSITY) + [3] * len(PROPORTIONALITY) + [2] * len(MIXED)

    return random.choices(pool, weights=weights, k=1)[0]


def main():
    conn = connect()
    try:
        ensure_user(conn, USERNAME)

        # Reset old mock runs so radar won’t stay flat from older generic text
        deleted = delete_old_mock_runs(conn, USERNAME)

        # Clear persona cache so Decision DNA regenerates (using new seeded data)
        clear_persona_cache(conn, USERNAME)

        # Preferences mimic “library / research page” choices
        prefs = {
            "domain": "Ethics & policy",
            "setting": "Public-sector decision making",
            "difficulty": "Intermediate",
            "focus": ["fairness", "risk", "accountability", "LOAC"],
        }

        total_runs = 8
        total_steps = 6

        for r_i in range(total_runs):
            run_id = insert_run(
                conn,
                USERNAME,
                sid="A",
                title=f"{RUN_PREFIX}{r_i+1}",
                total_steps=total_steps,
                prefs=prefs,
            )

            for step in range(1, total_steps + 1):
                opt_value, opt_label, opt_consequence = pick_decision_for_step(step)

                insert_decision(
                    conn,
                    run_id,
                    step_index=step,
                    value=opt_value,
                    label=opt_label,
                    consequence=opt_consequence,
                )

                # Pre reflection (confidence + factor codes)
                pre_factor = random.choice(["CIV", "MS", "TP", "IQ"])
                pre_score = random.uniform(-0.15, 0.75)
                insert_reflection(
                    conn,
                    run_id,
                    step_index=step,
                    phase="pre",
                    q="Pre: How confident are you in your decision? (1-5)",
                    resp=str(random.choice([2, 3, 4, 5])),
                    sentiment_score=pre_score,
                    sentiment_label=sentiment_label(pre_score),
                    choice_value=pre_factor,
                    choice_label={
                        "CIV": "Civic impact",
                        "MS": "Moral stakes",
                        "TP": "Time pressure",
                        "IQ": "Information quality",
                    }[pre_factor],
                )

                # Post reflection (satisfaction + effect codes)
                post_effect = random.choice(["RR", "KS", "IR", "UN"])
                post_score = random.uniform(-0.25, 0.85)
                insert_reflection(
                    conn,
                    run_id,
                    step_index=step,
                    phase="post",
                    q="Post: How satisfied are you with the outcome? (1-5)",
                    resp=str(random.choice([2, 3, 4, 5])),
                    sentiment_score=post_score,
                    sentiment_label=sentiment_label(post_score),
                    choice_value=post_effect,
                    choice_label={
                        "RR": "Risk reduction",
                        "KS": "Knowledge sharing",
                        "IR": "Institutional response",
                        "UN": "Unintended effects",
                    }[post_effect],
                )

        conn.commit()
        print(
            f"Seed complete for '{USERNAME}'. Deleted {deleted} old mock runs. "
            f"Inserted {total_runs} runs × {total_steps} steps with keyword-rich decisions."
        )
        print("Persona cache cleared for test user.")
        print(f"DB used: {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
