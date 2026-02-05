# app.py (FULL UPDATED — GPT image generation is LAZY: generate only the current step image, one at a time)
# IMPORTANT: DO NOT hardcode your OpenAI key. Set OPENAI_API_KEY in your environment.

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3, os, json, re, base64, hashlib, time
from datetime import datetime, timezone
from functools import wraps
from collections import Counter
from typing import Optional, List

from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

DB_PATH = os.path.join("static", "database.db")
os.makedirs("static", exist_ok=True)
os.makedirs(os.path.join("static", "generated"), exist_ok=True)

# ===== GPT branch cache table (v2) =====
# Bump cache table version when the generation logic changes in a way that
# would make previously cached branches invalid.
GPT_BRANCH_TABLE = "gpt_branch_steps_v3"

# --- Jinja filters: timestamps -> local datetime strings ---
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


@app.template_filter("datetime")
def jinja_datetime_filter(value, fmt="%Y-%m-%d %H:%M"):
    if value is None:
        return ""
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return ""
    tz = ZoneInfo("Asia/Seoul") if ZoneInfo else None
    dt = datetime.fromtimestamp(ts, tz=tz) if tz else datetime.fromtimestamp(ts)
    return dt.strftime(fmt)


# ===== Safe url_for to prevent BuildError in templates =====
@app.context_processor
def inject_safe_url():
    def safe_url(endpoint, **values):
        try:
            return url_for(endpoint, **values)
        except Exception:
            return None

    return dict(safe_url=safe_url)


# ===== Simple DB helpers =====
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Backward-compatible alias (some older admin helpers used db_conn)
def db_conn():
    return get_db()


def ensure_column(conn: sqlite3.Connection, table: str, col: str, col_def_sql: str):
    """
    SQLite schema migration helper.
    Adds column if missing. col_def_sql should be like: "TEXT" or "INTEGER DEFAULT 0".
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def_sql}")
        conn.commit()


def safe_json_loads(s: Optional[str], default=None):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default



# ===== Research A/B assignment =====
def _assign_research_group(username: str, run_id: int) -> str:
    """Return 'A' or 'B' deterministically for a research run.
    A: scenario WITH images
    B: scenario WITHOUT images
    The user is not told which group they are in.
    """
    u = (username or '').strip().lower()
    seed = f"{u}|{int(run_id)}"
    h = hashlib.sha256(seed.encode('utf-8')).hexdigest()
    return 'A' if (int(h[:2], 16) % 2 == 0) else 'B'


def _persist_research_group(run_id: int, group_label: str):
    """Persist group label on Runs row (best-effort)."""
    try:
        conn = get_db()
        conn.execute("UPDATE Runs SET group_label=? WHERE id=?", (group_label, int(run_id)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def init_db():
    os.makedirs("static", exist_ok=True)
    conn = get_db()
    c = conn.cursor()

    # ---- Core tables ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS Users(
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        gender   TEXT,
        age_group TEXT,
        preferred_war TEXT,
        interest TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS Runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        run_type TEXT NOT NULL,  -- static | research | gpt
        sid TEXT,
        scenario_key TEXT,
        title TEXT,
        total_steps INTEGER,
        started_at TEXT,
        finished_at TEXT,
        prefs_json TEXT,
        FOREIGN KEY(username) REFERENCES Users(username)
    );
    """)

    ensure_column(conn, 'Runs', 'scenario_key', 'TEXT')
    ensure_column(conn, 'Runs', 'group_label', 'TEXT')


    # --- migration: store GPT analytic resolution for result page ---
    ensure_column(conn, "Runs", "gpt_resolution_json", "TEXT")
    ensure_column(conn, "Runs", "gpt_resolution_updated_at", "TEXT")

    c.execute("""
    CREATE TABLE IF NOT EXISTS RunDecisions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        step_index INTEGER NOT NULL,
        option_value TEXT,
        option_label TEXT,
        option_consequence TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    # ---- Decision analytics (LOAC / Strategy / Emotion) ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS RunDecisionMetrics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        step_index INTEGER NOT NULL,
        loac_json TEXT,                 -- JSON: Distinction/Proportionality/Necessity/Precaution scores 0..1
        strategy_json TEXT,             -- JSON: optional strategy traits 0..1
        emotional_index REAL,           -- 0..1 (higher = more emotionally intense)
        emotion_label TEXT,             -- e.g., calm | anxious | conflicted | distressed | confident
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(run_id, step_index),
        FOREIGN KEY(run_id) REFERENCES Runs(id) ON DELETE CASCADE
    );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_decisionmetrics_run ON RunDecisionMetrics(run_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_decisionmetrics_run_step ON RunDecisionMetrics(run_id, step_index);")

    # ---- Step Survey (modal) ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS RunStepSurveys(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        step_index INTEGER NOT NULL,
        choice_value TEXT,
        confidence TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        morality TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id) ON DELETE CASCADE
    );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_stepsurveys_run ON RunStepSurveys(run_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_stepsurveys_run_step ON RunStepSurveys(run_id, step_index);")

    c.execute("""
    CREATE TABLE IF NOT EXISTS RunReflections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        step_index INTEGER NOT NULL,
        phase TEXT CHECK(phase IN ('pre','post')) NOT NULL,
        question_text TEXT,
        response_text TEXT,
        sentiment_score REAL,
        sentiment_label TEXT,
        choice_value TEXT,
        choice_label TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS RunJournals(
        run_id INTEGER PRIMARY KEY,
        journal_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS RunSequences(
        run_id INTEGER PRIMARY KEY,
        sequence_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    # Stores images generated per run + step
    c.execute("""
    CREATE TABLE IF NOT EXISTS RunImages(
        run_id INTEGER NOT NULL,
        step_index INTEGER NOT NULL,
        image_path TEXT NOT NULL,        -- relative to /static
        prompt_hash TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(run_id, step_index),
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS ResearchSessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        access_code TEXT,
        start_time TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)


    ensure_column(conn, 'ResearchSessions', 'group_label', 'TEXT')

    # Persona cache

    # Persona cache
    c.execute("""
    CREATE TABLE IF NOT EXISTS UserPersonaCache(
        username TEXT PRIMARY KEY,
        persona_text TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # ---- Research-only analytics tables (STATIC research page only) ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS research_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        username TEXT NOT NULL,
        scenario_key TEXT NOT NULL,
        started_at INTEGER NOT NULL,
        finished_at INTEGER,
        source TEXT DEFAULT 'static',
        group_label TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS research_decisions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        user_id TEXT,
        username TEXT NOT NULL,
        scenario_key TEXT NOT NULL,
        step_id INTEGER NOT NULL,
        option_value TEXT NOT NULL,
        chosen_letter TEXT,
        chosen_label TEXT,
        scores_json TEXT,
        ethics_index REAL DEFAULT 0.0,
        created_at INTEGER NOT NULL,
        group_label TEXT,
        source TEXT DEFAULT 'static',
        UNIQUE(run_id, step_id, source),
        FOREIGN KEY(run_id) REFERENCES research_runs(id)
    );
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_research_runs_user ON research_runs(username);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_runs_finished ON research_runs(finished_at);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_decisions_run ON research_decisions(run_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_decisions_run_step ON research_decisions(run_id, step_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_decisions_group ON research_decisions(group_label);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_runs_group ON research_runs(group_label);")

    # ---- Static branch step cache (for research path-dependent steps) ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS static_branch_steps(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        base_sid TEXT NOT NULL,          -- e.g., "A"
        prefix TEXT NOT NULL,            -- e.g., "", "A", "AB", "ABCD" (choices before this step)
        step_index INTEGER NOT NULL,     -- 1..N
        step_json TEXT NOT NULL,         -- full step object as JSON
        created_at INTEGER NOT NULL,
        UNIQUE(base_sid, prefix, step_index)
    );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_branch_steps_sid ON static_branch_steps(base_sid);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_branch_steps_sid_prefix ON static_branch_steps(base_sid, prefix);")

    conn.commit()
    conn.close()


init_db()

# ===== Paths / Scenario load =====
SCENARIOS_JSON_PATHS = [
    os.path.join("static", "scenarios", "scenarios.json"),
    os.path.join("scenarios.json"),
]


def load_json_first(paths, required=True):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    if required:
        raise FileNotFoundError(f"None of these files found: {paths}")
    return {}


def validate_scenarios(schema):
    """Validate scenarios.json.

    This app supports both:
      - classic steps where fields are plain strings and options are a list
      - adaptive steps where fields can be {"byPrev": {...}, "default": ...}
        and options can be {"byPrev": {...}, "default": [...]}
    """
    if not isinstance(schema, dict) or not schema:
        raise ValueError("scenarios.json must be a non-empty top-level object keyed by scenario IDs (e.g., 'A').")

    for sid, seq in schema.items():
        if not isinstance(seq, dict):
            raise ValueError(f"Scenario '{sid}' must be an object.")
        # allow seq['id'] to be missing
        if "id" in seq and seq.get("id") != sid:
            raise ValueError(f"Scenario '{sid}' must have matching id field (or omit it).")

        steps = seq.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Scenario '{sid}' must include a non-empty steps array.")

        for st in steps:
            if not isinstance(st, dict):
                raise ValueError(f"Scenario '{sid}' has a step that is not an object.")
            if "id" not in st:
                raise ValueError(f"Scenario '{sid}' step missing 'id'.")
            # title optional (some users only show situation/question)
            if "situation" not in st:
                raise ValueError(f"Scenario '{sid}' step {st.get('id')} must include 'situation'.")
            if "question" not in st:
                raise ValueError(f"Scenario '{sid}' step {st.get('id')} must include 'question'.")
            if "options" not in st:
                raise ValueError(f"Scenario '{sid}' step {st.get('id')} must include 'options'.")

            opts = st.get("options")
            if isinstance(opts, list):
                if not opts:
                    raise ValueError(f"Scenario '{sid}' step {st.get('id')} has no options.")
                for opt in opts:
                    if not isinstance(opt, dict):
                        raise ValueError(f"Scenario '{sid}' step {st.get('id')} has an option that is not an object.")
                    if not all(k in opt for k in ("value", "label")):
                        raise ValueError(f"Scenario '{sid}' step {st.get('id')} option missing value/label.")
            elif isinstance(opts, dict):
                # adaptive options
                by_prev = opts.get("byPrev")
                default_list = opts.get("default")
                if by_prev is None and default_list is None:
                    raise ValueError(
                        f"Scenario '{sid}' step {st.get('id')} options must include 'byPrev' and/or 'default'."
                    )

                # validate lists inside
                def _check_list(lst, where):
                    if not isinstance(lst, list) or not lst:
                        raise ValueError(
                            f"Scenario '{sid}' step {st.get('id')} options {where} must be a non-empty list.")
                    for opt in lst:
                        if not isinstance(opt, dict) or "value" not in opt or "label" not in opt:
                            raise ValueError(
                                f"Scenario '{sid}' step {st.get('id')} option in {where} missing value/label.")

                if isinstance(default_list, list):
                    _check_list(default_list, "default")
                if isinstance(by_prev, dict):
                    for k, lst in by_prev.items():
                        _check_list(lst, f"byPrev[{k}]")
                elif by_prev is not None:
                    raise ValueError(f"Scenario '{sid}' step {st.get('id')} options.byPrev must be an object.")
            else:
                raise ValueError(f"Scenario '{sid}' step {st.get('id')} options must be a list or an object.")


SCENARIO_SEQUENCES = load_json_first(SCENARIOS_JSON_PATHS, required=True)
validate_scenarios(SCENARIO_SEQUENCES)


# ===== Helpers =====
def _resolve_by_prev(field, prev_choice):
    """Resolve a field that can be a string or an object like {'byPrev': {...}, 'default': ...}."""
    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        by_prev = field.get("byPrev")
        if prev_choice and isinstance(by_prev, dict) and prev_choice in by_prev:
            return by_prev.get(prev_choice)
        if "default" in field:
            return field.get("default")
    return field


def _resolve_options(options_field, prev_choice):
    """Resolve options field to a list[dict]."""
    if isinstance(options_field, list):
        return options_field
    if isinstance(options_field, dict):
        by_prev = options_field.get("byPrev")
        if prev_choice and isinstance(by_prev, dict) and prev_choice in by_prev:
            return by_prev.get(prev_choice) or []
        default_list = options_field.get("default")
        if isinstance(default_list, list):
            return default_list
    return []


def _resolve_step_view(step_raw, prev_choice):
    """Return a view-ready step dict with resolved title/situation/question/options."""
    if not isinstance(step_raw, dict):
        return {}
    return {
        **step_raw,
        "title": _resolve_by_prev(step_raw.get("title"), prev_choice),
        "situation": _resolve_by_prev(step_raw.get("situation"), prev_choice),
        "question": _resolve_by_prev(step_raw.get("question"), prev_choice),
        "options": _resolve_options(step_raw.get("options"), prev_choice),
    }


def _get_choice_details(seq: dict, progress: list, step_index_1based: int) -> dict:
    """Return resolved step + chosen option details for a completed step.

    step_index_1based: 1..N (must be <= len(progress) and <= len(seq['steps']))
    """
    steps = seq.get("steps", []) or []
    if step_index_1based < 1 or step_index_1based > len(steps):
        return {"value": "", "label": "", "consequence": "", "question": "", "situation": "", "title": ""}

    if step_index_1based < 1 or step_index_1based > len(progress):
        return {"value": "", "label": "", "consequence": "", "question": "", "situation": "", "title": ""}

    # Resolve step text/options based on the *previous* choice (adaptive steps)
    prev_choice = None
    if step_index_1based >= 2 and len(progress) >= (step_index_1based - 1):
        prev_choice = (progress[step_index_1based - 2] or "").strip() or None

    raw = steps[step_index_1based - 1]
    st = _resolve_step_view(raw, prev_choice)

    choice_val = (progress[step_index_1based - 1] or "").strip()
    opt = next(
        (o for o in (st.get("options") or [])
         if isinstance(o, dict) and (o.get("value") or "").strip() == choice_val),
        None
    ) or {}

    return {
        "value": choice_val,
        "label": (opt.get("label") or "").strip(),
        "consequence": (opt.get("consequence") or "").strip(),
        "story_line": (opt.get("story_line") or "").strip(),
        "question": (st.get("question") or "").strip(),
        "situation": (st.get("situation") or "").strip(),
        "title": (st.get("title") or "").strip(),
    }


# ===== Static branch step cache helpers (Research mode) =====
def _prefix_from_progress(progress: list, current_step: int) -> str:
    """
    Build a compact prefix from choices before current_step.
    Example: progress=["A1","B2"] and current_step=3 -> "AB"
    """
    letters = []
    upto = max(0, int(current_step) - 1)
    for v in (progress or [])[:upto]:
        if v:
            letters.append(str(v).strip().upper()[0])
    return "".join(letters)



# ===== GPT branch cache helpers (collision-free prefix) =====

def _gpt_prefix_from_progress(progress: list, step_index: int) -> str:
    """Return decision-path prefix up to (step_index-1) using full option_value tokens.

    Example:
      progress = ["accept", "route_b"]
      step_index=3 -> "accept/route_b"
      step_index=2 -> "accept"
    """
    if step_index <= 1:
        return ""
    upto = max(0, int(step_index) - 1)
    toks = []
    for v in (progress or [])[:upto]:
        vv = (str(v).strip() if v is not None else "")
        if vv:
            toks.append(vv)
    return "/".join(toks)


def _gpt_scenario_key_from_prefs(prefs: dict) -> str:
    """Stable key for a GPT scenario across users (research reuse)."""
    try:
        raw = json.dumps(prefs or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        raw = str(prefs or {})
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"GPT:{h}"



def ensure_gpt_branch_schema(conn: sqlite3.Connection):
    """Ensure the GPT branch-cache table exists (v2, path-dependent). Safe to call repeatedly."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {GPT_BRANCH_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_key TEXT NOT NULL,
            prefix TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(scenario_key, prefix, step_index)
        )
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GPT_BRANCH_TABLE}_lookup
        ON {GPT_BRANCH_TABLE}(scenario_key, prefix, step_index)
    """)
    conn.commit()


def _get_gpt_branch_step(conn: sqlite3.Connection, scenario_key: str, prefix: str, step_index: int) -> Optional[dict]:
    ensure_gpt_branch_schema(conn)
    row = conn.execute(
        f"""SELECT payload_json FROM {GPT_BRANCH_TABLE}
              WHERE scenario_key=? AND prefix=? AND step_index=?""",
        (scenario_key, prefix, int(step_index)),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload_json"])
    except Exception:
        return None


def _upsert_gpt_branch_step(conn: sqlite3.Connection, scenario_key: str, prefix: str, step_index: int, step_obj: dict):
    ensure_gpt_branch_schema(conn)

    # Never write NULL into payload_json
    if not isinstance(step_obj, dict) or not step_obj:
        step_obj = {
            "title": f"Step {step_index}",
            "situation": "This step could not be generated at the moment.",
            "question": "Please continue.",
            "options": [{"label": "Continue", "value": "continue", "consequence": ""}],
        }

    payload = json.dumps(step_obj, ensure_ascii=False)
    conn.execute(
        f"""INSERT INTO {GPT_BRANCH_TABLE}(scenario_key, prefix, step_index, payload_json, created_at)
              VALUES(?,?,?,?,?)
              ON CONFLICT(scenario_key, prefix, step_index) DO UPDATE SET
                payload_json=excluded.payload_json""",
        (scenario_key, prefix, int(step_index), payload, int(time.time())),
    )
    conn.commit()


def _generate_gpt_root_step(prefs: dict) -> dict:
    """Generate the root (Step 1) GPT step.

    Step 1 does not depend on any prior user choice, so we cache it once
    under prefix="". We reuse the existing generator that produces
    title/intro + step 1, then return only the step object.
    """
    seq = generate_gpt_initial_sequence_with_llm(prefs or {})
    step1 = (seq.get("steps") or [{}])[0] if isinstance(seq, dict) else {}
    if not isinstance(step1, dict) or not step1:
        return {
            "title": "Step 1",
            "situation": "This scenario could not be generated at the moment.",
            "question": "Please continue.",
            "options": [
                {"label": "Continue", "value": "A1", "consequence": ""},
                {"label": "Continue", "value": "B1", "consequence": ""},
                {"label": "Continue", "value": "C1", "consequence": ""},
                {"label": "Continue", "value": "D1", "consequence": ""},
            ],
        }
    # Ensure minimal keys exist (defensive)
    step1.setdefault("id", 1)
    step1.setdefault("title", "Step 1")
    step1.setdefault("question", "What do you do?")
    step1["options"] = _normalize_options_to_4(1, step1.get("options"))
    return step1

def _ensure_gpt_steps_upto(conn, scenario_key, prefs, progress, upto_step):
    """
    Ensure GPT steps up to `upto_step` exist.
    Step 1 is cached once.
    Step 2+ are cached by decision path (prefix).
    """
    ensure_gpt_branch_schema(conn)

    # ---------------------------
    # Step 1 (root, prefix = "")
    # ---------------------------
    root = _get_gpt_branch_step(conn, scenario_key, "", 1)
    if not root:
        root = _generate_gpt_root_step(prefs)
        _upsert_gpt_branch_step(conn, scenario_key, "", 1, root)

    steps_path = [root]

    # ---------------------------
    # Step 2+
    # ---------------------------
    for step_index in range(2, upto_step + 1):
        # prefix includes decisions up to step_index - 1
        prefix_tokens = [
            (v or "").strip()
            for v in progress[: step_index - 1]
            if (v or "").strip()
        ]
        prefix = "/".join(prefix_tokens)

        step_obj = _get_gpt_branch_step(
            conn,
            scenario_key,
            prefix,
            step_index
        )

        if not step_obj:
            # IMPORTANT: next steps must depend on the user's previous choice.
            # Build a recap (story_so_far) and a structured description of the last choice,
            # then generate the next step with the same LLM helper used by the admin regen.
            story_so_far = _story_from_gpt_path(steps_path, progress, step_index - 1)

            prev_step_obj = steps_path[step_index - 2] if len(steps_path) >= (step_index - 1) else {}
            prev_choice_val = progress[step_index - 2] if len(progress) >= (step_index - 1) else ""
            last_detail = _choice_detail_from_step(prev_step_obj, prev_choice_val)
            last_choice = (
                f"Value: {last_detail.get('value') or 'UNKNOWN'}\n"
                f"Question: {last_detail.get('question') or ''}\n"
                f"Chosen option: {last_detail.get('label') or ''}\n"
                f"Consequence: {last_detail.get('consequence') or ''}"
            )

            step_obj = generate_gpt_next_step_with_llm(
                prefs=prefs or {},
                story_so_far=story_so_far,
                last_choice_value=last_choice,
                step_index=step_index,
                admin_direction="",
            )
            _upsert_gpt_branch_step(
                conn,
                scenario_key,
                prefix,
                step_index,
                step_obj
            )

        steps_path.append(step_obj)

    return steps_path

def _choice_detail_from_step(step_obj: dict, chosen_value: Optional[str]) -> dict:
    step_obj = step_obj or {}
    chosen_value = (chosen_value or "").strip()
    chosen_opt = None
    for o in (step_obj.get("options") or []):
        try:
            if (o.get("value") or "").strip() == chosen_value:
                chosen_opt = o
                break
        except Exception:
            continue
    return {
        "question": step_obj.get("question", "") or "",
        "value": chosen_value,
        "label": (chosen_opt.get("label") if chosen_opt else "") or "",
        "consequence": (chosen_opt.get("consequence") if chosen_opt else "") or "",
    }


def _story_from_gpt_path(steps_path: List[dict], progress: list, upto: int) -> str:
    """Create a compact story recap up to 'upto' steps (1-based count)."""
    lines = []
    upto = max(0, int(upto))
    for i in range(1, upto + 1):
        step_obj = steps_path[i-1] if len(steps_path) >= i else {}
        choice_val = progress[i-1] if len(progress) >= i else None
        detail = _choice_detail_from_step(step_obj, choice_val)
        situation = (step_obj.get("situation") or "").strip()
        if situation:
            lines.append(f"Step {i} situation: {situation}")
        q = (detail.get("question") or "").strip()
        if q:
            lines.append(f"Step {i} question: {q}")
        if detail.get("label"):
            lines.append(f"Step {i} choice: {detail['label']}")
        if detail.get("consequence"):
            lines.append(f"Outcome: {detail['consequence']}")
    return "\n".join(lines).strip()
def _get_cached_branch_step(base_sid: str, prefix: str, step_index: int) -> Optional[dict]:
    if not base_sid:
        return None
    try:
        step_index = int(step_index)
    except Exception:
        return None
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT step_json FROM static_branch_steps
        WHERE base_sid=? AND prefix=? AND step_index=?
        LIMIT 1
        """,
        (base_sid, prefix or "", step_index),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row["step_json"])
    except Exception:
        return None


def _save_cached_branch_step(base_sid: str, prefix: str, step_index: int, step_obj: dict):
    if not base_sid:
        return
    try:
        step_index = int(step_index)
    except Exception:
        return
    now_ts = int(datetime.now(timezone.utc).timestamp())
    payload = json.dumps(step_obj or {}, ensure_ascii=False)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO static_branch_steps(base_sid, prefix, step_index, step_json, created_at)
        VALUES(?,?,?,?,?)
        """,
        (base_sid, prefix or "", step_index, payload, now_ts),
    )
    conn.commit()
    conn.close()


def generate_static_branch_step_with_llm(
        *,
        base_sid: str,
        base_seq: dict,
        step_index: int,
        prefix: str,
        story_so_far: str
) -> Optional[dict]:
    """
    Generate ONE step (not the whole scenario) for a given path prefix.
    Only used when a cached step for (base_sid, prefix, step_index) does not exist.
    """
    if not _client:
        return None

    try:
        step_index = int(step_index)
    except Exception:
        return None

    # Typically branch from step 2+
    if step_index < 2:
        return None

    title = (base_seq or {}).get("title") or f"Scenario {base_sid}"
    intro = (base_seq or {}).get("intro") or ""

    system = (
        "You write one step of an educational, non-graphic ethics decision scenario. "
        "Return STRICT JSON only (no markdown). "
        "The step must be coherent with the story so far and the user's prior choices."
    )

    user = (
        f"Base Scenario ID: {base_sid}\n"
        f"Scenario Title: {title}\n"
        f"Scenario Intro: {intro}\n"
        f"Path Prefix (choices before this step): {prefix}\n"
        f"Current Step Index to generate: {step_index} (1-based)\n\n"
        f"Story so far (context): {story_so_far}\n\n"
        "Generate a single step object with keys:\n"
        "- id: integer (must equal the current step index)\n"
        "- title: short string\n"
        "- situation: 2-5 sentences, non-graphic, no gore\n"
        "- question: one sentence\n"
        "- options: array of EXACTLY 4 objects; each object must include:\n"
        "  - value: must be exactly A{step}, B{step}, C{step}, D{step}\n"
        "  - label: short action\n"
        "  - consequence: 1-2 sentences\n"
        "Return only JSON."
    )

    try:
        step_obj = _llm_json(
            model=os.getenv("STATIC_BRANCH_STEP_MODEL", "gpt-4o-mini"),
            system=system,
            user=user,
            temperature=0.5
        )
    except Exception as e:
        print("[branch step] LLM failed:", e)
        return None

    if not isinstance(step_obj, dict):
        return None

    step_obj = dict(step_obj)
    step_obj["id"] = step_index
    step_obj["title"] = (step_obj.get("title") or f"Step {step_index}").strip()
    step_obj["situation"] = (step_obj.get("situation") or "").strip()
    step_obj["question"] = (step_obj.get("question") or "What do you do?").strip()

    # normalize to 4 options and enforce A{step}..D{step}
    fixed = _normalize_options_to_4(step_index, step_obj.get("options"))
    step_obj["options"] = [
        {"value": o["value"], "label": o["label"], "consequence": o.get("consequence", "")}
        for o in fixed
    ]
    return step_obj


def derive_recaps(seq: dict, progress: list):
    """Build recap objects for completed steps (supports adaptive steps)."""
    recaps = []
    steps = seq.get("steps", [])
    prev_choice = None
    for idx, choice in enumerate(progress[:len(steps)], start=1):
        raw = steps[idx - 1]
        st = _resolve_step_view(raw, prev_choice)
        opts = st.get("options", []) or []
        opt = next((o for o in opts if isinstance(o, dict) and o.get("value") == choice), None)
        if not opt:
            prev_choice = choice
            continue

        recaps.append({
            "situation": st.get("situation", "") or "",
            "chosen_label": opt.get("label", "") or "",
            "chosen_consequence": opt.get("consequence", "") or "",
            "chosen_story_line": opt.get("story_line", "") or "",
            "other_labels": [o.get("label", "") for o in opts if isinstance(o, dict) and o.get("value") != choice],
            "step_title": st.get("title") or f"Step {idx}",
            "step_index": idx
        })
        prev_choice = choice
    return recaps


def story_from_progress(seq: dict, progress: list, upto_step_exclusive: int):
    """Story so far = situation + explicit decision recap (label + story_line/consequence).

    This is used to condition GPT generation. Make the chosen option explicit so the model
    does not drift into other branches.
    """
    chunks = []
    intro = seq.get("intro")
    if isinstance(intro, str) and intro.strip():
        chunks.append(intro.strip())

    steps = seq.get("steps", [])
    upto = max(0, min(upto_step_exclusive, len(progress), len(steps)))

    prev_choice = None
    for idx in range(upto):
        raw = steps[idx]
        st = _resolve_step_view(raw, prev_choice)
        choice = (progress[idx] or "").strip()

        situation = st.get("situation")
        if isinstance(situation, str) and situation.strip():
            chunks.append(situation.strip())

        opt = next(
            (o for o in (st.get("options") or [])
             if isinstance(o, dict) and (o.get("value") or "").strip() == choice),
            None
        )
        if opt:
            label = (opt.get("label") or "").strip()
            line = (opt.get("story_line") or opt.get("consequence") or "").strip()

            if label and line:
                chunks.append(f"Decision: {label}. {line}")
            elif label:
                chunks.append(f"Decision: {label}.")
            elif line:
                chunks.append(line)

        prev_choice = choice

    return " ".join(chunks).strip()


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        if session.get("username") not in ("testtest", "scotttest"):
            flash("Admin access only.", "danger")
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)

    return wrapper


def _is_admin_user(username: Optional[str]) -> bool:
    return (username or "") in ("testtest", "scotttest")


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper

def _start_run(
    run_type,
    sid,
    title,
    total_steps,
    prefs=None,
    scenario_key=None
):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO Runs(
            username,
            run_type,
            sid,
            scenario_key,
            title,
            total_steps,
            started_at,
            prefs_json
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            session.get("username", "anonymous"),
            run_type,
            sid,
            scenario_key,
            title,
            total_steps,
            datetime.utcnow().isoformat(timespec="seconds"),
            json.dumps(prefs or {}, ensure_ascii=False),
        ),
    )

    run_id = c.lastrowid
    conn.commit()
    conn.close()

    session["run_id"] = run_id
    return run_id


def _fallback_decision_metrics(step_question: str, situation_text: str, opt_label: str, opt_consequence: str) -> dict:
    """Heuristic fallback when OpenAI key is missing or the LLM fails."""
    blob = f"{step_question}\n{situation_text}\n{opt_label}\n{opt_consequence}".lower()

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    # Very lightweight keyword heuristics (better than nothing; LLM is preferred).
    loac = {
        "Distinction": 0.5,
        "Proportionality": 0.5,
        "Necessity": 0.5,
        "Precaution": 0.5,
    }

    if any(k in blob for k in ["civilian", "non-combatant", "hostage", "school", "hospital"]):
        loac["Distinction"] = 0.7
        loac["Precaution"] = 0.65
    if any(k in blob for k in ["warning", "evacu", "delay", "confirm", "verify", "minimize"]):
        loac["Precaution"] = 0.75
    if any(k in blob for k in ["collateral", "excessive", "harm", "casualt", "damage"]):
        loac["Proportionality"] = 0.7
    if any(k in blob for k in ["only way", "necessary", "essential", "objective", "military advantage"]):
        loac["Necessity"] = 0.7

    emotional_index = 0.35
    emotion_label = "calm"
    if any(k in blob for k in ["kill", "execute", "bomb", "strike", "hostage", "children"]):
        emotional_index = 0.75
        emotion_label = "distressed"
    elif any(k in blob for k in ["risk", "uncertain", "unknown", "ambiguous"]):
        emotional_index = 0.55
        emotion_label = "anxious"

    strategy = {
        "risk_taking": 0.5,
        "caution": 0.5,
        "cooperation": 0.5,
        "aggression": 0.5
    }
    if any(k in blob for k in ["negot", "coordinate", "share", "joint", "collabor"]):
        strategy["cooperation"] = 0.7
        strategy["aggression"] = 0.35
    if any(k in blob for k in ["immediate", "force", "attack", "strike"]):
        strategy["aggression"] = 0.75
        strategy["caution"] = 0.35
        strategy["risk_taking"] = 0.65
    if any(k in blob for k in ["wait", "delay", "verify", "recon"]):
        strategy["caution"] = 0.75
        strategy["risk_taking"] = 0.35

    return {
        "loac": {k: _clamp01(v) for k, v in loac.items()},
        "strategy": {k: _clamp01(v) for k, v in strategy.items()},
        "emotional_index": _clamp01(emotional_index),
        "emotion_label": emotion_label,
        "rationale": "heuristic_fallback"
    }


def _analyze_decision_metrics(step_question: str, situation_text: str, opt_value: str, opt_label: str,
                              opt_consequence: str) -> dict:
    """Use GPT to score LOAC adherence + strategy traits + emotional index for a decision."""
    # If API key missing, fall back.
    if not _client:
        return _fallback_decision_metrics(step_question, situation_text, opt_label, opt_consequence)

    system = (
        "You are a strict evaluator for a scenario-based decision-making study. "
        "Given a scenario step and the user's chosen option, output compact JSON only."
    )

    user = f"""Evaluate the user's decision.

STEP_QUESTION:
{step_question}

SITUATION:
{situation_text}

CHOSEN_OPTION:
- value: {opt_value}
- label: {opt_label}
- consequence: {opt_consequence}

Return JSON with:
- loac: object with Distinction, Proportionality, Necessity, Precaution each in [0,1]
- strategy: object with risk_taking, caution, cooperation, aggression each in [0,1]
- emotional_index: number in [0,1]
- emotion_label: one of ['calm','anxious','conflicted','distressed','confident']
- rationale: short string <= 140 chars

Return only valid JSON. No markdown.
"""

    try:
        data = _llm_json(
            model=os.getenv("DECISION_METRICS_MODEL", "gpt-4o-mini"),
            system=system,
            user=user,
            temperature=0.2
        )
    except Exception as e:
        print("[decision metrics] LLM failed:", e)
        return _fallback_decision_metrics(step_question, situation_text, opt_label, opt_consequence)

    # Normalize + hard clamp
    def _clamp01(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    loac_in = (data.get("loac") or {})
    strat_in = (data.get("strategy") or {})
    out = {
        "loac": {
            "Distinction": _clamp01(loac_in.get("Distinction")),
            "Proportionality": _clamp01(loac_in.get("Proportionality")),
            "Necessity": _clamp01(loac_in.get("Necessity")),
            "Precaution": _clamp01(loac_in.get("Precaution")),
        },
        "strategy": {
            "risk_taking": _clamp01(strat_in.get("risk_taking")),
            "caution": _clamp01(strat_in.get("caution")),
            "cooperation": _clamp01(strat_in.get("cooperation")),
            "aggression": _clamp01(strat_in.get("aggression")),
        },
        "emotional_index": _clamp01(data.get("emotional_index")),
        "emotion_label": (data.get("emotion_label") or "conflicted").strip().lower(),
        "rationale": (data.get("rationale") or "").strip()[:140]
    }
    return out


def _upsert_decision_metrics(run_id: int, step_index: int, metrics: dict):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO RunDecisionMetrics(run_id, step_index, loac_json, strategy_json, emotional_index, emotion_label)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(run_id, step_index) DO UPDATE SET
            loac_json=excluded.loac_json,
            strategy_json=excluded.strategy_json,
            emotional_index=excluded.emotional_index,
            emotion_label=excluded.emotion_label
        """,
        (
            run_id,
            step_index,
            json.dumps(metrics.get("loac") or {}, ensure_ascii=False),
            json.dumps(metrics.get("strategy") or {}, ensure_ascii=False),
            metrics.get("emotional_index"),
            metrics.get("emotion_label"),
        ),
    )
    conn.commit()
    conn.close()


def _log_decision(run_id, step_index, opt_value, opt_label, opt_consequence, step_question=None, situation_text=None):
    """
    Persist the user's choice and also compute/store decision-level analytics.

    IMPORTANT FIX:
    - Some older DBs have RunDecisions.opt_value instead of RunDecisions.option_value.
    - This function now detects the column names and writes safely to either schema.
    """
    conn = get_db()
    c = conn.cursor()

    # Detect schema (option_value vs opt_value)
    c.execute("PRAGMA table_info(RunDecisions)")
    cols = {row[1] for row in c.fetchall()}  # row[1] = column name

    value_col = "option_value" if "option_value" in cols else ("opt_value" if "opt_value" in cols else None)
    label_col = "option_label" if "option_label" in cols else ("opt_label" if "opt_label" in cols else None)
    cons_col = "option_consequence" if "option_consequence" in cols else (
        "opt_consequence" if "opt_consequence" in cols else None)

    if not value_col:
        conn.close()
        raise RuntimeError("RunDecisions table missing decision value column (option_value/opt_value).")

    # Insert using whichever columns exist
    insert_cols = ["run_id", "step_index", value_col]
    insert_vals = [run_id, step_index, opt_value]

    if label_col:
        insert_cols.append(label_col)
        insert_vals.append(opt_label)

    if cons_col:
        insert_cols.append(cons_col)
        insert_vals.append(opt_consequence)

    sql = f"INSERT INTO RunDecisions({', '.join(insert_cols)}) VALUES({', '.join(['?'] * len(insert_cols))})"
    c.execute(sql, insert_vals)

    conn.commit()
    conn.close()

    # Decision analytics (best-effort; never block navigation)
    try:
        metrics = _analyze_decision_metrics(
            step_question=step_question or "",
            situation_text=situation_text or "",
            opt_value=opt_value or "",
            opt_label=opt_label or "",
            opt_consequence=opt_consequence or ""
        )
        _upsert_decision_metrics(run_id=run_id, step_index=step_index, metrics=metrics)
    except Exception as e:
        print("[decision metrics] store failed:", e)


def _finish_run(run_id):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "UPDATE Runs SET finished_at=? WHERE id=?",
        (datetime.utcnow().isoformat(timespec="seconds"), run_id),
    )
    conn.commit()
    conn.close()


# ===== OpenAI client (DO NOT hardcode keys) =====

_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _strip_json_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            mid = parts[1]
            if mid.lower().startswith("json"):
                mid = mid.split("\n", 1)[-1]
            return mid
        return t.replace("```", "")
    return t


def _get_run_resolution(run_id: Optional[int]) -> Optional[str]:
    if not run_id:
        return None
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT gpt_resolution_json FROM Runs WHERE id=?", (run_id,))
    row = cur.fetchone()
    conn.close()
    return row["gpt_resolution_json"] if row and row["gpt_resolution_json"] else None


def _save_run_resolution(run_id: int, resolution_json_text: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE Runs SET gpt_resolution_json=?, gpt_resolution_updated_at=? WHERE id=?",
        (resolution_json_text, datetime.utcnow().isoformat(timespec="seconds"), run_id)
    )
    conn.commit()
    conn.close()


def _llm_json(model: str, system: str, user: str, temperature: float = 0.3) -> dict:
    """
    Call OpenAI and return parsed JSON dict.
    Tries Responses API first; falls back to Chat Completions.
    """
    if not _client:
        raise RuntimeError("OPENAI_API_KEY not set")

    content_text = None

    # Prefer Responses API when available in the SDK
    if hasattr(_client, "responses"):
        resp = _client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            temperature=temperature,
        )
        content_text = getattr(resp, "output_text", None) or ""
    else:
        resp = _client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content_text = resp.choices[0].message.content or ""

    content_text = _strip_json_fences(content_text)

    try:
        return json.loads(content_text)
    except Exception:
        m = re.search(r"\{.*\}", content_text, re.DOTALL)
        if not m:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(m.group(0))


def generate_analytic_resolution(decisions_view: list, scenario_title: str, scenario_intro: str,
                                 path_letters: str) -> dict:
    """
    Build a research-friendly analytic resolution grounded in what the user chose.
    Returns a dict suitable for rendering.
    """
    # Minimize tokens: only send what we need
    trace = []
    for d in (decisions_view or []):
        trace.append({
            "step": d.get("step_id"),
            "title": d.get("step_title"),
            "situation": d.get("situation_text"),
            "question": d.get("question_text"),
            "choice": d.get("chosen_value"),
            "choice_label": d.get("chosen_label") or d.get("label"),
            "consequence": d.get("consequence"),
        })

    system = (
        "You are an expert analyst writing a concise post-scenario report for a research study. "
        "You MUST ground all claims in the provided decision trace. "
        "Return STRICT JSON only (no markdown)."
    )

    user = (
        f"Scenario Title: {scenario_title}\n"
        f"Scenario Intro: {scenario_intro or ''}\n"
        f"Path Letters: {path_letters}\n\n"
        f"Decision Trace (chronological, 5 steps):\n{json.dumps(trace, ensure_ascii=False)}\n\n"
        "Write an analytic resolution based on what the participant chose.\n"
        "Output JSON with these keys:\n"
        "- executive_summary: string (2-4 sentences)\n"
        "- decision_pattern: string (4-7 sentences; describe consistent priorities/strategy)\n"
        "- tradeoffs: array of 3-6 short bullet strings\n"
        "- ethical_analysis: string (6-10 sentences; mention fairness/harm/autonomy/accountability)\n"
        "- final_resolution: string (5-8 sentences; confident, specific closing)\n"
        "- counterfactual: string (3-5 sentences; one plausible alternative choice and impact)\n"
        "Keep it concrete, readable, and not too long."
    )

    # Pick a strong, affordable model for text analysis
    return _llm_json(model="gpt-4.1-mini", system=system, user=user, temperature=0.35)


def _safe_option_label(opt: dict, fallback: str) -> str:
    raw = (opt.get("label") or opt.get("text") or "").strip()
    if not raw or raw == "—":
        return fallback
    return raw


def _safe_option_consequence(opt: dict) -> str:
    return (opt.get("consequence") or opt.get("result") or opt.get("outcome") or "").strip()


def _extract_option_letter(value: str, fallback_letter: str) -> str:
    v = (value or "").strip().upper()
    m = re.match(r"([A-D])", v)
    return m.group(1) if m else fallback_letter


def _normalize_options_to_4(step_id: int, options_in) -> list:
    letters = ["A", "B", "C", "D"]

    opts_list = []
    if isinstance(options_in, dict):
        for k, v in list(options_in.items()):
            if isinstance(v, dict):
                tmp = dict(v)
                tmp.setdefault("value", k)
                opts_list.append(tmp)
            else:
                opts_list.append({"value": k, "label": str(v), "consequence": ""})
    elif isinstance(options_in, list):
        opts_list = [o for o in options_in if isinstance(o, dict)]
    else:
        opts_list = []

    opts_list = opts_list[:4]

    fixed = []
    used_letters = set()

    for i in range(min(4, len(opts_list))):
        opt = opts_list[i]
        fallback_letter = letters[i]
        letter = _extract_option_letter(opt.get("value", ""), fallback_letter)

        if letter in used_letters:
            letter = next((L for L in letters if L not in used_letters), fallback_letter)

        used_letters.add(letter)
        fixed.append({
            "value": f"{letter}{step_id}",
            "label": _safe_option_label(opt, f"Option {letter}"),
            "consequence": _safe_option_consequence(opt),
            "hint": (opt.get("hint") or "").strip(),
        })

    for L in letters:
        if len(fixed) >= 4:
            break
        if L in used_letters:
            continue
        fixed.append({
            "value": f"{L}{step_id}",
            "label": f"Option {L}",
            "consequence": "",
            "hint": "",
        })

    return fixed


def _coerce_gpt_sequence(seq: dict) -> dict:
    if not isinstance(seq, dict):
        raise ValueError("Generated scenario is not a JSON object.")

    seq = dict(seq)
    seq["id"] = "G"
    seq["title"] = (seq.get("title") or "Ethics-in-War Scenario").strip()
    seq["intro"] = (seq.get("intro") or "").strip()

    steps_in = seq.get("steps") or []
    if not isinstance(steps_in, list):
        steps_in = []

    fixed_steps = []
    for i in range(1, 6):
        st_in = steps_in[i - 1] if i - 1 < len(steps_in) and isinstance(steps_in[i - 1], dict) else {}
        st = dict(st_in)

        st["id"] = i
        st["title"] = (st.get("title") or f"Step {i}").strip()
        st["situation"] = (st.get("situation") or "").strip()
        st["question"] = (st.get("question") or "What do you do?").strip()
        st["options"] = _normalize_options_to_4(i, st.get("options"))

        fixed_steps.append(st)

    seq["steps"] = fixed_steps
    return seq


def _validate_generated_sequence_gpt(seq: dict):
    if not isinstance(seq, dict):
        raise ValueError("Generated scenario is not a JSON object.")
    if seq.get("id") != "G":
        raise ValueError("Generated scenario must have id='G'.")
    if not seq.get("title"):
        raise ValueError("Generated scenario missing 'title'.")

    steps = seq.get("steps")
    if not isinstance(steps, list) or len(steps) != 5:
        raise ValueError("Generated scenario must include exactly 5 steps.")

    for st in steps:
        for k in ("id", "title", "situation", "question", "options"):
            if k not in st:
                raise ValueError(f"Step missing '{k}'.")
        opts = st["options"]
        if not isinstance(opts, list) or len(opts) != 4:
            raise ValueError("Each step must have exactly 4 options.")
        sid = int(st["id"])
        expected = {f"A{sid}", f"B{sid}", f"C{sid}", f"D{sid}"}
        actual = {o.get("value") for o in opts}
        if actual != expected:
            raise ValueError(f"Step {sid} options are invalid values: {actual}")


def generate_ethics_sequence_with_llm(prefs: dict) -> dict:
    if not _client:
        raise RuntimeError("OPENAI_API_KEY not set")

    role_context = "Field Commander"
    if prefs.get("role") == "student":
        role_context = (
            "A high school student living in a conflict zone. Focus on dilemmas involving "
            "attending school, protecting fellow students, and managing limited educational resources "
            "or the use of school buildings by military forces."
        )
    elif prefs.get("role") == "medic":
        role_context = "A field medic focusing on triage and medical ethics."
    elif prefs.get("role") == "war-reporter":
        role_context = "A war reporter focusing on truth, safety, and harm minimization."
    elif prefs.get("role") == "humanitarian-coordinator":
        role_context = "A humanitarian relief coordinator balancing neutrality and access."

    system_prompt = (
        "Return STRICT JSON ONLY (no markdown, no explanations).\n"
        "Generate a 5-step ethics-in-war decision scenario aligned with International Humanitarian Law.\n"
        "No graphic violence.\n\n"
        "Rules:\n"
        "1) EXACTLY 5 steps.\n"
        "2) EACH step MUST have EXACTLY 4 options.\n"
        "3) Option values MUST be EXACTLY: A{step_id}, B{step_id}, C{step_id}, D{step_id}.\n"
        "4) Each option MUST include label and consequence.\n"
        "5) Each situation MUST be 250–400 characters (3–5 sentences).\n\n"
        "JSON schema:\n"
        "{\n"
        '  "id":"G",\n'
        '  "title":"...",\n'
        '  "intro":"...",\n'
        '  "steps":[\n'
        '    {\n'
        '      "id":1,\n'
        '      "title":"...",\n'
        '      "situation":"(250-400 chars)",\n'
        '      "question":"...",\n'
        '      "options":[\n'
        '        {"value":"A1","label":"...","consequence":"..."},\n'
        '        {"value":"B1","label":"...","consequence":"..."},\n'
        '        {"value":"C1","label":"...","consequence":"..."},\n'
        '        {"value":"D1","label":"...","consequence":"..."}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user_content = (
        f"Conflict: {prefs.get('war')}.\n"
        f"Front/Theatre: {prefs.get('theatre') or 'auto'}.\n"
        f"Role: {role_context}.\n"
        f"Tone: {prefs.get('tone')}.\n"
        f"Learning Goal: {prefs.get('goal')}.\n"
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    content = _strip_json_fences(resp.choices[0].message.content or "{}")
    try:
        raw = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise ValueError("Model did not return valid JSON.")
        raw = json.loads(m.group(0))

    seq = _coerce_gpt_sequence(raw)
    _validate_generated_sequence_gpt(seq)
    return seq


# ===== Store/retrieve GPT run sequences =====
def _save_run_sequence(run_id: int, seq: dict):
    conn = get_db()
    c = conn.cursor()
    c.execute("REPLACE INTO RunSequences(run_id, sequence_json) VALUES(?,?)",
              (run_id, json.dumps(seq, ensure_ascii=False)))
    conn.commit()
    conn.close()


def _get_run_sequence(run_id: int):
    if not run_id:
        return None
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT sequence_json FROM RunSequences WHERE run_id=?", (run_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row["sequence_json"])
    except Exception:
        return None


def _get_active_gpt_sequence():
    return _get_run_sequence(session.get("run_id"))


# ===== RunImages helpers (GPT lazy images) =====
def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _get_run_image(run_id: int, step_index: int) -> Optional[dict]:
    if not run_id:
        return None
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT run_id, step_index, image_path, prompt_hash FROM RunImages WHERE run_id=? AND step_index=?",
        (run_id, step_index),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def _save_run_image(run_id: int, step_index: int, image_path_rel_to_static: str, prompt_hash: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "REPLACE INTO RunImages(run_id, step_index, image_path, prompt_hash) VALUES(?,?,?,?)",
        (run_id, step_index, image_path_rel_to_static, prompt_hash),
    )
    conn.commit()
    conn.close()


def _delete_gpt_run_data_from_step(run_id: int, start_step: int):
    """When regenerating a GPT step, downstream steps/choices/images must be cleared."""
    if not run_id:
        return
    start_step = max(1, int(start_step))
    conn = get_db()
    c = conn.cursor()

    # Remove decisions at/after start_step (they no longer correspond to the regenerated narrative)
    c.execute("DELETE FROM RunDecisions WHERE run_id=? AND step_index>=?", (run_id, start_step))

    # Remove cached images at/after start_step (prompt/story changed)
    c.execute("DELETE FROM RunImages WHERE run_id=? AND step_index>=?", (run_id, start_step))

    # Optional: reflections (if you later add GPT pre/post reflections per step)
    c.execute("DELETE FROM RunReflections WHERE run_id=? AND step_index>=?", (run_id, start_step))

    conn.commit()
    conn.close()


def _build_gpt_image_prompt(seq: dict, prefs: dict, step_obj: dict, story_so_far: str) -> str:
    # Keep it safe + fast: illustration style, no gore, no explicit violence.
    # Also: avoid real-world identifiable flags/emblems; keep generic.
    role = (prefs or {}).get("role") or "field-commander"
    tone = (prefs or {}).get("tone") or "serious"
    war = (prefs or {}).get("war") or "conflict"
    theatre = (prefs or {}).get("theatre") or ""
    goal = (prefs or {}).get("goal") or ""

    ctx = (story_so_far or "")
    if len(ctx) > 600:
        ctx = ctx[-600:]

    situation = (step_obj.get("situation") or "").strip()
    title = (step_obj.get("title") or "").strip()

    prompt = (
            "Create a single, calm, educational illustration for a decision-making story.\n"
            "Style: modern storybook illustration, soft lighting, realistic proportions, gentle color palette.\n"
            "Safety: no gore, no graphic injury, no explicit violence, no weapons in focus.\n"
            "Do not include text, captions, logos, flags, or watermarks.\n\n"
            f"Scene title: {title}\n"
            f"Context: {war}" + (f" / {theatre}" if theatre else "") + "\n"
                                                                       f"Role POV: {role}\n"
                                                                       f"Learning focus: {goal}\n"
                                                                       f"Tone: {tone}\n\n"
                                                                       f"Story so far (brief): {ctx}\n\n"
                                                                       f"Current situation to illustrate: {situation}\n\n"
                                                                       "Composition: one key moment, clear facial expressions, ethical tension shown through body language, "
                                                                       "background hints of a disrupted civic environment (school, clinic, street, shelter) depending on the scene.\n"
    )
    return prompt


def _extract_b64_from_image_response(img_resp) -> Optional[str]:
    """
    Tries to be robust to slight SDK shape changes.
    Expected most commonly: img_resp.data[0].b64_json
    """
    if not img_resp:
        return None

    # object-style
    if hasattr(img_resp, "data") and img_resp.data:
        d0 = img_resp.data[0]
        # pydantic object or dict
        if hasattr(d0, "b64_json") and getattr(d0, "b64_json"):
            return getattr(d0, "b64_json")
        if isinstance(d0, dict) and d0.get("b64_json"):
            return d0.get("b64_json")

    # dict-style fallback
    if isinstance(img_resp, dict):
        data = img_resp.get("data") or []
        if data and isinstance(data[0], dict) and data[0].get("b64_json"):
            return data[0].get("b64_json")

    return None


def _generate_and_store_step_image(
        run_id: int,
        step_index: int,
        seq: dict,
        prefs: dict,
        step_obj: dict,
        story_so_far: str
) -> Optional[str]:
    """
    Generates ONLY ONE image (for the CURRENT step) and caches it in RunImages.
    Returns image URL path like: "/static/generated/....png" or None on failure.
    """
    if not _client:
        return None
    if not run_id:
        return None

    prompt = _build_gpt_image_prompt(seq, prefs, step_obj, story_so_far)
    ph = _sha256(prompt)

    existing = _get_run_image(run_id, step_index)
    if existing and existing.get("image_path") and existing.get("prompt_hash") == ph:
        return "/static/" + existing["image_path"].lstrip("/")

    out_dir = os.path.join(app.static_folder, "generated")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"run_{run_id}_step_{step_index}.png"
    abs_out = os.path.join(out_dir, filename)

    try:
        img = _client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )

        b64 = _extract_b64_from_image_response(img)
        if not b64:
            return None

        png_bytes = base64.b64decode(b64)
        with open(abs_out, "wb") as f:
            f.write(png_bytes)

        rel_path = os.path.join("generated", filename).replace("\\", "/")
        _save_run_image(run_id, step_index, rel_path, ph)

        return "/static/" + rel_path
    except Exception as e:
        print("[image] generation failed:", e)
        return None


# ===== Basic pages =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ===== Auth =====
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT username, password FROM Users WHERE username=?', (username,))
        user = c.fetchone()
        conn.close()
        if user and user["password"] == password:
            session["username"] = user["username"]
            return redirect(url_for("index"))
        flash("Invalid credentials. Try again.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        gender = request.form.get("gender")
        age_group = request.form.get("age_group")
        preferred_war = request.form.get("preferred_war")
        interest = request.form.get("interest")
        if not username or not password:
            flash("Username and password are required.", "warning")
            return render_template("register.html")
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute(
                'INSERT INTO Users (username, password, gender, age_group, preferred_war, interest) VALUES (?,?,?,?,?,?)',
                (username, password, gender, age_group, preferred_war, interest)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
    return render_template("register.html")


# ========= STATIC / RESEARCH =========
@app.route("/scenario", methods=["GET"])
def scenario_choose():
    return redirect(url_for("scenario_start", sid=get_active_static_sid()))


@app.route("/scenario/<sid>/start", methods=["GET"])
@login_required
def scenario_start(sid):
    if sid not in SCENARIO_SEQUENCES:
        sid = "A"
    session["mode"] = "static"
    session["scenario_sid"] = sid
    session["scenario_progress"] = []
    session.pop("step_start_time", None)
    session["make_image"] = False

    seq = SCENARIO_SEQUENCES[sid]
    _start_run(
        run_type="static",
        sid=sid,
        scenario_key=(f"LIB:{session.get('library_run_id')}" if (sid or '').strip().upper() == 'LIB' else (
                    seq.get('id') or (sid or '').strip().upper())),
        title=seq.get("title", f"Scenario {sid}"),
        total_steps=len(seq["steps"]),
        prefs=None
    )
    return redirect(url_for("scenario_step", sid=sid, step=1))


RESEARCH_ACCESS_CODE = "111"


RESEARCH_AB_SEED = os.environ.get("RESEARCH_AB_SEED", "research_ab_seed_v1")

def _assign_research_group(username: str, run_id: int) -> str:
    """Deterministic hidden A/B assignment for research.
    Group A = images ON, Group B = images OFF.
    """
    base = f"{RESEARCH_AB_SEED}|{username}|{run_id}"
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return "A" if (int(h[:8], 16) % 2 == 0) else "B"

def _persist_research_group(run_id: int, group_label: str, access_code: str = "") -> None:
    """Persist group assignment in Runs + ResearchSessions (best-effort)."""
    try:
        conn = get_db()
        conn.execute("UPDATE Runs SET group_label=? WHERE id=?", (group_label, run_id))
        # also mirror into the most recent ResearchSessions row for this run
        conn.execute("UPDATE ResearchSessions SET group_label=? WHERE run_id=?", (group_label, run_id))
        conn.commit()
        conn.close()
    except Exception:
        pass


@app.route("/research", methods=["GET", "POST"])
@login_required
def research_gate():
    if request.method == "POST":
        code = (request.form.get("access_code") or "").strip()
        if code == RESEARCH_ACCESS_CODE:
            # mark mode
            session["mode"] = "research"

            # If admin selected a Library story for static runs, use it.
            active_lib = (_get_cfg("active_static_library_run_id", "") or "").strip()
            if active_lib:
                session["scenario_sid"] = "LIB"
                try:
                    session["library_run_id"] = int(active_lib)
                except Exception:
                    session["library_run_id"] = None
            else:
                session["scenario_sid"] = "A"
                session.pop("library_run_id", None)

            session["scenario_progress"] = []
            session["step_start_time"] = datetime.utcnow().timestamp()

            # Start a research run record (stored in Runs)
            if session.get("scenario_sid") == "LIB":
                lib_seq = _get_library_sequence(session.get("library_run_id"))
                lib_steps = (lib_seq or {}).get("steps") or []
                run_title = "IHL Research Study (Library)"
                try:
                    dbt = get_db()
                    rr = dbt.execute("SELECT title FROM Runs WHERE id=?", (session.get("library_run_id"),)).fetchone()
                    dbt.close()
                    if rr and rr["title"]:
                        run_title = f'IHL Research Study (Library): {rr["title"]}'
                except Exception:
                    pass

                run_id = _start_run(
                    run_type="research",
                    sid=f"LIB:{session.get('library_run_id')}",
                    title=run_title,
                    total_steps=len(lib_steps),
                    scenario_key=f"LIB:{session.get('library_run_id')}"
                )
                redirect_sid = "LIB"
            else:
                run_id = _start_run(
                    run_type="research",
                    sid="A",
                    title="IHL Research Study",
                    total_steps=len(SCENARIO_SEQUENCES["A"]["steps"]),
                    scenario_key="A"
                )
                redirect_sid = "A"

            # --- assign hidden A/B group for research ---
            group_label = _assign_research_group(session.get("username") or "", int(run_id))
            session["research_group"] = group_label

            # Group A: with image, Group B: no image
            session["make_image"] = (group_label == "A")

            _persist_research_group(int(run_id), group_label)

            # Persist research session metadata (best-effort)
            try:
                conn = get_db()
                conn.execute(
                    "INSERT INTO ResearchSessions(run_id, access_code, start_time, group_label) VALUES(?,?,?,?)",
                    (int(run_id), code, datetime.utcnow().isoformat(timespec="seconds"), group_label)
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            session["run_id"] = int(run_id)

            # --- Pre-generate Step 1 image for Group A (so the first page loads fast) ---
            if session.get("make_image") and session.get("run_id"):
                try:
                    run_id_i = int(session.get("run_id"))
                    if redirect_sid == "LIB":
                        lib_run_id = session.get("library_run_id")
                        scenario_key = f"LIB:{lib_run_id}"
                        # Ensure step 1 exists in DB for this library story, then generate its image once.
                        prefs = dict(_load_run_prefs_by_run_id(lib_run_id) or {})
                        prefs["make_image"] = True
                        conn_g = get_db()
                        try:
                            steps_path = _ensure_gpt_steps_upto(
                                conn=conn_g,
                                scenario_key=scenario_key,
                                prefs=prefs,
                                progress=[],
                                upto_step=1,
                            )
                        finally:
                            conn_g.close()

                        step_obj = (steps_path or [{}])[0] if steps_path else {}
                        seq_for_img = _get_library_sequence(lib_run_id) or {"id": "LIB", "title": "Library Scenario"}
                        _generate_and_store_step_image(
                            run_id=run_id_i,
                            step_index=1,
                            seq=seq_for_img,
                            prefs=prefs,
                            step_obj=step_obj,
                            story_so_far=""
                        )
                    else:
                        seq_for_img = SCENARIO_SEQUENCES.get("A") or {}
                        step_obj = ((seq_for_img.get("steps") or [{}])[0])
                        prefs = _get_run_prefs(run_id_i) or {}
                        prefs["make_image"] = True
                        _generate_and_store_step_image(
                            run_id=run_id_i,
                            step_index=1,
                            seq=seq_for_img,
                            prefs=prefs,
                            step_obj=step_obj,
                            story_so_far=""
                        )
                except Exception as e:
                    print("[research] step1 image pregen failed:", e)

            return redirect(url_for("scenario_step", sid=redirect_sid, step=1))

        flash("Invalid Research Access Code.", "danger")

    return render_template("research_gate.html")


# ===== STATIC research-only scoring helpers (used by research_runs/research_decisions) =====
def load_scenarios_for_research():
    p = os.path.join("static", "scenarios", "scenarios.json")
    if not os.path.exists(p):
        raise FileNotFoundError("Missing static/scenarios/scenarios.json for research scoring.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_option(scenarios: dict, scenario_key: str, step_id: int, option_value: str):
    """Find an option dict by its value (e.g., 'A2').

    Supports both:
      - options: [ {value,label,...}, ... ]
      - options: { byPrev: {A1:[...], ...}, default:[...] }
    """
    s = scenarios.get(scenario_key)
    if not s:
        return None, None

    step = None
    for st in s.get("steps", []):
        try:
            if int(st.get("id")) == int(step_id):
                step = st
                break
        except Exception:
            continue
    if not step:
        return None, None

    options_field = step.get("options", [])
    candidate_lists = []

    if isinstance(options_field, list):
        candidate_lists.append(options_field)
    elif isinstance(options_field, dict):
        default_list = options_field.get("default")
        if isinstance(default_list, list):
            candidate_lists.append(default_list)

        by_prev = options_field.get("byPrev")
        if isinstance(by_prev, dict):
            for _, lst in by_prev.items():
                if isinstance(lst, list):
                    candidate_lists.append(lst)

    for lst in candidate_lists:
        for opt in lst:
            if isinstance(opt, dict) and opt.get("value") == option_value:
                return step, opt

    return step, None


def compute_ethics_index(scores: dict) -> float:
    dims = ["Distinction", "Proportionality", "Necessity", "Precaution"]
    total = 0.0
    for d in dims:
        v = scores.get(d, 0)
        try:
            total += float(v)
        except Exception:
            total += 0.0
    return total



def _upsert_static_research_decision(
        username: str,
        user_id: Optional[str],
        scenario_key: str,
        step_id: int,
        option_value: str,
        *,
        chosen_label_override: Optional[str] = None,
        scores_override: Optional[dict] = None,
        group_label: Optional[str] = None
):
    """Mirror a (static/research) step decision into the research_* tables.

    - If scenario_key exists in scenarios.json, we pull label/scores from there.
    - If it does NOT exist (e.g., Library stories), we still store the decision
      with best-effort fields (chosen_label_override / scores_override) so the
      admin research log can load instead of failing.
    """
    chosen_letter = (option_value[0] if option_value else "").upper()

    # Best-effort defaults
    chosen_label = (chosen_label_override or "").strip()
    scores = scores_override or {}
    if not isinstance(scores, dict):
        scores = {}

    # Try to enrich from scenarios.json (when available)
    scenarios = load_scenarios_for_research()
    try:
        _, opt = find_option(scenarios, scenario_key, step_id, option_value)
    except Exception:
        opt = None

    if isinstance(opt, dict) and opt:
        chosen_label = (opt.get("label") or chosen_label).strip()
        s2 = opt.get("scores")
        if isinstance(s2, dict):
            scores = s2

    ethics_index = compute_ethics_index(scores)
    now_ts = int(datetime.now().timestamp())

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Open run (unfinished) for this user+scenario_key+source
    cur.execute("""
        SELECT id FROM research_runs
        WHERE username=? AND scenario_key=? AND source='static' AND finished_at IS NULL
        ORDER BY started_at DESC LIMIT 1
    """, (username, scenario_key))
    row = cur.fetchone()

    if row:
        run_id = int(row["id"])
    else:
        cur.execute("""
            INSERT INTO research_runs(user_id, username, scenario_key, started_at, source, group_label)
            VALUES(?,?,?,?, 'static', ?)
        """, (user_id, username, scenario_key, now_ts, group_label))
        run_id = int(cur.lastrowid)

    # Persist group label on the research run (best-effort)
    if group_label:
        try:
            cur.execute("UPDATE research_runs SET group_label=? WHERE id=?", (group_label, run_id))
        except Exception:
            pass

    payload_scores_json = json.dumps(scores, ensure_ascii=False)

    # Upsert decision row
    cur.execute("""
        SELECT id FROM research_decisions
        WHERE run_id=? AND step_id=? AND source='static'
        LIMIT 1
    """, (run_id, int(step_id)))
    existing = cur.fetchone()

    if existing:
        cur.execute("""
            UPDATE research_decisions
            SET option_value=?, chosen_letter=?, chosen_label=?,
                scores_json=?, ethics_index=?, created_at=?, group_label=?
            WHERE id=?
        """, (
            option_value, chosen_letter, chosen_label,
            payload_scores_json, ethics_index, now_ts, group_label, int(existing["id"])
        ))
    else:
        cur.execute("""
            INSERT INTO research_decisions(
              run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              scores_json, ethics_index, created_at, group_label, source
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'static')
        """, (
            run_id, user_id, username, scenario_key, int(step_id),
            option_value, chosen_letter, chosen_label,
            payload_scores_json, ethics_index, now_ts, group_label
        ))

    # Mark run finished when we can infer total steps (known scenarios) and step_id is last.
    total_steps = len((scenarios.get(scenario_key) or {}).get("steps", [])) or 0
    if total_steps and int(step_id) >= total_steps:
        cur.execute("""
            UPDATE research_runs SET finished_at=?
            WHERE id=? AND finished_at IS NULL
        """, (now_ts, run_id))

    conn.commit()
    conn.close()
    return True, None


def _upsert_gpt_research_decision(
        username: str,
        user_id: Optional[str],
        run_id: int,
        step_id: int,
        option_value: str,
):
    """Mirror a GPT step decision into the research_* tables (source='gpt').

    We store LOAC-style scores in research_decisions.scores_json so the admin research
    log and report structure match the static scenario flow.
    """
    chosen_letter = (option_value[0] if option_value else "").upper()

    # Pull label + consequence from the stored GPT sequence (best-effort).
    chosen_label = ""
    scenarios_key = f"GPT:{run_id}"

    seq = _get_run_sequence(run_id)
    opt_consequence = ""
    step_question = ""
    situation_text = ""

    if isinstance(seq, dict):
        steps = seq.get("steps") or []
        if 1 <= step_id <= len(steps):
            st = steps[step_id - 1] or {}
            step_question = (st.get("question") or "").strip()
            situation_text = (st.get("situation") or "").strip()
            opt = next(
                (o for o in (st.get("options") or []) if isinstance(o, dict) and o.get("value") == option_value),
                None
            )
            if opt:
                chosen_label = (opt.get("label") or "").strip()
                opt_consequence = (opt.get("consequence") or "").strip()

    # Produce LOAC scores (best-effort)
    try:
        metrics = _analyze_decision_metrics(
            step_question=step_question,
            situation_text=situation_text,
            opt_value=option_value,
            opt_label=chosen_label or "",
            opt_consequence=opt_consequence,
        )
        scores = metrics.get("loac") or {}
        if not isinstance(scores, dict):
            scores = {}
    except Exception:
        scores = {}

    ethics_index = compute_ethics_index(scores)
    now_ts = int(datetime.now().timestamp())

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Open run (unfinished) for this user+scenario_key+source
    cur.execute(
        """
        SELECT id FROM research_runs
        WHERE username=? AND scenario_key=? AND source='gpt' AND finished_at IS NULL
        ORDER BY started_at DESC LIMIT 1
        """,
        (username, scenarios_key)
    )
    row = cur.fetchone()

    if row:
        research_run_id = row["id"]
    else:
        cur.execute(
            """
            INSERT INTO research_runs(user_id, username, scenario_key, started_at, source)
            VALUES(?,?,?,?, 'gpt')
            """,
            (user_id, username, scenarios_key, now_ts)
        )
        research_run_id = cur.lastrowid

    payload_scores_json = json.dumps(scores, ensure_ascii=False)

    # Upsert decision row
    cur.execute(
        """
        SELECT id FROM research_decisions
        WHERE run_id=? AND step_id=? AND source='gpt'
        LIMIT 1
        """,
        (research_run_id, step_id)
    )
    existing = cur.fetchone()

    if existing:
        cur.execute(
            """
            UPDATE research_decisions
            SET option_value=?, chosen_letter=?, chosen_label=?,
                scores_json=?, ethics_index=?, created_at=?, group_label=?
            WHERE id=?
            """,
            (option_value, chosen_letter, chosen_label, payload_scores_json, ethics_index, now_ts, existing["id"])
        )
    else:
        cur.execute(
            """
            INSERT INTO research_decisions(
              run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              scores_json, ethics_index, created_at, source
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?, 'gpt')
            """,
            (research_run_id, user_id, username, scenarios_key, step_id,
             option_value, chosen_letter, chosen_label, payload_scores_json, ethics_index, now_ts)
        )

    # GPT flow always has 5 steps
    if step_id >= 5:
        cur.execute(
            """
            UPDATE research_runs SET finished_at=?
            WHERE id=? AND finished_at IS NULL
            """,
            (now_ts, research_run_id)
        )

    conn.commit()
    conn.close()
    return True, None

@app.route("/scenario/<sid>/step/<int:step>", methods=["GET", "POST"])
@login_required
def scenario_step(sid, step):
    sid_u = (sid or "").strip().upper()
    mode = session.get("mode", "static")  # "static" | "research"
    progress = session.get("scenario_progress") or []
    run_id = session.get("run_id")

    # --- LIB: must have selected a library run id ---
    lib_run_id = session.get("library_run_id") if sid_u == "LIB" else None
    if sid_u == "LIB" and not lib_run_id:
        flash("No library story is selected.", "warning")
        return redirect(url_for("research_gate"))

    # --- Load sequence / steps ---
    # For LIB we treat it as a GPT-sequence stored in RunSequences for that library run.
    gpt_steps_path = None
    if sid_u == "LIB":
        run_id = int(lib_run_id)
        session["run_id"] = run_id  # IMPORTANT: make it stable on every request

        prefs = dict(_load_run_prefs_by_run_id(run_id) or {})
        prefs["make_image"] = bool(session.get("make_image", False))

        scenario_key = f"LIB:{run_id}"
        session["gpt_scenario_key"] = scenario_key

        # pull stored GPT sequence for this library run
        seq = _get_run_sequence(run_id)
        if not seq:
            flash("Selected library story has no saved sequence.", "danger")
            return redirect(url_for("research_gate"))

        steps = (seq.get("steps") or [])
        if not isinstance(steps, list) or not steps:
            flash("Selected library story has no steps.", "danger")
            return redirect(url_for("research_gate"))

        # Build the already-materialized steps along the path (if you store them),
        # else just use seq["steps"] as the canonical steps list.
        gpt_steps_path = steps

    else:
        # Static/Research uses SCENARIO_SEQUENCES
        if sid_u not in SCENARIO_SEQUENCES:
            flash("Scenario not found.", "danger")
            return redirect(url_for("index"))
        seq = SCENARIO_SEQUENCES[sid_u]
        steps = (seq.get("steps") or [])

    total_steps = len(steps)
    if step < 1 or step > total_steps:
        flash("Invalid step.", "warning")
        return redirect(url_for("scenario_start", sid=sid))

    # Ensure progress list length up to current step
    if len(progress) < step:
        progress = progress + [None] * (step - len(progress))
        session["scenario_progress"] = progress

    # --- Ensure run exists EARLY (even on GET) so images/answers can cache properly ---
    if sid_u != "LIB":
        if not run_id:
            run_id = _start_run(
                run_type=("research" if mode == "research" else "static"),
                sid=sid_u,
                title=seq.get("title", f"Scenario {sid_u}"),
                total_steps=total_steps,
                prefs=session.get("prefs") or {},
            )
            session["run_id"] = run_id

    # --- POST: save choice and advance ---
    if request.method == "POST":
        choice = (request.form.get("choice") or "").strip()
        if not choice:
            flash("Please select an option.", "warning")
            return redirect(url_for("scenario_step", sid=sid, step=step))

        progress[step - 1] = choice
        session["scenario_progress"] = progress

        # Persist answer
        try:
            _save_answer(run_id, step, choice, (seq.get("id") or sid_u))
        except Exception as e:
            print("[save_answer] failed:", e)

        # Next step / finish
        if step < total_steps:
            return redirect(url_for("scenario_step", sid=sid, step=step + 1))
        return redirect(url_for("run_result", run_id=run_id))

    # --- GET: build current step object ---
    if sid_u == "LIB":
        current = gpt_steps_path[step - 1]
    else:
        # Resolve adaptive static steps based on previous choice
        prev_choice = None
        if step >= 2 and len(progress) >= (step - 1):
            prev_choice = (progress[step - 2] or "").strip() or None
        current = _resolve_step_view(steps[step - 1], prev_choice)

    # --- Story so far (for image prompt + UI) ---
    if sid_u == "LIB":
        story_text = _story_from_gpt_path(gpt_steps_path, progress, step - 1)
    else:
        story_text = story_from_progress(seq, progress, step - 1)

    # --- Previous recap (for UI) ---
    prev_recap = None
    if step > 1 and len(progress) >= (step - 1):
        if sid_u == "LIB":
            try:
                prev_step_obj = gpt_steps_path[step - 2]
                prev_choice_val = progress[step - 2]
                prev_opt = next(
                    (o for o in (prev_step_obj.get("options") or []) if o.get("value") == prev_choice_val),
                    None,
                )
                if prev_opt:
                    prev_recap = {
                        "title": prev_step_obj.get("title") or f"Step {step-1}",
                        "choice": prev_choice_val,
                        "label": prev_opt.get("label", ""),
                        "consequence": prev_opt.get("consequence", ""),
                    }
            except Exception:
                prev_recap = None
        else:
            derived = derive_recaps(seq, progress)
            prev_recap = derived[step - 2] if len(derived) >= (step - 1) else None

    # --- Image policy ---
    # research: follow A/B toggle (make_image in session)
    # static: images enabled
    show_images = (bool(session.get("make_image", True)) if mode == "research" else (mode == "static"))

    hero_image = None
    hero_image_url = None
    img_debug_candidates = []
    img_debug_progress = progress[:]

    if show_images:
        # 1) Check packaged static images first
        prev_letters = []
        for c in progress[:max(0, step - 1)]:
            if c:
                prev_letters.append(str(c).strip().upper()[0])

        suffix = sid_u + "".join(prev_letters)
        cand1 = f"{sid_u}/step_{step}_{suffix}.png"
        cand2 = f"{sid_u}/step_{step}_{sid_u}.png"
        cand3 = f"{sid_u}/step_{step}.png"
        img_debug_candidates.extend([cand1, cand2, cand3])

        for cand in img_debug_candidates:
            abs_path = os.path.join(app.static_folder, "scenarios", cand)
            if os.path.exists(abs_path):
                hero_image = cand
                break

        # 2) Prefer scenario-path cache (prevents regenerating across new run_id)
        #    LIB: key MUST be stable -> "LIB:<library_run_id>"
        try:
            if sid_u == "LIB":
                scenario_key_for_img = f"LIB:{int(lib_run_id)}"
                prefix_for_img = _gpt_prefix_from_progress(progress, step)
            else:
                scenario_key_for_img = sid_u
                prefix_for_img = _prefix_from_progress(progress, step)

            hero_image_url = _generate_and_store_step_image_for_path(
                scenario_key=scenario_key_for_img,
                prefix=prefix_for_img,
                step_index=step,
                seq=seq,
                prefs=_get_run_prefs(run_id) or {},
                step_obj=current,
                story_so_far=story_text,
            )

            # Also record it to RunImages so report pages can locate by run_id (optional)
            if hero_image_url and run_id:
                rel = hero_image_url.replace("/static/", "", 1).lstrip("/")
                try:
                    _save_run_image(run_id, step, rel, None)
                except Exception:
                    pass

        except Exception as e:
            print("[image][path-cache] failed:", e)
            hero_image_url = None

        # 3) Legacy per-run cache fallback
        if (not hero_image_url) and run_id:
            cached_img = _get_run_image(run_id, step)
            if cached_img and cached_img.get("image_path"):
                hero_image_url = "/static/" + cached_img["image_path"].lstrip("/")

        # 4) Packaged static file fallback
        if (not hero_image_url) and hero_image:
            hero_image_url = "/static/scenarios/" + hero_image.lstrip("/")

    return render_template(
        "scenario_step_static.html",
        scenario_id=seq.get("id", sid),
        scenario_key=(seq.get("id") or (sid_u or sid)),
        scenario_title=seq.get("title", f"Scenario {sid}"),
        step=current,
        step_index=step,
        total_steps=total_steps,
        situation_text=current.get("situation", ""),
        story_so_far=story_text,
        prev_recap=prev_recap,
        hero_image=hero_image,
        hero_image_url=hero_image_url,
        show_images=show_images,
        is_last=(step == total_steps),
        sid=sid,
        selected=(progress[step - 1] if len(progress) >= step else None),
        run_id=run_id,
        is_admin=_is_admin_user(session.get("username")),
        img_debug_mode=mode,
        img_debug_progress=img_debug_progress,
        img_debug_candidates=img_debug_candidates,
    )

@app.route("/scenario/<sid>/result", methods=["GET"])
@login_required
def scenario_result(sid):
    # --- resolve sequence + steps (support LIB too) ---
    if sid == "LIB":
        lib_run_id = session.get("library_run_id")
        if not lib_run_id:
            flash("No library story is selected.", "warning")
            return redirect(url_for("research_gate"))

        prefs = dict(_load_run_prefs_by_run_id(lib_run_id) or {})
        prefs["make_image"] = bool(session.get("make_image", False))

        scenario_key = f"LIB:{lib_run_id}"
        session["gpt_scenario_key"] = scenario_key

        # Ensure all steps exist (or are loaded) for this library story.
        progress = session.get("scenario_progress", []) or []
        conn_g = get_db()
        try:
            steps_path = _ensure_gpt_steps_upto(
                conn=conn_g,
                scenario_key=scenario_key,
                prefs=prefs,
                progress=progress,
                upto_step=int(TOTAL_GPT_STEPS),
            )
        finally:
            conn_g.close()

        seq_title = "Library Scenario"
        try:
            conn_t = get_db()
            cur_t = conn_t.cursor()
            cur_t.execute("SELECT title FROM Runs WHERE id=?", (lib_run_id,))
            rr = cur_t.fetchone()
            conn_t.close()
            if rr and (rr.get("title") if isinstance(rr, dict) else rr["title"]):
                seq_title = (rr.get("title") if isinstance(rr, dict) else rr["title"]) or seq_title
        except Exception:
            pass

        seq = {"id": "LIB", "title": seq_title, "intro": "", "steps": steps_path, "resolutions": {}}
        steps = steps_path
    else:
        if sid not in SCENARIO_SEQUENCES:
            return redirect(url_for("scenario_start", sid="A"))
        seq = SCENARIO_SEQUENCES[sid]
        steps = seq.get("steps", []) or []

    if not steps:
        flash("Scenario is missing steps.", "danger")
        return redirect(url_for("scenario_start", sid="A"))

    progress = session.get("scenario_progress", []) or []
    total = len(steps)

    # --- IMPORTANT: don’t restart users if progress is slightly off ---
    if len(progress) < total:
        # Try to recover progress from DB (helps if session state was lost)
        run_id = session.get("run_id")
        if run_id:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "SELECT step_index, opt_value FROM RunDecisions WHERE run_id=? ORDER BY step_index ASC",
                (run_id,)
            )
            rows = cur.fetchall()
            conn.close()

            recovered = [""] * total
            for r in rows:
                si = int(r["step_index"])
                if 1 <= si <= total:
                    recovered[si - 1] = (r["opt_value"] or "").strip()

            if all(recovered):
                progress = recovered
                session["scenario_progress"] = progress
                session.modified = True
            else:
                # send them to the next unanswered step
                return redirect(url_for("scenario_step", sid=sid, step=len(progress) + 1))
        else:
            # send them to the next unanswered step
            return redirect(url_for("scenario_step", sid=sid, step=len(progress) + 1))

    if len(progress) > total:
        progress = progress[:total]
        session["scenario_progress"] = progress
        session.modified = True

    # ---- (the rest of your existing rendering logic) ----
    letters = [c[0].upper() for c in progress if c]
    dominant_letter = Counter(letters).most_common(1)[0][0] if letters else "A"

    resolutions = seq.get("resolutions", {})
    exec_summary = resolutions.get(dominant_letter) or resolutions.get("A") or {
        "title": "The Ethical Journey",
        "resolution": "Your choices shaped a distinct path under uncertainty."
    }

    decisions_view = []
    prev_choice = None
    for idx, choice in enumerate(progress, start=1):
        raw = steps[idx - 1]
        st = _resolve_step_view(raw, prev_choice)

        chosen_opt = next((o for o in (st.get("options") or []) if o.get("value") == choice), None) or {
            "label": "Unknown Choice",
            "consequence": "No data recorded.",
            "story_line": ""
        }

        decisions_view.append({
            "step_id": idx,
            "step_title": st.get("title", f"Phase {idx}"),
            "situation_text": st.get("situation", "") or "",
            "question_text": st.get("question", "") or "",
            "chosen_value": choice,
            "chosen_label": chosen_opt.get("label", "") or "",
            "label": chosen_opt.get("label", "") or "",
            "consequence": chosen_opt.get("consequence", "") or "",
            "story_line": chosen_opt.get("story_line", "") or "",
            "all_options": st.get("options", []) or []
        })

        prev_choice = choice

    run_id = session.get("run_id")

    # --- GPT analytic resolution (cached per run) ---
    gpt_report = None
    if run_id:
        cached = _get_run_resolution(run_id)
        if cached:
            gpt_report = safe_json_loads(cached, cached)
        elif _client:
            try:
                rep = generate_analytic_resolution(
                    decisions_view=decisions_view,
                    scenario_title=seq.get("title", "Ethics Report"),
                    scenario_intro=seq.get("intro", ""),
                    path_letters="".join(letters)
                )
                cached_text = json.dumps(rep, ensure_ascii=False)
                _save_run_resolution(run_id, cached_text)
                gpt_report = rep
            except Exception as e:
                print("[gpt resolution] generation failed:", e)

        _finish_run(run_id)

    return render_template(
        "scenario_result.html",
        scenario_id=sid,
        scenario_title=seq.get("title", "Ethics Report"),
        scenario_intro=seq.get("intro", ""),
        decisions=decisions_view,
        path_letters="".join(letters),
        exec_summary=(
            {"title": (exec_summary.get("title") or "Executive Summary"),
             "resolution": (gpt_report.get("executive_summary") if isinstance(gpt_report, dict) and gpt_report.get(
                 "executive_summary") else exec_summary.get("resolution"))}
            if isinstance(gpt_report, dict) else exec_summary
        ),
        ending=(gpt_report.get("final_resolution") if isinstance(gpt_report, dict) and gpt_report.get(
            "final_resolution") else exec_summary.get("resolution", "Path complete.")),
        timestamp=int(datetime.utcnow().timestamp()),
        gpt_report=gpt_report,
        run_journal=None
    )


@app.route("/gpt-scenario", methods=["GET", "POST"])
@login_required
def gpt_scenario_prefs():
    if request.method == "POST":
        make_image = bool(request.form.get("make_image"))
        session["make_image"] = make_image

        war_key = (request.form.get("war", "") or "").strip()
        war_payload = (request.form.get("war_custom_payload", "") or "").strip()

        # If custom-war selected, store the backend-safe payload string
        war_value = war_payload if (war_key == "custom-war" and war_payload) else war_key

        prefs = {
            "war": war_value,
            "theatre": (request.form.get("theatre", "") or "").strip(),
            "role": (request.form.get("role", "field-commander") or "").strip(),
            "tone": (request.form.get("tone", "serious & age-appropriate") or "").strip(),
            "goal": (request.form.get("goal", "") or "").strip(),
            "make_image": make_image,
        }

        try:
            # Only generates title + intro + step 1
            seq = generate_gpt_initial_sequence_with_llm(prefs)
            seq["total_steps"] = TOTAL_GPT_STEPS
        except Exception as e:
            flash(f"Scenario generation failed: {e}", "danger")
            return redirect(url_for("gpt_scenario_prefs"))

        session["mode"] = "gpt"
        session["scenario_sid"] = "G"
        session["scenario_progress"] = []
        session.pop("step_start_time", None)

        run_id = _start_run(
            run_type="gpt",
            sid="G",
            title=seq.get("title", "GPT Scenario"),
            total_steps=TOTAL_GPT_STEPS,  # IMPORTANT: not len(seq["steps"])
            prefs=prefs
        )
        session["run_id"] = run_id
        _save_run_sequence(run_id, seq)

        # Store GPT scenario metadata + seed step-1 into branch cache (prefix="")
        session["gpt_title"] = seq.get("title", "GPT Scenario")
        session["gpt_total_steps"] = TOTAL_GPT_STEPS
        scenario_key = _gpt_scenario_key_from_prefs(prefs)
        session["gpt_scenario_key"] = scenario_key

        try:
            conn = get_db()
            root_step = (seq.get("steps") or [{}])[0]
            if isinstance(root_step, dict):
                _upsert_gpt_branch_step(conn, scenario_key, "", 1, root_step)
                conn.commit()
            conn.close()
        except Exception:
            pass

        return redirect(url_for("gpt_scenario_step", step=1))

    session.setdefault("make_image", False)
    return render_template("gpt_scenario_prefs.html")


@app.route("/gpt-scenario/step/<int:step>", methods=["GET", "POST"])
@login_required
def gpt_scenario_step(step: int):
    """
    GPT scenario step view (path-dependent):
    - Step objects are cached by (scenario_key, prefix, step_index) in gpt_branch_steps
    - prefix is built from full option_value tokens up to (step_index-1) to avoid collisions
    """
    run_id = session.get("run_id")
    prefs = _get_run_prefs(run_id) if run_id else {}

    # Scenario key must be stable across users for research reuse
    scenario_key = session.get("gpt_scenario_key") or _gpt_scenario_key_from_prefs(prefs)
    session["gpt_scenario_key"] = scenario_key

    # total steps: keep existing behavior
    total_steps = int(session.get("gpt_total_steps") or TOTAL_GPT_STEPS)

    # Allow user to visit step up to total_steps
    if step < 1 or step > total_steps:
        return redirect(url_for("gpt_scenario_prefs"))

    progress = session.get("scenario_progress", []) or []

    # Prevent jumping ahead without prior choices
    if step > 1 and len(progress) < (step - 1):
        return redirect(url_for("gpt_scenario_step", step=max(1, len(progress) + 1)))

    conn = get_db()
    step_source = "static"

    try:
        # Determine whether THIS step exists before ensuring (for UI messaging)
        prefix_for_step = _gpt_prefix_from_progress(progress, step)
        existing = _get_gpt_branch_step(conn, scenario_key, (prefix_for_step if step > 1 else ""), step)

        # Ensure branch steps exist up to current step for this path
        steps_path = _ensure_gpt_steps_upto(conn, scenario_key, prefs, progress, step)

        # Current step object is the last in steps_path
        current = steps_path[step - 1] if len(steps_path) >= step else {}

        if existing is not None:
            step_source = "cache"
        else:
            step_source = "generate" if step >= 2 else ("cache" if _get_gpt_branch_step(conn, scenario_key, "", 1) else "generate")

        if request.method == "POST":
            choice = (request.form.get("choice") or "").strip()
            if not choice:
                flash("Please select one option.", "warning")
                return redirect(url_for("gpt_scenario_step", step=step))

            # Save/overwrite choice for this step
            if len(progress) >= step:
                progress[step - 1] = choice
            else:
                while len(progress) < step - 1:
                    progress.append("")
                progress.append(choice)

            # CRITICAL: truncate downstream steps when an earlier decision changes
            progress = progress[:step]
            session["scenario_progress"] = progress
            session.modified = True

            # Clear downstream run data (decisions/images/reflections) for this run
            if run_id:
                _delete_gpt_run_data_from_step(run_id, start_step=step + 1)

            chosen_opt = next((o for o in (current.get("options") or []) if (o.get("value") or "").strip() == choice), None)
            if run_id and chosen_opt:
                _log_decision(
                    run_id=run_id,
                    step_index=step,
                    opt_value=choice,
                    opt_label=chosen_opt.get("label", "") or "",
                    opt_consequence=chosen_opt.get("consequence", "") or "",
                    step_question=current.get("question", "") or "",
                    situation_text=current.get("situation", "") or ""
                )

            if step < total_steps:
                return redirect(url_for("gpt_scenario_step", step=step + 1))
            return redirect(url_for("gpt_scenario_result"))

        # ---- GET ----
        story_text = _story_from_gpt_path(steps_path, progress, upto=step - 1)

        prev_recap = None
        if step > 1 and len(progress) >= (step - 1):
            # reuse existing recap builder if available, else keep None
            try:
                derived = derive_recaps({"steps": steps_path}, progress)
                prev_recap = derived[step - 2] if len(derived) >= (step - 1) else None
            except Exception:
                prev_recap = None

        # LAZY IMAGE: generate ONLY the current step image if enabled.
        show_images = bool(session.get("make_image", False))
        hero_image_url = None
        if show_images and run_id:
            cached_img = _get_run_image(run_id, step)
            if cached_img and cached_img.get("image_path"):
                hero_image_url = "/static/" + cached_img["image_path"].lstrip("/")
            else:
                hero_image_url = _generate_and_store_step_image(
                    run_id=run_id,
                    step_index=step,
                    seq={"steps": steps_path, "title": session.get("gpt_title", "GPT Scenario")},
                    prefs=prefs or {},
                    step_obj=current,
                    story_so_far=story_text
                )

        return render_template(
            "scenario_step_gpt.html",
            scenario_id=scenario_key,
            scenario_title=session.get("gpt_title", "GPT Scenario"),
            step=current,
            step_index=step,
            total_steps=total_steps,
            situation_text=current.get("situation", ""),
            story_so_far=story_text,
            prev_recap=prev_recap,
            show_images=show_images,
            hero_image_url=hero_image_url,
        is_last=(step == total_steps),
            selected=(progress[step - 1] if len(progress) >= step else None),
            pre_survey=None,
            post_survey=None,
            step_missing=False,
            is_admin=_is_admin_user(session.get("username")),
            step_source=step_source
        )
    finally:
        conn.commit()
        conn.close()
@app.route("/admin/gpt/regenerate/step/<int:step>", methods=["POST"])
@admin_required
def admin_gpt_regenerate_step(step: int):
    """Admin-only: regenerate the CURRENT GPT step (and its options), with optional guidance, then clear downstream steps/choices/images."""
    run_id = session.get("run_id")
    if not run_id:
        flash("No active run found in session. Start a GPT scenario first.", "warning")
        return redirect(url_for("gpt_scenario_prefs"))

    seq = _get_run_sequence(run_id)
    if not seq:
        flash("No GPT sequence stored for this run.", "warning")
        return redirect(url_for("gpt_scenario_prefs"))

    total_steps = int(seq.get("total_steps") or TOTAL_GPT_STEPS)
    if step < 1 or step > total_steps:
        flash("Invalid step index.", "danger")
        return redirect(url_for("gpt_scenario_step", step=1))

    # NEW: guidance from admin
    admin_direction = (request.form.get("admin_direction") or "").strip()

    prefs = _get_run_prefs(run_id) or {}
    progress = session.get("scenario_progress", []) or []
    steps = seq.get("steps", []) or []

    # Ensure steps are generated up to the requested step (so we can replace it safely).
    if len(steps) < step:
        while len(steps) < step:
            next_step_index = len(steps) + 1
            if next_step_index == 1:
                initial = generate_gpt_initial_sequence_with_llm(prefs)
                seq["title"] = initial.get("title", seq.get("title"))
                seq["intro"] = initial.get("intro", seq.get("intro"))
                steps = initial.get("steps", []) or []
                seq["steps"] = steps
                seq["total_steps"] = total_steps
            else:
                story_text_prev = story_from_progress(seq, progress, next_step_index - 1)
                last_detail = _get_choice_details(seq, progress, next_step_index - 1)
                last_choice = (
                    f"Value: {last_detail.get('value') or 'UNKNOWN'}\\n"
                    f"Question: {last_detail.get('question') or ''}\\n"
                    f"Chosen option: {last_detail.get('label') or ''}\\n"
                    f"Consequence: {last_detail.get('consequence') or ''}"
                )
                new_step = generate_gpt_next_step_with_llm(
                    prefs=prefs,
                    story_so_far=story_text_prev,
                    last_choice_value=last_choice,
                    step_index=next_step_index,
                    admin_direction=""  # normal fill, no admin steer
                )
                steps.append(new_step)
                seq["steps"] = steps

        _save_run_sequence(run_id, seq)

    # Regenerate the specific step.
    try:
        if step == 1:
            # Step 1 regeneration is currently done by initial generator (title/intro/step1)
            # If you want admin guidance to influence step 1 as well, tell me and I’ll extend it safely.
            initial = generate_gpt_initial_sequence_with_llm(prefs)
            seq["title"] = initial.get("title", seq.get("title"))
            seq["intro"] = initial.get("intro", seq.get("intro"))
            steps = initial.get("steps", []) or []
            seq["steps"] = steps[:1]  # truncate after step 1
        else:
            story_text_prev = story_from_progress(seq, progress, step - 1)
            last_detail = _get_choice_details(seq, progress, step - 1)
            last_choice = (
                f"Value: {last_detail.get('value') or 'UNKNOWN'}\\n"
                f"Question: {last_detail.get('question') or ''}\\n"
                f"Chosen option: {last_detail.get('label') or ''}\\n"
                f"Consequence: {last_detail.get('consequence') or ''}"
            )

            new_step = generate_gpt_next_step_with_llm(
                prefs=prefs,
                story_so_far=story_text_prev,
                last_choice_value=last_choice,
                step_index=step,
                admin_direction=admin_direction  # NEW: steer regeneration
            )

            steps = (seq.get("steps", []) or [])[:]
            if len(steps) >= step:
                steps[step - 1] = new_step
            else:
                while len(steps) < step - 1:
                    steps.append({})
                steps.append(new_step)

            seq["steps"] = steps[:step]  # truncate after regenerated step

        seq["total_steps"] = total_steps
        _save_run_sequence(run_id, seq)

        # Clear downstream choices so user cannot keep later answers that no longer match regenerated story
        if len(progress) > step:
            progress = progress[:step]
            session["scenario_progress"] = progress
            session.modified = True

        # Clear downstream images (lazy image regeneration will rebuild as needed)
        try:
            _delete_gpt_run_data_from_step(run_id, step)  # if you already have this helper
        except Exception:
            # If helper doesn't exist, your app will still function;
            # images will be overwritten lazily when viewed.
            pass

        flash(f"Regenerated step {step} successfully.", "success")
        return redirect(url_for("gpt_scenario_step", step=step))

    except Exception as e:
        flash(f"Regeneration failed: {e}", "danger")
        return redirect(url_for("gpt_scenario_step", step=step))


@app.route("/gpt-scenario/result", methods=["GET"])
@login_required
def gpt_scenario_result():
    run_id = session.get("run_id")
    if not run_id:
        return redirect(url_for("gpt_scenario_prefs"))

    seq = _get_run_sequence(run_id)
    if not seq:
        return redirect(url_for("gpt_scenario_prefs"))

    steps = seq.get("steps", []) or []
    total_steps = int(seq.get("total_steps") or TOTAL_GPT_STEPS)

    progress = session.get("scenario_progress", [])

    # IMPORTANT FIX: user is done only when progress reaches total_steps
    if len(progress) < total_steps:
        # Try to recover progress from DB (helps if session state was lost)
        run_id = session.get("run_id")
        if run_id:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "SELECT step_index, opt_value FROM RunDecisions WHERE run_id=? ORDER BY step_index ASC",
                (run_id,)
            )
            rows = cur.fetchall()
            conn.close()

            recovered = [""] * total_steps
            for r in rows:
                si = int(r["step_index"])
                if 1 <= si <= total_steps:
                    recovered[si - 1] = (r["opt_value"] or "").strip()

            if all(recovered):
                progress = recovered
                session["gpt_scenario_progress"] = progress
                session.modified = True
            else:
                return redirect(url_for("gpt_scenario_step", step=len(progress) + 1))
        else:
            return redirect(url_for("gpt_scenario_step", step=len(progress) + 1))

    # Optional: ensure steps exist (in case someone reached result but steps not generated)
    prefs = _get_run_prefs(run_id) if run_id else {}
    while len(steps) < total_steps:
        next_step_index = len(steps) + 1
        story_text_prev = story_from_progress(seq, progress, next_step_index - 1)
        last_detail = _get_choice_details(seq, progress, next_step_index - 1)
        last_choice = (
            f"Value: {last_detail.get('value') or 'UNKNOWN'}\\n"
            f"Question: {last_detail.get('question') or ''}\\n"
            f"Chosen option: {last_detail.get('label') or ''}\\n"
            f"Consequence: {last_detail.get('consequence') or ''}"
        )
        new_step = generate_gpt_next_step_with_llm(
            prefs=prefs,
            story_so_far=story_text_prev,
            last_choice_value=last_choice,
            step_index=next_step_index
        )
        steps.append(new_step)

    seq["steps"] = steps
    _save_run_sequence(run_id, seq)

    decisions_view = []
    for idx, choice in enumerate(progress[:total_steps], start=1):
        st = steps[idx - 1]
        opt = next((o for o in (st.get("options") or []) if o.get("value") == choice), None)

        decisions_view.append({
            "step_id": idx,
            "step_title": st.get("title", f"Step {idx}"),
            "situation_text": st.get("situation", "") or "",
            "question_text": st.get("question", "") or "",
            "chosen_value": choice,
            "chosen_label": (opt.get("label") if opt else ""),
            "label": (opt.get("label") if opt else "Unknown Choice"),
            "consequence": (opt.get("consequence") if opt else ""),
            "story_line": (opt.get("story_line") if opt else ""),
            "all_options": st.get("options", []) or []
        })

    # --- GPT analytic resolution (cached per run) ---
    gpt_report = None
    cached = _get_run_resolution(run_id)
    if cached:
        gpt_report = safe_json_loads(cached, cached)
    elif _client:
        try:
            rep = generate_analytic_resolution(
                decisions_view=decisions_view,
                scenario_title=seq.get("title", "GPT Scenario"),
                scenario_intro=seq.get("intro", ""),
                path_letters="".join((c[0] for c in progress if c))
            )
            cached_text = json.dumps(rep, ensure_ascii=False)
            _save_run_resolution(run_id, cached_text)
            gpt_report = rep
        except Exception as e:
            print("[gpt resolution] generation failed:", e)

    _finish_run(run_id)

    ending = (gpt_report.get("final_resolution") if isinstance(gpt_report, dict) and gpt_report.get("final_resolution")
              else "Your decisions were recorded. Review the consequences and the evolving narrative.")

    return render_template(
        "scenario_result.html",
        scenario_id=seq.get("id", "G"),
        scenario_title=seq.get("title", "GPT Scenario"),
        scenario_intro=seq.get("intro", ""),
        decisions=decisions_view,
        full_story=story_from_progress(seq, progress, total_steps),
        ending=ending,
        path_letters="".join((c[0] for c in progress if c)),
        timestamp=int(datetime.utcnow().timestamp()),
        gpt_report=gpt_report,
        run_journal=None
    )


# ===== Profile analytics =====
def _safe_json_loads(s: Optional[str], default):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


def _iso_to_epoch(iso_text: Optional[str]) -> Optional[int]:
    """Convert stored UTC ISO strings (naive) to epoch seconds for JS."""
    if not iso_text:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_text))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _sentiment_bucket(score: Optional[float]):
    if score is None:
        return "neutral"
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _kw_score(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(t.count(k) for k in keywords)


def compute_profile_stats(runs: list) -> dict:
    total_runs = len(runs)
    total_decisions = sum(len(r.get("decisions", [])) for r in runs)
    total_reflections = sum(len(r.get("reflections", [])) for r in runs)

    letters = []
    for r in runs:
        for d in r.get("decisions", []):
            cv = (d.get("chosen_value") or "").strip().upper()
            if cv:
                letters.append(cv[0])
    letter_counts = Counter([x for x in letters if x in ("A", "B", "C", "D")])

    # ---- LOAC adherence + Emotional Index (from RunDecisionMetrics) ----
    loac_sum = {"Distinction": 0.0, "Proportionality": 0.0, "Necessity": 0.0, "Precaution": 0.0}
    loac_n = 0
    emo_scores = []
    emo_dist = {"calm": 0, "anxious": 0, "conflicted": 0, "distressed": 0, "confident": 0}

    for r in runs:
        for d in r.get("decisions", []):
            loac = d.get("loac") or {}
            if isinstance(loac, dict) and any(k in loac for k in loac_sum.keys()):
                try:
                    for k in loac_sum.keys():
                        v = loac.get(k)
                        if isinstance(v, (int, float)):
                            loac_sum[k] += float(v)
                    loac_n += 1
                except Exception:
                    pass

            ei = d.get("emotional_index")
            if isinstance(ei, (int, float)):
                emo_scores.append(float(ei))
            lab = (d.get("emotion_label") or "").strip().lower()
            if lab in emo_dist:
                emo_dist[lab] += 1

    DIM_KWS = {
        "Distinction": [
            "civilian", "non-combatant", "combatant", "distinction", "target", "identify",
            "verify", "uniform", "weapon", "armed"
        ],
        "Proportionality": [
            "proportion", "excessive", "collateral", "harm", "risk", "casualties",
            "damage", "balanc", "trade-off"
        ],
        "Necessity": [
            "necessary", "necessity", "objective", "mission", "military advantage",
            "critical", "essential", "only way"
        ],
        "Precaution": [
            "precaution", "warning", "evacu", "delay", "minimize", "avoid", "safe route",
            "confirm", "reduce risk", "protect"
        ],
    }

    def _safe_text(x):
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    raw = {k: 0 for k in DIM_KWS.keys()}
    for r in runs:
        for d in r.get("decisions", []):
            blob = " ".join([
                _safe_text(d.get("step_title")),
                _safe_text(d.get("chosen_label")),
                _safe_text(d.get("chosen_consequence")),
            ])
            for dim, kws in DIM_KWS.items():
                raw[dim] += _kw_score(blob, kws)

    maxv = max(raw.values()) if raw else 0
    norm = {k: (raw[k] / maxv if maxv > 0 else 0) for k in raw.keys()}

    pre_scores, post_scores = [], []
    dist = {"positive": 0, "neutral": 0, "negative": 0}
    for r in runs:
        for rf in r.get("reflections", []):
            s = rf.get("sentiment_score")
            phase = (rf.get("phase") or "").lower()
            if isinstance(s, (int, float)):
                if phase == "pre":
                    pre_scores.append(float(s))
                elif phase == "post":
                    post_scores.append(float(s))
            dist[_sentiment_bucket(s)] += 1

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0

    sentiment = {
        "pre_avg": _avg(pre_scores),
        "post_avg": _avg(post_scores),
        "dist": dist
    }

    def _extract_1to5(txt: str):
        if not txt:
            return None
        m = re.search(r"\b([1-5])\b", txt)
        return int(m.group(1)) if m else None

    pre_conf = []
    post_sat = []
    for r in runs:
        for rf in r.get("reflections", []):
            q = (rf.get("question_text") or "").lower()
            val = _extract_1to5(rf.get("response_text") or "")
            if val is None:
                continue
            if rf.get("phase") == "pre" and "confiden" in q:
                pre_conf.append(val)
            if rf.get("phase") == "post" and "satisf" in q:
                post_sat.append(val)

    survey = {
        "pre_confidence_avg": _avg(pre_conf),
        "post_satisfaction_avg": _avg(post_sat),
        "pre_factor_dist": {"CIV": 0, "MS": 0, "TP": 0, "IQ": 0},
        "post_effect_dist": {"RR": 0, "KS": 0, "IR": 0, "UN": 0},
    }

    for r in runs:
        for rf in r.get("reflections", []):
            cv = (rf.get("choice_value") or "").strip().upper()
            if rf.get("phase") == "pre" and cv in survey["pre_factor_dist"]:
                survey["pre_factor_dist"][cv] += 1
            if rf.get("phase") == "post" and cv in survey["post_effect_dist"]:
                survey["post_effect_dist"][cv] += 1

    # If decision-level LOAC metrics exist, prefer them over keyword heuristics.
    if loac_n > 0:
        loac_avg = {k: (loac_sum[k] / loac_n) for k in loac_sum.keys()}
        raw = loac_avg
        norm = loac_avg  # already 0..1

    def _avg01(xs):
        return sum(xs) / len(xs) if xs else 0.0

    emotional = {
        "avg": _avg01(emo_scores),
        "dist": emo_dist,
        "count": len(emo_scores)
    }

    return {
        "counts": {"runs": total_runs, "decisions": total_decisions, "reflections": total_reflections},
        "letter_counts": dict(letter_counts),
        "dimensions_raw": raw,
        "dimensions_norm": norm,
        "sentiment": sentiment,
        "emotional": emotional,
        "survey": survey,
    }


def compute_step_survey_stats(username: str) -> dict:
    """Aggregate modal step-survey answers across all runs for a user."""
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT s.confidence, s.sentiment, s.morality, COUNT(*) AS cnt
        FROM RunStepSurveys s
        JOIN Runs r ON r.id = s.run_id
        WHERE r.username = ?
        GROUP BY s.confidence, s.sentiment, s.morality
        """,
        (username,),
    )
    rows = cur.fetchall()
    conn.close()

    total = sum(int(row["cnt"]) for row in rows) if rows else 0
    if total <= 0:
        return {"count": 0}

    conf_score = {"very_sure": 1.0, "somewhat_sure": 0.66, "unsure": 0.33, "guessing": 0.0}
    sent_score = {"positive": 1.0, "neutral": 0.5, "mixed": 0.5, "negative": 0.0}
    moral_score = {"morally_right": 1.0, "morally_neutral": 0.5, "morally_wrong": 0.0, "unsure": 0.33}

    confidence_dist = {"very_sure": 0, "somewhat_sure": 0, "unsure": 0, "guessing": 0}
    sentiment_dist = {"positive": 0, "neutral": 0, "mixed": 0, "negative": 0}
    morality_dist = {"morally_right": 0, "morally_neutral": 0, "morally_wrong": 0, "unsure": 0}

    conf_sum = sent_sum = moral_sum = 0.0

    for row in rows:
        c = (row["confidence"] or "").strip()
        s = (row["sentiment"] or "").strip()
        m = (row["morality"] or "").strip()
        n = int(row["cnt"])

        conf_sum += conf_score.get(c, 0.0) * n
        sent_sum += sent_score.get(s, 0.0) * n
        moral_sum += moral_score.get(m, 0.0) * n

        if c in confidence_dist:
            confidence_dist[c] += n
        if s in sentiment_dist:
            sentiment_dist[s] += n
        if m in morality_dist:
            morality_dist[m] += n

    return {
        "count": total,
        "confidence_avg": round(conf_sum / total, 4),
        "sentiment_avg": round(sent_sum / total, 4),
        "morality_avg": round(moral_sum / total, 4),
        "confidence_dist": confidence_dist,
        "sentiment_dist": sentiment_dist,
        "morality_dist": morality_dist,
    }


def compute_emotional_stats(username: str) -> dict:
    """
    Uses RunDecisionMetrics (emotion_label + emotional_index) as the profile's
    emotion source. This works even if you didn't save RunReflections.
    """
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Average emotional index
    cur.execute(
        """
        SELECT AVG(m.emotional_index) AS avg_idx, COUNT(*) AS n
        FROM RunDecisionMetrics m
        JOIN Runs r ON r.id = m.run_id
        WHERE r.username = ?
        """,
        (username,),
    )
    row = cur.fetchone()
    n = int(row["n"] or 0)
    avg_idx = float(row["avg_idx"]) if row and row["avg_idx"] is not None else None

    # Distribution of emotion labels
    cur.execute(
        """
        SELECT LOWER(COALESCE(m.emotion_label, 'unknown')) AS lab, COUNT(*) AS cnt
        FROM RunDecisionMetrics m
        JOIN Runs r ON r.id = m.run_id
        WHERE r.username = ?
        GROUP BY LOWER(COALESCE(m.emotion_label, 'unknown'))
        """,
        (username,),
    )
    dist_rows = cur.fetchall()
    conn.close()

    dist = {}
    for rr in dist_rows:
        dist[rr["lab"]] = int(rr["cnt"])

    return {
        "count": n,
        "avg": None if avg_idx is None else round(float(avg_idx), 4),
        "dist": dist
    }


def count_user_runs(username: str) -> int:
    """Total number of Runs rows for this user (used for pagination)."""
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM Runs WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return int(row["n"] if row and row["n"] is not None else 0)


@app.route("/profile", methods=["GET"], endpoint="profile")
@login_required
def profile():
    username = session["username"]

    PER_PAGE = 5
    try:
        page = int(request.args.get("page", "1"))
    except ValueError:
        page = 1
    if page < 1:
        page = 1

    total_runs = count_user_runs(username)
    total_pages = max(1, (total_runs + PER_PAGE - 1) // PER_PAGE)
    if page > total_pages:
        page = total_pages

    offset = (page - 1) * PER_PAGE

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, gender, age_group, preferred_war, interest
        FROM Users
        WHERE username=?
        """,
        (username,),
    )
    urow = cur.fetchone()
    conn.close()

    user = dict(urow) if urow else {
        "username": username,
        "gender": "",
        "age_group": "",
        "preferred_war": "",
        "interest": ""
    }

    page_runs = hydrate_runs_with_details(username=username, limit=PER_PAGE, offset=offset)

    # Compute analytics from ALL runs (not just the current page)
    max_stats_runs = min(total_runs, 2000)
    all_runs = hydrate_runs_with_details(username=username, limit=max_stats_runs, offset=0)
    stats = compute_profile_stats(all_runs)
    stats["step_survey"] = compute_step_survey_stats(username)
    stats["emotional"] = compute_emotional_stats(username)

    # ✅ AJAX partial response (only mission log HTML)
    if request.args.get("partial") == "1":
        html = render_template(
            "_profile_mission_log.html",
            page_runs=page_runs,
            page=page,
            total_pages=total_pages,
            total_runs=total_runs,
            per_page=PER_PAGE
        )
        return jsonify({"html": html, "page": page, "total_pages": total_pages})

    return render_template(
        "profile.html",
        user=user,
        page_runs=page_runs,
        runs=page_runs,
        stats=stats,
        page=page,
        total_pages=total_pages,
        total_runs=total_runs,
        per_page=PER_PAGE
    )


def _log_step_survey(run_id: int, step_index: int, choice_value: str,
                     confidence: str, sentiment: str, morality: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """INSERT INTO RunStepSurveys(run_id, step_index, choice_value, confidence, sentiment, morality)
           VALUES(?,?,?,?,?,?)""",
        (run_id, step_index, choice_value, confidence, sentiment, morality),
    )
    conn.commit()
    conn.close()


@app.route("/api/run/step-survey", methods=["POST"])
@login_required
def api_run_step_survey():
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "No active run in session."}), 400

    data = request.get_json(silent=True) or {}
    try:
        step_index = int(data.get("step_index"))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid step_index."}), 400

    choice_value = (data.get("choice_value") or "").strip()
    confidence = (data.get("confidence") or "").strip()
    sentiment = (data.get("sentiment") or "").strip()
    morality = (data.get("morality") or "").strip()

    allowed_conf = {"very_sure", "somewhat_sure", "unsure", "guessing"}
    allowed_sent = {"positive", "neutral", "negative", "mixed"}
    allowed_moral = {"morally_right", "morally_neutral", "morally_wrong", "unsure"}

    if confidence not in allowed_conf:
        return jsonify({"ok": False, "error": "Invalid confidence option."}), 400
    if sentiment not in allowed_sent:
        return jsonify({"ok": False, "error": "Invalid sentiment option."}), 400
    if morality not in allowed_moral:
        return jsonify({"ok": False, "error": "Invalid morality option."}), 400

    _log_step_survey(
        run_id=run_id,
        step_index=step_index,
        choice_value=choice_value,
        confidence=confidence,
        sentiment=sentiment,
        morality=morality
    )
    return jsonify({"ok": True})


@app.route("/profile/update", methods=["POST"], endpoint="profile_update")
@login_required
def profile_update():
    username = session["username"]

    gender = request.form.get("gender") or None
    age_group = request.form.get("age_group") or None
    preferred_war = (request.form.get("preferred_war") or "").strip() or None
    interest = (request.form.get("interest") or "").strip() or None
    new_password = (request.form.get("new_password") or "").strip()

    conn = get_db()
    c = conn.cursor()

    if new_password:
        c.execute("""
          UPDATE Users
          SET gender=?, age_group=?, preferred_war=?, interest=?, password=?
          WHERE username=?
        """, (gender, age_group, preferred_war, interest, new_password, username))
    else:
        c.execute("""
          UPDATE Users
          SET gender=?, age_group=?, preferred_war=?, interest=?
          WHERE username=?
        """, (gender, age_group, preferred_war, interest, username))

    conn.commit()
    conn.close()

    flash("Profile updated.", "success")
    return redirect(url_for("profile"))


@app.route("/api/persona", methods=["GET"])
@login_required
def api_persona():
    username = session["username"]
    TTL_SECONDS = 6 * 60 * 60

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT persona_text, updated_at FROM UserPersonaCache WHERE username=?", (username,))
    row = c.fetchone()

    if row:
        try:
            updated_at = datetime.fromisoformat(row["updated_at"])
        except Exception:
            updated_at = None

        if updated_at:
            age = (datetime.utcnow() - updated_at).total_seconds()
            if age < TTL_SECONDS:
                conn.close()
                return jsonify({"persona": row["persona_text"], "cached": True})

    conn.close()

    runs = hydrate_runs_with_details(username=username, limit=2000, offset=0)
    stats = compute_profile_stats(runs)
    persona = generate_user_persona(stats)

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "REPLACE INTO UserPersonaCache(username, persona_text, updated_at) VALUES(?,?,?)",
        (username, persona, datetime.utcnow().isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()

    return jsonify({"persona": persona, "cached": False})


# ===== Admin: Research (ONLY research runs, NOT GPT) =====
@app.route("/admin/research", methods=["GET"])
@admin_required
def admin_research():
    return render_template("admin_research.html")


@app.route("/admin/api/users", methods=["GET"])
@admin_required
def admin_api_users():
    q = (request.args.get("q") or "").strip().lower()
    conn = get_db()
    c = conn.cursor()

    sql = """
      SELECT U.username, U.gender, U.age_group,
             COUNT(R.id) AS research_runs,
             MAX(COALESCE(R.finished_at, R.started_at)) AS last_activity
      FROM Users U
      JOIN Runs R ON R.username = U.username
      WHERE R.run_type = 'research'
    """
    params = []
    if q:
        sql += " AND LOWER(U.username) LIKE ?"
        params.append(f"%{q}%")
    sql += " GROUP BY U.username ORDER BY last_activity DESC"

    c.execute(sql, params)
    users = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({"users": users})


import json
import sqlite3
from typing import List


def hydrate_runs_with_details(username: str, limit: int = 12, offset: int = 0) -> list:
    """
    Returns Runs for a user, newest-first, with:
      - run['decisions']    : list of decisions for that run (ordered by step_index ASC)
      - run['reflections']  : list of reflections for that run (ordered by step_index ASC, phase ASC)

    Pagination:
      - limit  : number of runs to return
      - offset : how many runs to skip (for page N => offset=(N-1)*limit)
    """
    limit = max(1, int(limit or 12))
    offset = max(0, int(offset or 0))

    conn = get_db()
    cur = conn.cursor()

    # 1) Fetch runs (newest-first)
    cur.execute(
        """
        SELECT id, username, run_type, sid, title, total_steps, started_at, finished_at, prefs_json
        FROM Runs
        WHERE username=?
        ORDER BY
          CASE WHEN finished_at IS NULL OR finished_at='' THEN 0 ELSE 1 END DESC,
          COALESCE(finished_at, started_at) DESC,
          id DESC
        LIMIT ? OFFSET ?
        """,
        (username, limit, offset),
    )
    run_rows = cur.fetchall()

    if not run_rows:
        conn.close()
        return []

    runs = []
    run_ids = []
    for row in run_rows:
        r = dict(row)
        run_ids.append(r["id"])

        # parse prefs_json safely
        prefs = {}
        try:
            if r.get("prefs_json"):
                prefs = json.loads(r["prefs_json"])
        except Exception:
            prefs = {}

        r["prefs"] = prefs
        r["decisions"] = []
        r["reflections"] = []
        runs.append(r)

    # Build lookup
    by_id = {r["id"]: r for r in runs}

    # 2) Hydrate decisions for these runs
    placeholders = ",".join(["?"] * len(run_ids))
    cur.execute(
        f"""
        SELECT run_id, step_index, option_value, option_label, option_consequence, created_at
        FROM RunDecisions
        WHERE run_id IN ({placeholders})
        ORDER BY run_id ASC, step_index ASC, id ASC
        """,
        run_ids,
    )
    for drow in cur.fetchall():
        d = dict(drow)
        rid = d.get("run_id")
        if rid in by_id:
            step_idx = d.get("step_index") or 0
            # keep the key names your template expects
            by_id[rid]["decisions"].append(
                {
                    "step_index": step_idx,
                    "step_title": f"Step {step_idx}",
                    "chosen_value": d.get("option_value"),
                    "chosen_label": d.get("option_label") or d.get("option_value") or "",
                    "chosen_consequence": d.get("option_consequence") or "",
                    "created_at": d.get("created_at"),
                }
            )

    # 2b) Hydrate decision metrics (LOAC/strategy/emotion) for these runs (best-effort)
    cur.execute(
        f"""
        SELECT run_id, step_index, loac_json, strategy_json, emotional_index, emotion_label
        FROM RunDecisionMetrics
        WHERE run_id IN ({placeholders})
        """,
        run_ids,
    )
    metrics_map = {}
    for mrow in cur.fetchall():
        md = dict(mrow)
        key = (md.get("run_id"), md.get("step_index"))
        loac = {}
        strat = {}
        try:
            if md.get("loac_json"):
                loac = json.loads(md["loac_json"])
        except Exception:
            loac = {}
        try:
            if md.get("strategy_json"):
                strat = json.loads(md["strategy_json"])
        except Exception:
            strat = {}

        metrics_map[key] = {
            "loac": loac,
            "strategy": strat,
            "emotional_index": md.get("emotional_index"),
            "emotion_label": md.get("emotion_label"),
        }

    # Attach metrics onto each decision dict
    for rid, run in by_id.items():
        for d in run.get("decisions", []):
            key = (rid, d.get("step_index"))
            if key in metrics_map:
                d.update(metrics_map[key])

    # 3) Hydrate reflections for these runs (if any)
    cur.execute(
        f"""
        SELECT run_id, step_index, phase, question_text, response_text,
               sentiment_score, sentiment_label, choice_value, choice_label, created_at
        FROM RunReflections
        WHERE run_id IN ({placeholders})
        ORDER BY run_id ASC, step_index ASC, phase ASC, id ASC
        """,
        run_ids,
    )
    for rrow in cur.fetchall():
        rr = dict(rrow)
        rid = rr.get("run_id")
        if rid in by_id:
            by_id[rid]["reflections"].append(rr)

    conn.close()
    return runs


@app.route("/admin/api/user_detail/<username>", methods=["GET"])
@admin_required
def admin_api_user_detail(username):
    runs = hydrate_runs_with_details(username=username, limit=5)

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT persona_text FROM UserPersonaCache WHERE username=?", (username,))
    row = c.fetchone()
    persona = row["persona_text"] if row else "Identity analysis in progress..."
    conn.close()

    return jsonify({
        "username": username,
        "persona": persona,
        "runs": runs
    })


# --- Add this near your existing step survey route(s) ---
@app.route("/api/research/step-survey", methods=["POST"])
@login_required
def api_research_step_survey():
    """
    Research/static page endpoint.
    This matches scenario_step_static.html which posts to /api/research/step-survey.
    Internally we log the same RunStepSurveys table as /api/run/step-survey.
    """
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "No active run in session."}), 400

    data = request.get_json(silent=True) or {}

    try:
        step_index = int(data.get("step_index"))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid step_index."}), 400

    choice_value = (data.get("choice_value") or "").strip()
    confidence = (data.get("confidence") or "").strip()
    sentiment = (data.get("sentiment") or "").strip()
    morality = (data.get("morality") or "").strip()

    allowed_conf = {"very_sure", "somewhat_sure", "unsure", "guessing"}
    allowed_sent = {"positive", "neutral", "negative", "mixed"}
    allowed_moral = {"morally_right", "morally_neutral", "morally_wrong", "unsure"}

    if confidence not in allowed_conf:
        return jsonify({"ok": False, "error": "Invalid confidence option."}), 400
    if sentiment not in allowed_sent:
        return jsonify({"ok": False, "error": "Invalid sentiment option."}), 400
    if morality not in allowed_moral:
        return jsonify({"ok": False, "error": "Invalid morality option."}), 400

    _log_step_survey(
        run_id=run_id,
        step_index=step_index,
        choice_value=choice_value,
        confidence=confidence,
        sentiment=sentiment,
        morality=morality
    )

    return jsonify({"ok": True})


@app.route("/api/research/static-decision", methods=["POST"])
@login_required
def api_research_static_decision():
    """
    Called by scenario_step_static.html before form submit to ensure decisions are mirrored
    to research_runs / research_decisions, even if the normal POST flow is interrupted.
    """
    payload = request.get_json(silent=True) or {}
    scenario_key = (payload.get("scenario_key") or "").strip()
    step_id = int(payload.get("step_id") or 0)
    option_value = (payload.get("option_value") or "").strip()

    if not scenario_key or step_id <= 0 or not option_value:
        return jsonify({"ok": False, "error": "Missing scenario_key/step_id/option_value"}), 400

    ok, err = _upsert_static_research_decision(
        username=session.get("username"),
        user_id=session.get("user_id"),
        scenario_key=scenario_key,
        step_id=step_id,
        option_value=option_value,
        chosen_label_override=(payload.get("chosen_label") or ""),
        scores_override=(payload.get("scores") if isinstance(payload.get("scores"), dict) else None)
    )
    if not ok:
        return jsonify({"ok": False, "error": err or "save failed"}), 500
    return jsonify({"ok": True})


@app.route("/api/research/gpt-decision", methods=["POST"])
@login_required
def api_research_gpt_decision():
    """Mirror GPT decisions into research_* tables (source='gpt').

    Called by scenario_step_gpt.html before form submit so the research log and report
    structure matches scenario_step_static.
    """
    payload = request.get_json(silent=True) or {}
    step_id = int(payload.get("step_id") or 0)
    option_value = (payload.get("option_value") or "").strip()

    run_id = session.get("run_id")
    if not run_id or step_id <= 0 or not option_value:
        return jsonify({"ok": False, "error": "Missing run_id/step_id/option_value"}), 400

    ok, err = _upsert_gpt_research_decision(
        username=session.get("username"),
        user_id=session.get("user_id"),
        run_id=int(run_id),
        step_id=step_id,
        option_value=option_value
    )
    if not ok:
        return jsonify({"ok": False, "error": err or "save failed"}), 500
    return jsonify({"ok": True})


@app.route("/admin/api/research_stats", methods=["GET"])
@admin_required
def admin_api_research_stats():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
      SELECT COUNT(DISTINCT username)
      FROM research_runs
      WHERE source='static'
    """)
    users_with_any = c.fetchone()[0] or 0

    c.execute("""
      SELECT COUNT(DISTINCT username)
      FROM research_runs
      WHERE source='static' AND finished_at IS NOT NULL
    """)
    completed_users = c.fetchone()[0] or 0

    c.execute("""
      SELECT username
      FROM research_runs
      WHERE source='static'
      ORDER BY COALESCE(finished_at, started_at) DESC
      LIMIT 1
    """)
    row = c.fetchone()
    recent_user = row[0] if row else "None"

    c.execute("""
      SELECT rr.username, rr.id as run_id
      FROM research_runs rr
      JOIN (
        SELECT username, MAX(finished_at) AS mx
        FROM research_runs
        WHERE source='static' AND finished_at IS NOT NULL
        GROUP BY username
      ) t ON t.username=rr.username AND t.mx=rr.finished_at
      WHERE rr.source='static' AND rr.finished_at IS NOT NULL
    """)
    latest_finished = c.fetchall()

    path_counts = {}
    for r in latest_finished:
        run_id = r["run_id"]
        c.execute("""
          SELECT chosen_letter
          FROM research_decisions
          WHERE run_id=? AND source='static'
          ORDER BY step_id ASC
        """, (run_id,))
        letters = "".join([x[0] for x in c.fetchall() if x and x[0]])
        if letters:
            path_counts[letters] = path_counts.get(letters, 0) + 1

    popular_path = max(path_counts, key=path_counts.get) if path_counts else None

    c.execute("""
      SELECT rd.run_id, MAX(rd.step_id) AS last_step
      FROM research_decisions rd
      JOIN research_runs rr ON rr.id=rd.run_id
      WHERE rr.source='static' AND rr.finished_at IS NOT NULL AND rd.source='static'
      GROUP BY rd.run_id
    """)
    last_steps = c.fetchall()

    final_dim_sum = {"Distinction": 0.0, "Proportionality": 0.0, "Necessity": 0.0, "Precaution": 0.0}
    final_total_sum = 0.0
    final_n = 0

    for ls in last_steps:
        run_id = ls["run_id"]
        last_step = ls["last_step"]
        c.execute("""
          SELECT scores_json, ethics_index
          FROM research_decisions
          WHERE run_id=? AND step_id=? AND source='static'
          LIMIT 1
        """, (run_id, last_step))
        rr = c.fetchone()
        if not rr:
            continue
        try:
            scores = json.loads(rr["scores_json"] or "{}")
        except Exception:
            scores = {}
        final_total_sum += float(rr["ethics_index"] or 0.0)
        for k in final_dim_sum.keys():
            final_dim_sum[k] += float(scores.get(k, 0) or 0)
        final_n += 1

    avg_dims = {k: (final_dim_sum[k] / final_n if final_n else 0.0) for k in final_dim_sum.keys()}
    avg_final_ethics_index = (final_total_sum / final_n) if final_n else 0.0

    c.execute("""
      SELECT U.age_group, COUNT(*) as cnt
      FROM Users U
      JOIN (
        SELECT DISTINCT username
        FROM research_runs
        WHERE source='static' AND finished_at IS NOT NULL
      ) t ON t.username = U.username
      GROUP BY U.age_group
      ORDER BY cnt DESC
    """)
    by_age = {(r[0] or "Unknown"): (r[1] or 0) for r in c.fetchall()}

    conn.close()

    return jsonify({
        "counts": {
            "users_with_any_research": users_with_any,
            "completed_users": completed_users
        },
        "recent_user": recent_user,
        "avg_final_ethics_index": round(avg_final_ethics_index, 2),
        "popular_path": popular_path,
        "avg_dims": {k: round(v, 2) for k, v in avg_dims.items()},
        "by_age": by_age
    })


@app.post("/research/static/save_decision")
@login_required
def save_static_research_decision():
    username = session.get("username")
    user_id = session.get("user_id")

    body = request.get_json(silent=True) or {}
    scenario_key = (request.form.get("scenario_key") or body.get("scenario_key") or "").strip()
    step_id_raw = request.form.get("step_id") or body.get("step_id")
    option_value = (request.form.get("option_value") or body.get("option_value") or "").strip()

    if not scenario_key or not option_value or step_id_raw is None:
        return jsonify({"ok": False, "error": "Missing scenario_key / step_id / option_value"}), 400

    try:
        step_id = int(step_id_raw)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid step_id"}), 400

    scenarios = load_scenarios_for_research()
    _, opt = find_option(scenarios, scenario_key, step_id, option_value)
    if not opt:
        return jsonify({"ok": False, "error": "Option not found in scenarios.json"}), 400

    chosen_letter = (option_value[0] if option_value else "").upper()
    chosen_label = opt.get("label", "")

    scores = opt.get("scores") or {}
    if not isinstance(scores, dict):
        scores = {}

    ethics_index = compute_ethics_index(scores)
    now_ts = int(datetime.now().timestamp())

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id FROM research_runs
        WHERE username=? AND scenario_key=? AND source='static' AND finished_at IS NULL
        ORDER BY started_at DESC LIMIT 1
    """, (username, scenario_key))
    row = cur.fetchone()

    if row:
        run_id = row["id"]
    else:
        cur.execute("""
            INSERT INTO research_runs(user_id, username, scenario_key, started_at, source, group_label)
            VALUES(?,?,?,?, 'static', ?)
        """, (user_id, username, scenario_key, now_ts, group_label))
        run_id = cur.lastrowid

    if group_label:
        try:
            cur.execute("UPDATE research_runs SET group_label=? WHERE id=?", (group_label, run_id))
        except Exception:
            pass

    payload_scores_json = json.dumps(scores, ensure_ascii=False)

    cur.execute("""
        SELECT id FROM research_decisions
        WHERE run_id=? AND step_id=? AND source='static'
        LIMIT 1
    """, (run_id, step_id))
    existing = cur.fetchone()

    if existing:
        cur.execute("""
            UPDATE research_decisions
            SET option_value=?, chosen_letter=?, chosen_label=?,
                scores_json=?, ethics_index=?, created_at=?, group_label=?
            WHERE id=?
        """, (option_value, chosen_letter, chosen_label,
              payload_scores_json, ethics_index, now_ts, group_label, existing["id"]))
    else:
        cur.execute("""
            INSERT INTO research_decisions(
              run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              scores_json, ethics_index, created_at, group_label, source
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'static')
        """, (run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              payload_scores_json, ethics_index, now_ts, group_label))

    total_steps = len((scenarios.get(scenario_key) or {}).get("steps", [])) or 4
    if step_id >= total_steps:
        cur.execute("""
            UPDATE research_runs SET finished_at=?
            WHERE id=? AND finished_at IS NULL
        """, (now_ts, run_id))

    conn.commit()
    conn.close()

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "saved": {
            "scenario_key": scenario_key,
            "step_id": step_id,
            "option_value": option_value,
            "ethics_index": ethics_index,
            "scores": scores
        }
    })


@app.get("/admin/api/research_trajectory")
@admin_required
def admin_research_trajectory():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
      SELECT r1.*
      FROM research_runs r1
      JOIN (
        SELECT username, MAX(COALESCE(finished_at, started_at)) AS mx
        FROM research_runs
        WHERE source='static' AND finished_at IS NOT NULL
        GROUP BY username
      ) t
      ON t.username=r1.username AND t.mx=COALESCE(r1.finished_at, r1.started_at)
      WHERE r1.source='static' AND r1.finished_at IS NOT NULL
      ORDER BY COALESCE(r1.finished_at, r1.started_at) DESC
      LIMIT 200
    """)
    runs = cur.fetchall()

    all_data = []
    path_counts = {}

    for r in runs:
        run_id = r["id"]
        username = r["username"]

        cur.execute("""
          SELECT step_id, chosen_letter, chosen_label, ethics_index, scores_json
          FROM research_decisions
          WHERE run_id=? AND source='static'
          ORDER BY step_id ASC
        """, (run_id,))
        decs = cur.fetchall()

        path_letters = "".join([d["chosen_letter"] for d in decs if d["chosen_letter"]])
        if path_letters:
            path_counts[path_letters] = path_counts.get(path_letters, 0) + 1

        path_points = []
        for d in decs:
            try:
                scores = json.loads(d["scores_json"] or "{}")
            except Exception:
                scores = {}

            path_points.append({
                "step_id": d["step_id"],
                "step_title": f"Step {d['step_id']}",
                "chosen_letter": d["chosen_letter"],
                "chosen_label": d["chosen_label"],
                "scores": scores,
                "total": float(d["ethics_index"] or 0.0)
            })

        all_data.append({
            "username": username,
            "scenario_key": r["scenario_key"],
            "path_letters": path_letters,
            "path": path_points
        })

    top_paths = sorted(
        [{"path": k, "count": v} for k, v in path_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    conn.close()
    return jsonify({
        "all_data": all_data,
        "top_paths": top_paths
    })


# =========================
# Library (shared GPT scenarios)
# =========================

def _get_public_gpt_scenarios(limit: int = 60, q: str = "", exclude_username: Optional[str] = None):
    """
    Returns GPT runs that have stored sequences (RunSequences), for display in /library.
    """
    conn = get_db()
    c = conn.cursor()

    # Only runs that have sequence_json
    sql = """
      SELECT
        R.id AS run_id,
        R.username AS author,
        R.title,
        R.started_at,
        R.finished_at,
        R.prefs_json,
        S.sequence_json
      FROM Runs R
      JOIN RunSequences S ON S.run_id = R.id
      WHERE R.run_type='gpt'
    """
    params = []

    if exclude_username:
        sql += " AND R.username <> ?"
        params.append(exclude_username)

    if q:
        # lightweight search in title/author (and optionally prefs_json)
        sql += " AND (LOWER(R.title) LIKE ? OR LOWER(R.username) LIKE ? OR LOWER(R.prefs_json) LIKE ?)"
        qq = f"%{q.lower()}%"
        params.extend([qq, qq, qq])

    # Prefer finished ones first (optional), then newest
    sql += """
      ORDER BY
        CASE WHEN R.finished_at IS NULL THEN 1 ELSE 0 END,
        COALESCE(R.finished_at, R.started_at) DESC
      LIMIT ?
    """
    params.append(int(limit))

    c.execute(sql, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    # parse sequence and prefs safely
    out = []
    for r in rows:
        seq = _safe_json_loads(r.get("sequence_json"), {})
        prefs = _safe_json_loads(r.get("prefs_json"), {})
        if not isinstance(seq, dict) or not seq.get("steps"):
            continue
        out.append({
            "run_id": r["run_id"],
            "author": r["author"],
            "title": r.get("title") or (seq.get("title") or "GPT Scenario"),
            "intro": (seq.get("intro") or "")[:220] + ("..." if (seq.get("intro") or "")[220:] else ""),
            "steps_count": len(seq.get("steps") or []),
            "created_at": r.get("finished_at") or r.get("started_at"),
            "created_at_epoch": _iso_to_epoch(r.get("finished_at") or r.get("started_at")),
            "prefs": prefs,
        })
    return out


def _load_sequence_by_run_id(run_id: int) -> Optional[dict]:
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT sequence_json FROM RunSequences WHERE run_id=?", (run_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return _safe_json_loads(row["sequence_json"], None)


def _load_run_prefs_by_run_id(run_id: int) -> dict:
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT prefs_json FROM Runs WHERE id=?", (run_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {}
    return _safe_json_loads(row["prefs_json"], {})


@app.route("/library", methods=["GET"])
@login_required
def library():
    q = (request.args.get("q") or "").strip()
    # If you want users to see their own too, remove exclude_username
    scenarios = _get_public_gpt_scenarios(limit=80, q=q, exclude_username=None)

    return render_template("library.html", scenarios=scenarios, q=q)
def _get_scenario_image(scenario_key: str, prefix: str, step_index: int) -> Optional[dict]:
    if not scenario_key:
        return None
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """SELECT scenario_key, prefix, step_index, image_path, prompt_hash
           FROM ScenarioImages
           WHERE scenario_key=? AND prefix=? AND step_index=?""",
        (scenario_key, prefix or "", int(step_index)),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def _save_scenario_image(scenario_key: str, prefix: str, step_index: int, image_path_rel_to_static: str, prompt_hash: str):
    if not scenario_key:
        return
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """REPLACE INTO ScenarioImages(scenario_key, prefix, step_index, image_path, prompt_hash)
           VALUES(?,?,?,?,?)""",
        (scenario_key, prefix or "", int(step_index), image_path_rel_to_static, prompt_hash),
    )
    conn.commit()
    conn.close()
def _generate_and_store_step_image_for_path(
        *,
        scenario_key: str,
        prefix: str,
        step_index: int,
        seq: dict,
        prefs: dict,
        step_obj: dict,
        story_so_far: str
) -> Optional[str]:
    """
    Cache images by (scenario_key, prefix, step_index).
    Reuses the same image across different runs for the same unique path.
    """
    if not _client:
        return None

    scenario_key = (scenario_key or "").strip()
    prefix = (prefix or "").strip()
    step_index = int(step_index)

    prompt = _build_gpt_image_prompt(seq, prefs, step_obj, story_so_far)
    ph = _sha256(prompt)

    existing = _get_scenario_image(scenario_key, prefix, step_index)
    if existing and existing.get("image_path") and existing.get("prompt_hash") == ph:
        # also confirm file exists; if missing, regenerate
        abs_path = os.path.join(app.static_folder, existing["image_path"].lstrip("/"))
        if os.path.exists(abs_path):
            return "/static/" + existing["image_path"].lstrip("/")

    # stable filename by key+prefix+step
    key_hash = _sha256(f"{scenario_key}||{prefix}")[:16]
    out_dir = os.path.join(app.static_folder, "generated")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"sc_{key_hash}_step_{step_index}.png"
    abs_out = os.path.join(out_dir, filename)

    try:
        img = _client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )
        b64 = _extract_b64_from_image_response(img)
        if not b64:
            return None

        png_bytes = base64.b64decode(b64)
        with open(abs_out, "wb") as f:
            f.write(png_bytes)

        rel_path = os.path.join("generated", filename).replace("\\", "/")
        _save_scenario_image(scenario_key, prefix, step_index, rel_path, ph)
        return "/static/" + rel_path
    except Exception as e:
        print("[image][path-cache] generation failed:", e)
        return None


@app.route("/library/try/<int:source_run_id>", methods=["POST"])
@login_required
def library_try(source_run_id: int):
    """
    Fork someone else's GPT scenario into a new run owned by the current user,
    then start at step 1 using the normal gpt_scenario_step route.
    """
    src_seq = _load_sequence_by_run_id(source_run_id)
    if not src_seq:
        flash("This scenario is unavailable.", "warning")
        return redirect(url_for("library"))

    # validate/coerce to your expected GPT schema
    try:
        # If it already has the correct format, keep it; otherwise coerce.
        seq = _coerce_gpt_sequence(src_seq)
        _validate_generated_sequence_gpt(seq)
    except Exception:
        flash("This scenario format is invalid.", "danger")
        return redirect(url_for("library"))

    # Keep original prefs if available (for image prompts). Users can still play text-only.
    src_prefs = _load_run_prefs_by_run_id(source_run_id)
    # Force make_image based on current user's preference (session checkbox)
    make_image = bool(session.get("make_image", False))
    src_prefs = dict(src_prefs or {})
    src_prefs["make_image"] = make_image

    # Create a NEW run for current user
    session["mode"] = "gpt"
    session["scenario_sid"] = "G"
    session["scenario_progress"] = []
    session.pop("step_start_time", None)

    new_title = (seq.get("title") or "GPT Scenario").strip()
    new_title = f"{new_title} (from Library)"

    run_id = _start_run(
        run_type="gpt",
        sid="G",
        title=new_title,
        total_steps=len(seq.get("steps", [])),
        prefs=src_prefs
    )
    _save_run_sequence(run_id, seq)

    # IMPORTANT: do NOT carry over images; lazy generation will regenerate for this run_id
    # (RunImages is keyed by run_id, so it's naturally separate.)

    return redirect(url_for("gpt_scenario_step", step=1))


# ===== GPT STEP-BY-STEP GENERATION (NEW) =====

TOTAL_GPT_STEPS = 5


def _get_run_prefs(run_id: int) -> dict:
    if not run_id:
        return {}
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT prefs_json FROM Runs WHERE id=?", (run_id,))
        row = c.fetchone()
        conn.close()
        return json.loads(row["prefs_json"] or "{}") if row else {}
    except Exception:
        return {}


def _role_context_from_prefs(prefs: dict) -> str:
    role = (prefs.get("role") or "").strip()
    if role == "student":
        return (
            "A high school student living in a conflict zone. Focus on dilemmas involving "
            "attending school, protecting fellow students, and managing limited educational resources "
            "or the use of school buildings by military forces."
        )
    if role == "medic":
        return "A field medic focusing on triage and medical ethics."
    if role == "war-reporter":
        return "A war reporter focusing on truth, safety, and harm minimization."
    if role == "humanitarian-coordinator":
        return "A humanitarian relief coordinator balancing neutrality and access."
    return "A field commander balancing military necessity and civilian protection."


def generate_gpt_initial_sequence_with_llm(prefs: dict) -> dict:
    """
    Generates ONLY:
      - title
      - intro
      - step 1
    and stores a sequence object that will grow as the user makes choices.
    """
    if not _client:
        raise RuntimeError("OPENAI_API_KEY not set")

    role_context = _role_context_from_prefs(prefs)

    system_prompt = (
        "Return STRICT JSON ONLY (no markdown, no explanations).\n"
        "Generate an ethics-in-war decision scenario aligned with International Humanitarian Law.\n"
        "No graphic violence.\n\n"
        "You must return ONLY the scenario title/intro and Step 1.\n"
        f"Total steps for this scenario will be {TOTAL_GPT_STEPS}, but you generate ONLY step 1 now.\n\n"
        "Rules for Step 1:\n"
        "1) Step 1 MUST have EXACTLY 4 options.\n"
        "2) Option values MUST be EXACTLY: A1, B1, C1, D1.\n"
        "3) Each option MUST include label and consequence.\n"
        "4) Situation should be 3–5 sentences, age-appropriate.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "id":"G",\n'
        '  "title":"...",\n'
        '  "intro":"...",\n'
        '  "steps":[\n'
        "    {\n"
        '      "id":1,\n'
        '      "title":"...",\n'
        '      "situation":"...",\n'
        '      "question":"...",\n'
        '      "options":[\n'
        '        {"value":"A1","label":"...","consequence":"..."},\n'
        '        {"value":"B1","label":"...","consequence":"..."},\n'
        '        {"value":"C1","label":"...","consequence":"..."},\n'
        '        {"value":"D1","label":"...","consequence":"..."}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user_content = (
        f"Conflict: {prefs.get('war')}.\n"
        f"Front/Theatre: {prefs.get('theatre') or 'auto'}.\n"
        f"Role: {role_context}.\n"
        f"Tone: {prefs.get('tone')}.\n"
        f"Learning Goal: {prefs.get('goal')}.\n"
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    content = _strip_json_fences(resp.choices[0].message.content or "{}")
    raw = json.loads(content)

    # Coerce + normalize step 1 using your existing normalizer
    seq = {
        "id": "G",
        "title": (raw.get("title") or "Ethics-in-War Scenario").strip(),
        "intro": (raw.get("intro") or "").strip(),
        "total_steps": TOTAL_GPT_STEPS,
        "steps": []
    }

    step1_in = (raw.get("steps") or [{}])[0] if isinstance(raw.get("steps"), list) else {}
    st = dict(step1_in) if isinstance(step1_in, dict) else {}
    st["id"] = 1
    st["title"] = (st.get("title") or "Step 1").strip()
    st["situation"] = (st.get("situation") or "").strip()
    st["question"] = (st.get("question") or "What do you do?").strip()
    st["options"] = _normalize_options_to_4(1, st.get("options"))

    seq["steps"] = [st]
    return seq


def generate_gpt_next_step_with_llm(
        prefs: dict,
        story_so_far: str,
        last_choice_value: str,
        step_index: int,
        admin_direction: str = ""
) -> dict:
    """
    Generates ONE step (step_index) given story so far + last choice.
    If admin_direction is provided, it steers the regeneration of the current step.
    """
    if not _client:
        raise RuntimeError("OPENAI_API_KEY not set")
    if step_index < 2 or step_index > TOTAL_GPT_STEPS:
        raise ValueError("Invalid step_index for next-step generation.")

    role_context = _role_context_from_prefs(prefs)
    expected_vals = [f"A{step_index}", f"B{step_index}", f"C{step_index}", f"D{step_index}"]

    admin_direction = (admin_direction or "").strip()
    admin_block = ""
    if admin_direction:
        admin_block = (
            "\n\nADMIN GUIDANCE:\n"
            "- Follow this guidance when generating THIS step.\n"
            "- Do not rewrite earlier steps.\n"
            f"- Guidance: {admin_direction}\n"
        )

    system_prompt = (
        "Return STRICT JSON ONLY.\n"
        "Generate EXACTLY ONE ethics dilemma step that logically follows the previous decision.\n"
        "No graphic violence.\n\n"
        "Rules:\n"
        f"1) This is step {step_index} of {TOTAL_GPT_STEPS}.\n"
        "2) EXACTLY 4 options.\n"
        f"3) Option values MUST be EXACTLY: {', '.join(expected_vals)}.\n"
        "4) Each option MUST include label and consequence.\n"
        "5) Situation should be 3–5 sentences.\n\n"
        "JSON schema:\n"
        "{\n"
        f'  "id": {step_index},\n'
        '  "title":".",\n'
        '  "situation":".",\n'
        '  "question":".",\n'
        '  "options":[\n'
        f'    {{"value":"A{step_index}","label":".","consequence":"."}},\n'
        f'    {{"value":"B{step_index}","label":".","consequence":"."}},\n'
        f'    {{"value":"C{step_index}","label":".","consequence":"."}},\n'
        f'    {{"value":"D{step_index}","label":".","consequence":"."}}\n'
        "  ]\n"
        "}\n"
    )

    user_content = (
        f"Role context: {role_context}\n"
        f"Tone: {(prefs.get('tone') or 'serious & age-appropriate').strip()}\n"
        f"Scenario context (war/theatre/goal): war={prefs.get('war', '')}, theatre={prefs.get('theatre', '')}, goal={prefs.get('goal', '')}\n\n"
        "CANONICAL HISTORY (read carefully; do not contradict):\n"
        f"{(story_so_far or '').strip()}\n\n"
        "PREVIOUS DECISION (this is the only branch taken):\n"
        f"{(last_choice_value or '').strip()}\n\n"
        f"Now generate step {step_index}.\n"
        "Hard requirements:\n"
        "- The new situation MUST be a direct consequence of the previous decision above.\n"
        "- Do NOT reuse content from alternative branches that were not chosen.\n"
        "- Keep names, locations, and stakes consistent with the canonical history.\n"
        f"{admin_block}"
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    content = _strip_json_fences(resp.choices[0].message.content or "{}")
    raw = json.loads(content)

    # Normalize/coerce
    st = dict(raw) if isinstance(raw, dict) else {}
    st["id"] = step_index
    st["title"] = (st.get("title") or f"Step {step_index}").strip()
    st["situation"] = (st.get("situation") or "").strip()
    st["question"] = (st.get("question") or "What do you do?").strip()
    st["options"] = _normalize_options_to_4(step_index, st.get("options"))

    return st


import sqlite3, json
from datetime import datetime, timezone


def ensure_admin_tables():
    db = db_conn()
    cur = db.cursor()

    # Stores scenario "templates" (prefs + base seq containing Step 1)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS gpt_templates (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT,
      prefs_json TEXT NOT NULL,
      base_seq_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
    """)

    # Stores approved generated steps per choice-path
    cur.execute("""
    CREATE TABLE IF NOT EXISTS gpt_branch_steps (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      template_id INTEGER NOT NULL,
      path TEXT NOT NULL,
      step_index INTEGER NOT NULL,
      step_json TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(template_id, path, step_index),
      FOREIGN KEY(template_id) REFERENCES gpt_templates(id)
    )
    """)

    db.commit()
    db.close()


def _pick_scenarios_json_path() -> str:
    """
    Choose the actual scenarios.json file we will edit.
    Priority:
      1) first existing path in SCENARIOS_JSON_PATHS
      2) default to static/scenarios/scenarios.json
    """
    for p in SCENARIOS_JSON_PATHS:
        if os.path.exists(p):
            return p
    # default
    p = os.path.join("static", "scenarios", "scenarios.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _save_scenarios_to_disk(schema: dict):
    path = _pick_scenarios_json_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def _reload_scenarios_global():
    """
    Reload the global SCENARIO_SEQUENCES after edits.
    """
    global SCENARIO_SEQUENCES
    SCENARIO_SEQUENCES = load_json_first(SCENARIOS_JSON_PATHS, required=True)
    validate_scenarios(SCENARIO_SEQUENCES)


# =========================
# Admin: Active Scenario + Library Picker APIs
# =========================

def _ensure_app_config():
    try:
        db = sqlite3.connect(DB_PATH)
        cur = db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_config (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
        """)
        db.commit()
        db.close()
    except Exception as e:
        print("[app_config] ensure failed:", e)


def _get_cfg(key: str, default: str = "") -> str:
    _ensure_app_config()
    try:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        cur = db.cursor()
        cur.execute("SELECT value FROM app_config WHERE key=?", (key,))
        row = cur.fetchone()
        db.close()
        return (row["value"] if row else default) or default
    except Exception as e:
        print("[app_config] get failed:", e)
        return default


def _set_cfg(key: str, value: str):
    _ensure_app_config()
    try:
        ts = datetime.utcnow().isoformat(timespec="seconds")
        db = sqlite3.connect(DB_PATH)
        cur = db.cursor()
        cur.execute("""
            INSERT INTO app_config(key, value, updated_at)
            VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, str(value), ts))
        db.commit()
        db.close()
    except Exception as e:
        print("[app_config] set failed:", e)


def _get_library_sequence(run_id: int):
    """Return parsed sequence_json dict for a library run_id, or None."""
    try:
        rid = int(run_id)
    except Exception:
        return None
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT sequence_json FROM RunSequences WHERE run_id = ?", (rid,))
    row = cur.fetchone()
    db.close()
    if not row:
        return None
    try:
        return json.loads(row["sequence_json"])
    except Exception:
        return None


@app.get("/admin/api/active_static_scenario", endpoint="admin_get_active_static_scenario_picker")
@admin_required
def admin_get_active_static_scenario():
    sid = (_get_cfg("active_static_scenario_sid", "") or "").strip().upper()
    if not sid:
        sid = "A" if "A" in (SCENARIO_SEQUENCES or {}) else (next(iter((SCENARIO_SEQUENCES or {}).keys()), "A"))
    return jsonify({"ok": True, "sid": sid})


@app.post("/admin/api/active_static_scenario", endpoint="admin_set_active_static_scenario_picker")
@admin_required
def admin_set_active_static_scenario():
    payload = request.get_json(silent=True) or {}
    sid = (payload.get("sid") or "").strip().upper()
    if not sid:
        return jsonify({"ok": False, "error": "Missing sid"}), 400
    if sid not in (SCENARIO_SEQUENCES or {}):
        return jsonify({"ok": False, "error": "Unknown sid"}), 404
    _set_cfg("active_static_scenario_sid", sid)
    return jsonify({"ok": True, "sid": sid})


@app.get("/admin/api/library_stories", endpoint="admin_api_library_stories_picker")
@admin_required
def admin_api_library_stories():
    q = (request.args.get("q") or "").strip()
    try:
        items = _get_public_gpt_scenarios(limit=120, q=q, exclude_username=None)
        return jsonify({"ok": True, "stories": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/admin/api/active_static_library", endpoint="admin_get_active_static_library_picker")
@admin_required
def admin_get_active_static_library():
    run_id = (_get_cfg("active_static_library_run_id", "") or "").strip()
    return jsonify({"ok": True, "run_id": run_id})


@app.post("/admin/api/active_static_library", endpoint="admin_set_active_static_library_picker")
@admin_required
def admin_set_active_static_library():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id") or "").strip()

    # allow clearing
    if run_id == "":
        _set_cfg("active_static_library_run_id", "")
        return jsonify({"ok": True, "run_id": ""})

    try:
        rid = int(run_id)
    except ValueError:
        return jsonify({"ok": False, "error": "run_id must be int"}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT 1 FROM Runs WHERE id=? AND run_type='gpt'", (rid,))
    ok_run = c.fetchone() is not None
    c.execute("SELECT 1 FROM RunSequences WHERE run_id=?", (rid,))
    ok_seq = c.fetchone() is not None
    conn.close()

    if not ok_run or not ok_seq:
        return jsonify({"ok": False, "error": "Library story not found"}), 404

    _set_cfg("active_static_library_run_id", str(rid))
    return jsonify({"ok": True, "run_id": str(rid)})


@app.get("/admin/api/scenarios")
@admin_required
def admin_api_scenarios():
    """
    List all static scenarios available in scenarios.json.
    """
    items = []
    for sid, seq in (SCENARIO_SEQUENCES or {}).items():
        steps = (seq or {}).get("steps") or []
        items.append({
            "sid": sid,
            "title": (seq or {}).get("title") or f"Scenario {sid}",
            "steps": len(steps)
        })
    items.sort(key=lambda x: x["sid"])
    return jsonify({"ok": True, "scenarios": items})


@app.get("/admin/api/scenario/<sid>")
@admin_required
def admin_api_scenario_detail(sid):
    """
    Return the raw scenario object (editable).
    Note: This returns the stored structure (including adaptive objects if you used them).
    """
    seq = (SCENARIO_SEQUENCES or {}).get(sid)
    if not seq:
        return jsonify({"ok": False, "error": "Scenario not found"}), 404
    return jsonify({"ok": True, "scenario": seq})


@app.post("/admin/api/scenario/<sid>/step/<int:step_id>")
@admin_required
def admin_api_scenario_update_step(sid, step_id):
    """
    Update one step inside a scenario and persist to scenarios.json.
    This will immediately affect scenario_step_static for new runs.
    """
    payload = request.get_json(silent=True) or {}
    seq = (SCENARIO_SEQUENCES or {}).get(sid)
    if not seq:
        return jsonify({"ok": False, "error": "Scenario not found"}), 404

    steps = seq.get("steps") or []
    if step_id < 1 or step_id > len(steps):
        return jsonify({"ok": False, "error": "Invalid step_id"}), 400

    idx = step_id - 1
    st = steps[idx]
    if not isinstance(st, dict):
        return jsonify({"ok": False, "error": "Step is not editable"}), 400

    # Allow updating basic fields
    for k in ("title", "situation", "question"):
        if k in payload:
            st[k] = payload.get(k)

    # Options: allow replacing the whole list
    # Expected: [{value,label,consequence,scores,...}, ...]
    if "options" in payload:
        st["options"] = payload.get("options")

    # ensure step id stays correct
    st["id"] = st.get("id", step_id)

    # Write back to the global schema, validate, save, reload
    SCENARIO_SEQUENCES[sid]["steps"][idx] = st
    validate_scenarios(SCENARIO_SEQUENCES)
    _save_scenarios_to_disk(SCENARIO_SEQUENCES)
    _reload_scenarios_global()

    return jsonify({"ok": True, "saved": {"sid": sid, "step_id": step_id}})


def ensure_app_config_table():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS app_config (
        key TEXT PRIMARY KEY,
        value TEXT
      )
    """)
    conn.commit()
    conn.close()


def get_active_static_sid(default="A"):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT value FROM app_config WHERE key='active_static_sid' LIMIT 1")
    row = c.fetchone()
    conn.close()
    sid = (row["value"] if row else default) or default
    sid = sid.strip().upper()
    return sid if sid in SCENARIO_SEQUENCES else default


from collections import Counter
import json


@app.get("/run/<int:run_id>/result")
@login_required
def run_result(run_id: int):
    """
    Render scenario_result.html for a historical run (Mission Log entry).

    FIXES:
    1) Reads the latest decision per step_index (handles duplicates from re-submits).
    2) Works with both schemas: option_value OR opt_value.
    """

    username = session.get("username")

    # -------------------------
    # 1) Load run and validate owner
    # -------------------------
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, run_type, sid, title, started_at, finished_at
        FROM Runs
        WHERE id=?
        """,
        (run_id,)
    )
    run = cur.fetchone()
    conn.close()

    if not run or run["username"] != username:
        if request.args.get("partial") == "1":
            return jsonify({"html": "<div class='empty-state'><p>Run not found.</p></div>"}), 404
        flash("Run not found.", "warning")
        return redirect(url_for("profile"))

    if not run["finished_at"]:
        if request.args.get("partial") == "1":
            return jsonify({"html": "<div class='empty-state'><p>This mission is not finished yet.</p></div>"}), 400
        flash("This mission is not finished yet.", "warning")
        return redirect(url_for("profile"))

    # -------------------------
    # 2) Resolve scenario sequence for this run
    # -------------------------
    seq = _get_library_sequence(run_id)
    if not seq:
        sid_u = (run["sid"] or "").strip().upper()
        if not sid_u or sid_u not in SCENARIO_SEQUENCES:
            if request.args.get("partial") == "1":
                return jsonify({"html": "<div class='empty-state'><p>Scenario sequence not found.</p></div>"}), 404
            flash("Scenario sequence not found for this run.", "warning")
            return redirect(url_for("profile"))
        seq = SCENARIO_SEQUENCES[sid_u]

    steps = seq.get("steps", []) or []
    if not steps:
        if request.args.get("partial") == "1":
            return jsonify({"html": "<div class='empty-state'><p>Scenario is missing steps.</p></div>"}), 500
        flash("Scenario is missing steps.", "danger")
        return redirect(url_for("profile"))

    total = len(steps)

    # -------------------------
    # 3) Read progress (choices) from RunDecisions (LATEST per step)
    # -------------------------
    conn = get_db()
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(RunDecisions)")
    cols = {row[1] for row in cur.fetchall()}

    if "option_value" in cols:
        value_col = "option_value"
    elif "opt_value" in cols:
        value_col = "opt_value"
    else:
        conn.close()
        if request.args.get("partial") == "1":
            return jsonify({
                               "html": "<div class='empty-state'><p>RunDecisions table is missing a decision value column.</p></div>"}), 500
        flash("RunDecisions table is missing a decision value column.", "danger")
        return redirect(url_for("profile"))

    # Select the latest row per step_index (MAX(id)) to avoid duplicates issues
    cur.execute(
        f"""
        SELECT d.step_index, d.{value_col} AS choice_value
        FROM RunDecisions d
        JOIN (
            SELECT step_index, MAX(id) AS max_id
            FROM RunDecisions
            WHERE run_id=?
            GROUP BY step_index
        ) last
          ON last.step_index = d.step_index AND last.max_id = d.id
        WHERE d.run_id=?
        ORDER BY d.step_index ASC
        """,
        (run_id, run_id)
    )
    rows = cur.fetchall()
    conn.close()

    progress = [""] * total
    for r in rows:
        try:
            si = int(r["step_index"])
        except Exception:
            continue
        if 1 <= si <= total:
            progress[si - 1] = (r["choice_value"] or "").strip()

    if not all(progress):
        if request.args.get("partial") == "1":
            return jsonify({
                               "html": "<div class='empty-state'><p>This mission is incomplete or missing saved decisions.</p></div>"}), 400
        flash("This mission is incomplete or missing saved decisions.", "warning")
        return redirect(url_for("profile"))

    # -------------------------
    # 4) Executive summary (letter path)
    # -------------------------
    letters = [c[0].upper() for c in progress if c]
    path_letters = "".join(letters)
    dominant_letter = Counter(letters).most_common(1)[0][0] if letters else "A"

    resolutions = seq.get("resolutions", {}) or {}
    exec_summary = resolutions.get(dominant_letter) or resolutions.get("A") or {
        "title": "The Ethical Journey",
        "resolution": "Your choices shaped a distinct path under uncertainty."
    }

    # -------------------------
    # 5) Build decisions_view
    # -------------------------
    decisions_view = []
    prev_choice = None

    for idx, choice in enumerate(progress, start=1):
        raw = steps[idx - 1]
        st = _resolve_step_view(raw, prev_choice)

        chosen_opt = next(
            (o for o in (st.get("options") or []) if o.get("value") == choice),
            None
        ) or {
                         "label": "Unknown Choice",
                         "consequence": "No data recorded.",
                         "story_line": ""
                     }

        decisions_view.append({
            "step_id": idx,
            "step_title": st.get("title", f"Phase {idx}"),
            "situation_text": st.get("situation", "") or "",
            "question_text": st.get("question", "") or "",
            "chosen_value": choice,
            "chosen_label": chosen_opt.get("label", "") or "",
            "label": chosen_opt.get("label", "") or "",
            "consequence": chosen_opt.get("consequence", "") or "",
            "story_line": chosen_opt.get("story_line", "") or "",
            "all_options": st.get("options", []) or []
        })

        prev_choice = choice

    # -------------------------
    # 6) Use cached GPT resolution (or generate if missing)
    # -------------------------
    gpt_report = None
    cached = _get_run_resolution(run_id)
    if cached:
        gpt_report = safe_json_loads(cached, cached)

    if (not gpt_report) and _client:
        try:
            rep = generate_analytic_resolution(
                decisions_view=decisions_view,
                scenario_title=seq.get("title", run["title"] or "Ethics Report"),
                scenario_intro=seq.get("intro", "") or "",
                path_letters=path_letters
            )
            _save_run_resolution(run_id, json.dumps(rep, ensure_ascii=False))
            gpt_report = rep
        except Exception as e:
            print("[run_result] generate_analytic_resolution failed:", e)

    ending_text = None
    if isinstance(gpt_report, dict):
        ending_text = gpt_report.get("final_resolution")
    if not ending_text:
        ending_text = exec_summary.get("resolution", "")

    # -------------------------
    # 7) Render (full OR inline partial)
    # -------------------------
    ctx = dict(
        scenario_title=seq.get("title", run["title"] or "The Ethics Report"),
        path_letters=path_letters,
        decisions=decisions_view,
        ending=ending_text,
        gpt_report=gpt_report,
        run_id=run_id
    )

    if request.args.get("partial") == "1":
        html = render_template("_result_inline.html", **ctx)
        return jsonify({"html": html})

    return render_template("scenario_result.html", **ctx)


if __name__ == "__main__":
    print("\n=== URL MAP ===")
    print(app.url_map)
    print("==============\n")
    app.run(debug=True, port=8080)
