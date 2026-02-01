# app.py (FULL UPDATED — GPT image generation is LAZY: generate only the current step image, one at a time)
# IMPORTANT: DO NOT hardcode your OpenAI key. Set OPENAI_API_KEY in your environment.

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3, os, json, re, base64, hashlib
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
        title TEXT,
        total_steps INTEGER,
        started_at TEXT,
        finished_at TEXT,
        prefs_json TEXT,
        FOREIGN KEY(username) REFERENCES Users(username)
    );
    """)

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
        source TEXT DEFAULT 'static'
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
        source TEXT DEFAULT 'static',
        UNIQUE(run_id, step_id, source),
        FOREIGN KEY(run_id) REFERENCES research_runs(id)
    );
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_research_runs_user ON research_runs(username);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_runs_finished ON research_runs(finished_at);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_research_decisions_run ON research_decisions(run_id);")

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
    """Story so far = situation + decision story_line (fallback consequence)."""
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
        choice = progress[idx]

        situation = st.get("situation")
        if isinstance(situation, str) and situation.strip():
            chunks.append(situation.strip())

        opt = next((o for o in (st.get("options") or []) if isinstance(o, dict) and o.get("value") == choice), None)
        if opt:
            line = (opt.get("story_line") or opt.get("consequence") or "").strip()
            if line:
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


# ====== Run logging ======
def _start_run(run_type, sid, title, total_steps, prefs=None):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """INSERT INTO Runs(username, run_type, sid, title, total_steps, started_at, prefs_json)
           VALUES(?,?,?,?,?,?,?)""",
        (
            session.get("username", "anonymous"),
            run_type, sid, title, total_steps,
            datetime.utcnow().isoformat(timespec="seconds"),
            json.dumps(prefs or {}, ensure_ascii=False),
        ),
    )
    run_id = c.lastrowid
    conn.commit()
    conn.close()
    session["run_id"] = run_id
    return run_id


def _log_decision(run_id, step_index, opt_value, opt_label, opt_consequence):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """INSERT INTO RunDecisions(run_id, step_index, option_value, option_label, option_consequence)
           VALUES(?,?,?,?,?)""",
        (run_id, step_index, opt_value, opt_label, opt_consequence),
    )
    conn.commit()
    conn.close()


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


def generate_analytic_resolution(decisions_view: list, scenario_title: str, scenario_intro: str, path_letters: str) -> dict:
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
        title=seq.get("title", f"Scenario {sid}"),
        total_steps=len(seq["steps"]),
        prefs=None
    )
    return redirect(url_for("scenario_step", sid=sid, step=1))


RESEARCH_ACCESS_CODE = "111"


@app.route("/research", methods=["GET", "POST"])
@login_required
def research_gate():
    if request.method == "POST":
        code = request.form.get("access_code", "").strip()
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
                # Fall back to your default static scenario
                session["scenario_sid"] = "A"
                session.pop("library_run_id", None)

            session["scenario_progress"] = []
            session["make_image"] = False
            session["step_start_time"] = datetime.utcnow().timestamp()

            # Start a research run record (stored in Runs)
            if session.get("scenario_sid") == "LIB":
                lib_seq = _get_library_sequence(session.get("library_run_id"))
                lib_steps = (lib_seq or {}).get("steps") or []
                run_title = "IHL Research Study (Library)"
                try:
                    dbt = get_db()
                    cur = dbt.cursor()
                    cur.execute("SELECT title FROM Runs WHERE id=?", (session.get("library_run_id"),))
                    rr = cur.fetchone()
                    dbt.close()
                    if rr and rr["title"]:
                        run_title = f'IHL Research Study (Library): {rr["title"]}'
                except Exception:
                    pass

                run_id = _start_run(
                    run_type="research",
                    sid=f"LIB:{session.get('library_run_id')}",
                    title=run_title,
                    total_steps=len(lib_steps)
                )
                redirect_sid = "LIB"
            else:
                run_id = _start_run(
                    run_type="research",
                    sid="A",
                    title="IHL Research Study",
                    total_steps=len(SCENARIO_SEQUENCES["A"]["steps"])
                )
                redirect_sid = "A"

            # Store minimal research session metadata
            conn = get_db()
            conn.execute(
                "INSERT INTO ResearchSessions(run_id, access_code, start_time) VALUES(?,?,?)",
                (run_id, code, datetime.utcnow().isoformat(timespec="seconds"))
            )
            conn.commit()
            conn.close()

            session["run_id"] = run_id
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
        option_value: str
):
    scenarios = load_scenarios_for_research()
    _, opt = find_option(scenarios, scenario_key, step_id, option_value)
    if not opt:
        return False, "Option not found in scenarios.json"

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
            INSERT INTO research_runs(user_id, username, scenario_key, started_at, source)
            VALUES(?,?,?,?, 'static')
        """, (user_id, username, scenario_key, now_ts))
        run_id = cur.lastrowid

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
                scores_json=?, ethics_index=?, created_at=?
            WHERE id=?
        """, (option_value, chosen_letter, chosen_label,
              payload_scores_json, ethics_index, now_ts, existing["id"]))
    else:
        cur.execute("""
            INSERT INTO research_decisions(
              run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              scores_json, ethics_index, created_at, source
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?, 'static')
        """, (run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              payload_scores_json, ethics_index, now_ts))

    total_steps = len((scenarios.get(scenario_key) or {}).get("steps", [])) or 4
    if step_id >= total_steps:
        cur.execute("""
            UPDATE research_runs SET finished_at=?
            WHERE id=? AND finished_at IS NULL
        """, (now_ts, run_id))

    conn.commit()
    conn.close()
    return True, None


@app.route("/scenario/<sid>/step/<int:step>", methods=["GET", "POST"])
@login_required
def scenario_step(sid, step):
    # Special: Library story used for static/research runs
    if sid == "LIB":
        lib_run_id = session.get("library_run_id")
        lib_seq = _get_library_sequence(lib_run_id)
        if not lib_seq:
            flash("Selected library story is missing or invalid.", "warning")
            return redirect(url_for("research_gate"))
        seq = lib_seq
        steps = seq.get("steps", []) or []
    else:
        if sid not in SCENARIO_SEQUENCES:
            return redirect(url_for("scenario_start", sid="A"))

        seq = SCENARIO_SEQUENCES[sid]
        steps = seq.get("steps", [])

    if not steps:
        flash("Scenario is missing steps.", "danger")
        return redirect(url_for("scenario_start", sid="A"))

    if step < 1 or step > len(steps):
        return redirect(url_for("scenario_start", sid=sid))

    progress = session.get("scenario_progress", [])
    run_id = session.get("run_id")

    # Resolve current step adaptively based on the previous choice
    prev_choice = None
    if step > 1 and len(progress) >= (step - 1):
        prev_choice = progress[step - 2] or None

    current_raw = steps[step - 1]
    current = _resolve_step_view(current_raw, prev_choice)

    if request.method == "POST":
        choice = (request.form.get("choice") or "").strip()
        if not choice:
            flash("Please select one option.", "warning")
            return redirect(url_for("scenario_step", sid=sid, step=step))

        if len(progress) >= step:
            progress[step - 1] = choice
        else:
            while len(progress) < step - 1:
                progress.append("")
            progress.append(choice)

        session["scenario_progress"] = progress
        session.modified = True

        chosen_opt = next((o for o in current.get("options", []) if o.get("value") == choice), None)
        if run_id and chosen_opt:
            _log_decision(
                run_id=run_id,
                step_index=step,
                opt_value=choice,
                opt_label=chosen_opt.get("label", "") or "",
                opt_consequence=chosen_opt.get("consequence", "") or ""
            )

        if session.get("mode") == "research":
            ok, err = _upsert_static_research_decision(
                username=session.get("username"),
                user_id=session.get("user_id"),
                scenario_key=(sid or "").strip().upper(),
                step_id=step,
                option_value=choice
            )
            if not ok:
                print("[research mirror] failed:", err)

        if step < len(steps):
            return redirect(url_for("scenario_step", sid=sid, step=step + 1))
        return redirect(url_for("scenario_result", sid=sid))

    story_text = story_from_progress(seq, progress, step - 1)

    prev_recap = None
    if step > 1 and len(progress) >= (step - 1):
        derived = derive_recaps(seq, progress)
        prev_recap = derived[step - 2] if len(derived) >= (step - 1) else None

    sid_u = (sid or "").strip().upper()
    mode = session.get("mode", "static")
    show_images = (mode in ("static", "research"))  # static/research only

    hero_image = None
    img_debug_candidates = []
    img_debug_progress = progress[:]

    if show_images:
        prev_letters = []
        for c in progress[:max(0, step - 1)]:
            if c:
                prev_letters.append(str(c).strip().upper()[0])

        suffix = sid_u + "".join(prev_letters)

        cand1 = f"{sid_u}/step_{step}_{suffix}.png"
        img_debug_candidates.append(cand1)

        cand2 = f"{sid_u}/step_{step}_{sid_u}.png"
        cand3 = f"{sid_u}/step_{step}.png"
        img_debug_candidates.extend([cand2, cand3])

        for cand in img_debug_candidates:
            abs_path = os.path.join(app.static_folder, "scenarios", cand)
            if os.path.exists(abs_path):
                hero_image = cand
                break

    return render_template(
        "scenario_step_static.html",
        scenario_id=seq.get("id", sid),
        scenario_title=seq.get("title", f"Scenario {sid}"),
        step=current,
        step_index=step,
        total_steps=len(steps),
        situation_text=current.get("situation", ""),
        story_so_far=story_text,
        prev_recap=prev_recap,
        hero_image=hero_image,
        show_images=show_images,
        is_last=(step == len(steps)),
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
        seq = _get_library_sequence(lib_run_id)
        if not seq:
            flash("Selected library story is missing or invalid.", "warning")
            return redirect(url_for("research_gate"))
        steps = seq.get("steps", []) or []
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
             "resolution": (gpt_report.get("executive_summary") if isinstance(gpt_report, dict) and gpt_report.get("executive_summary") else exec_summary.get("resolution"))}
            if isinstance(gpt_report, dict) else exec_summary
        ),
        ending=(gpt_report.get("final_resolution") if isinstance(gpt_report, dict) and gpt_report.get("final_resolution") else exec_summary.get("resolution", "Path complete.")),
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

        return redirect(url_for("gpt_scenario_step", step=1))

    session.setdefault("make_image", False)
    return render_template("gpt_scenario_prefs.html")


@app.route("/gpt-scenario/step/<int:step>", methods=["GET", "POST"])
@login_required
def gpt_scenario_step(step: int):
    seq = _get_active_gpt_sequence()
    if not seq:
        return redirect(url_for("gpt_scenario_prefs"))

    steps = seq.get("steps", []) or []
    total_steps = int(seq.get("total_steps") or TOTAL_GPT_STEPS)

    # Allow user to visit step up to total_steps
    if step < 1 or step > total_steps:
        return redirect(url_for("gpt_scenario_prefs"))

    progress = session.get("scenario_progress", [])
    run_id = session.get("run_id")
    prefs = _get_run_prefs(run_id) if run_id else {}

    # If requested step not generated yet, generate it now (sequentially up to step)
    if len(steps) < step:
        while len(steps) < step:
            next_step_index = len(steps) + 1

            if next_step_index == 1:
                # Step 1 should already exist, but just in case:
                initial = generate_gpt_initial_sequence_with_llm(prefs)
                steps = initial.get("steps", []) or []
                seq["title"] = initial.get("title", seq.get("title"))
                seq["intro"] = initial.get("intro", seq.get("intro"))
                seq["total_steps"] = total_steps
                seq["steps"] = steps
                if run_id:
                    _save_run_sequence(run_id, seq)
            else:
                # Build story up to previous step
                story_text_prev = story_from_progress(seq, progress, next_step_index - 1)
                last_choice = progress[next_step_index - 2] if len(progress) >= (next_step_index - 1) else "UNKNOWN"

                new_step = generate_gpt_next_step_with_llm(
                    prefs=prefs,
                    story_so_far=story_text_prev,
                    last_choice_value=last_choice,
                    step_index=next_step_index
                )
                steps.append(new_step)
                seq["steps"] = steps
                if run_id:
                    _save_run_sequence(run_id, seq)

    # Now safe to read the current step object
    current = steps[step - 1]

    if request.method == "POST":
        choice = request.form.get("choice")
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

        session["scenario_progress"] = progress
        session.modified = True

        chosen_opt = next((o for o in (current.get("options") or []) if o.get("value") == choice), None)
        if run_id and chosen_opt:
            _log_decision(
                run_id,
                step,
                choice,
                chosen_opt.get("label", ""),
                chosen_opt.get("consequence", "")
            )

        # IMPORTANT FIX:
        # Move forward based on intended total_steps, not len(steps) (which may be only 1 early on)
        if step < total_steps:
            return redirect(url_for("gpt_scenario_step", step=step + 1))
        return redirect(url_for("gpt_scenario_result"))

    # ---- GET ----
    story_text = story_from_progress(seq, progress, step - 1)

    prev_recap = None
    if step > 1 and len(progress) >= (step - 1):
        derived = derive_recaps(seq, progress)
        prev_recap = derived[step - 2] if len(derived) >= (step - 1) else None

    # LAZY IMAGE: generate ONLY the current step image if enabled.
    show_images = bool(session.get("make_image", False))
    hero_image_url = None
    if show_images and run_id:
        cached = _get_run_image(run_id, step)
        if cached and cached.get("image_path"):
            hero_image_url = "/static/" + cached["image_path"].lstrip("/")
        else:
            prefs_for_image = prefs or {}
            hero_image_url = _generate_and_store_step_image(
                run_id=run_id,
                step_index=step,
                seq=seq,
                prefs=prefs_for_image,
                step_obj=current,
                story_so_far=story_text
            )

    return render_template(
        "scenario_step_gpt.html",
        scenario_id=seq.get("id", "G"),
        scenario_title=seq.get("title", "GPT Scenario"),
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
        is_admin=_is_admin_user(session.get("username"))
    )


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
                last_choice = progress[next_step_index - 2] if len(progress) >= (next_step_index - 1) else "UNKNOWN"
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
            last_choice = progress[step - 2] if len(progress) >= (step - 1) else "UNKNOWN"

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
        last_choice = progress[next_step_index - 2] if len(progress) >= (next_step_index - 1) else "UNKNOWN"
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

    ending = "Your decisions were recorded. Review the consequences and the evolving narrative."

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

    return {
        "counts": {"runs": total_runs, "decisions": total_decisions, "reflections": total_reflections},
        "letter_counts": dict(letter_counts),
        "dimensions_raw": raw,
        "dimensions_norm": norm,
        "sentiment": sentiment,
        "survey": survey,
    }


def hydrate_runs_with_details(username: str, limit: int = 12) -> list:
    conn = get_db()
    c = conn.cursor()

    c.execute("""
      SELECT id, run_type, sid, title, total_steps, started_at, finished_at, prefs_json
      FROM Runs
      WHERE username=?
      ORDER BY COALESCE(finished_at, started_at) DESC
      LIMIT ?
    """, (username, limit))
    runs = [dict(r) for r in c.fetchall()]

    for r in runs:
        run_id = r["id"]
        r["prefs"] = _safe_json_loads(r.get("prefs_json"), {})

        c.execute("""
          SELECT step_index, option_value, option_label, option_consequence
          FROM RunDecisions
          WHERE run_id=?
          ORDER BY step_index ASC
        """, (run_id,))
        decisions = []
        for d in c.fetchall():
            decisions.append({
                "step_index": d["step_index"],
                "step_title": f"Step {d['step_index']}",
                "chosen_value": d["option_value"],
                "chosen_label": d["option_label"],
                "chosen_consequence": d["option_consequence"],
                "all_options": []
            })
        r["decisions"] = decisions

        c.execute("""
          SELECT step_index, phase, question_text, response_text, sentiment_score, sentiment_label,
                 choice_value, choice_label
          FROM RunReflections
          WHERE run_id=?
          ORDER BY step_index ASC, phase ASC
        """, (run_id,))
        r["reflections"] = [dict(x) for x in c.fetchall()]

        sid = (r.get("sid") or "").strip().upper()

        if sid in SCENARIO_SEQUENCES:
            seq = SCENARIO_SEQUENCES[sid]
            steps = seq.get("steps", [])
            intro = (seq.get("intro") or "")
            r["scenario_intro_excerpt"] = intro[:180] + ("..." if intro[180:] else "")
            for d in r["decisions"]:
                si = d["step_index"] - 1
                if 0 <= si < len(steps):
                    st = steps[si]
                    d["step_title"] = st.get("title", d["step_title"])
                    d["all_options"] = st.get("options", [])
        elif sid == "G":
            seq = _get_run_sequence(run_id) or {}
            steps = (seq.get("steps") or [])
            intro = (seq.get("intro") or "")
            r["scenario_intro_excerpt"] = intro[:180] + ("..." if intro[180:] else "")
            for d in r["decisions"]:
                si = d["step_index"] - 1
                if 0 <= si < len(steps):
                    st = steps[si]
                    d["step_title"] = st.get("title", d["step_title"])
                    d["all_options"] = st.get("options", [])

    conn.close()
    return runs

def generate_user_persona(stats: dict, user_prefs: dict | None = None, runs: list | None = None) -> str:
    """
    Decision DNA (profile page):
    - Professional tone
    - Uses: dominant decision type (A/B/C/D), ethics radar, user prefs, and recent mission content.
    """
    # Safe fallback if OpenAI client not available
    if not _client:
        return "Decision profile is unavailable at the moment. Please try again later."

    user_prefs = user_prefs or {}
    runs = runs or []

    dominant_letter = (
        max(stats.get("letter_counts", {}), key=stats["letter_counts"].get)
        if stats.get("letter_counts")
        else "N/A"
    )
    radar_data = stats.get("dimensions_norm") or {}

    # Pull a compact “evidence” snippet from recent decisions so it reflects research/library activity
    evidence_bits = []
    for r in (runs or [])[:6]:
        for d in (r.get("decisions") or [])[:3]:
            lab = (d.get("chosen_label") or "").strip()
            cons = (d.get("chosen_consequence") or "").strip()
            if lab:
                if cons:
                    evidence_bits.append(f"{lab} → {cons}")
                else:
                    evidence_bits.append(lab)
    evidence_bits = evidence_bits[:6]

    preferred_war = (user_prefs.get("preferred_war") or "").strip()
    interest = (user_prefs.get("interest") or "").strip()

    prompt = (
        "You are writing a short 'Decision DNA' summary for a user profile in a research platform.\n"
        "Write 2–3 sentences in a formal, professional tone.\n"
        "Constraints:\n"
        "- No childish language, no metaphors (no superhero/captain/team leader comparisons).\n"
        "- No exclamation marks.\n"
        "- Avoid vague praise; be specific and analytical.\n"
        "- Use second person ('You...') but keep it professional.\n"
        "- Keep it under 65 words.\n\n"
        f"Signals:\n"
        f"- Dominant decision type: {dominant_letter}\n"
        f"- Ethics radar (0–1): {radar_data}\n"
        f"- Preferred conflict context (optional): {preferred_war or 'N/A'}\n"
        f"- Stated interest (optional): {interest or 'N/A'}\n"
        f"- Recent decision evidence (optional): {evidence_bits if evidence_bits else 'N/A'}\n\n"
        "Output only the Decision DNA text, nothing else."
    )

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an analyst writing concise professional profile summaries."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip().strip('"')
    except Exception:
        return "Your decisions reflect structured reasoning under constraints, with consistent attention to ethical trade-offs and operational feasibility."


def count_user_runs(username: str) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) AS cnt FROM Runs WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return int(row["cnt"] if row and "cnt" in row.keys() else 0)


from flask import request, render_template, session
import sqlite3
from flask import jsonify, render_template, request, session


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

    runs = hydrate_runs_with_details(username=username, limit=PER_PAGE, offset=offset)
    stats = compute_profile_stats(runs)

    # ✅ AJAX partial response (only mission log HTML)
    if request.args.get("partial") == "1":
        html = render_template(
            "_profile_mission_log.html",
            page_runs=runs,
            page=page,
            total_pages=total_pages,
            total_runs=total_runs,
            per_page=PER_PAGE
        )
        return jsonify({"html": html, "page": page, "total_pages": total_pages})

    return render_template(
        "profile.html",
        user=user,
        page_runs=runs,
        runs=runs,
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

    runs = hydrate_runs_with_details(username=username, limit=12)
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
            INSERT INTO research_runs(user_id, username, scenario_key, started_at, source)
            VALUES(?,?,?,?, 'static')
        """, (user_id, username, scenario_key, now_ts))
        run_id = cur.lastrowid

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
                scores_json=?, ethics_index=?, created_at=?
            WHERE id=?
        """, (option_value, chosen_letter, chosen_label,
              payload_scores_json, ethics_index, now_ts, existing["id"]))
    else:
        cur.execute("""
            INSERT INTO research_decisions(
              run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              scores_json, ethics_index, created_at, source
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?, 'static')
        """, (run_id, user_id, username, scenario_key, step_id,
              option_value, chosen_letter, chosen_label,
              payload_scores_json, ethics_index, now_ts))

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
        "Story so far (derived from prior steps + choices):\n"
        f"{(story_so_far or '').strip()}\n\n"
        f"Last choice made: {last_choice_value}\n"
        f"Now generate step {step_index}.\n"
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



if __name__ == "__main__":
    print("\n=== URL MAP ===")
    print(app.url_map)
    print("==============\n")
    app.run(debug=True, port=8080)
