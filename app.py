from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3, os, json, re, random
from datetime import datetime
from functools import wraps

# ===== App setup =====
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
os.makedirs(os.path.join("static", "generated"), exist_ok=True)

DB_PATH = os.path.join("static", "database.db")
# --- Jinja filters: timestamps -> local datetime strings ---
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

@app.template_filter("datetime")
def jinja_datetime_filter(value, fmt="%Y-%m-%d %H:%M"):
    """
    Convert a unix timestamp (int/str) to a formatted string.
    Defaults to Asia/Seoul. If zoneinfo isn't available, falls back to system tz.
    Usage in templates: {{ timestamp|int|datetime }}  or  {{ ts|datetime("%b %d, %H:%M") }}
    """
    if value is None:
        return ""
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return ""
    tz = ZoneInfo("Asia/Seoul") if ZoneInfo else None
    dt = datetime.fromtimestamp(ts, tz=tz) if tz else datetime.fromtimestamp(ts)
    return dt.strftime(fmt)

# ===== Simple DB helpers =====
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs("static", exist_ok=True)
    conn = get_db()
    c = conn.cursor()

    # Users
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

    # Runs: one row per playthrough
    c.execute("""
    CREATE TABLE IF NOT EXISTS Runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        run_type TEXT NOT NULL,   -- 'static' or 'gpt'
        sid TEXT,                 -- 'A'/'B'/'C'/'D' for static; 'G' for GPT
        title TEXT,
        total_steps INTEGER,
        started_at TEXT,
        finished_at TEXT,
        prefs_json TEXT,
        FOREIGN KEY(username) REFERENCES Users(username)
    );
    """)

    # Per-step decisions
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

    # Pre/Post survey & reflection per step
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

    # Reflective journal per run (JSON)
    c.execute("""
    CREATE TABLE IF NOT EXISTS RunJournals(
        run_id INTEGER PRIMARY KEY,
        journal_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    # Store GPT/static sequence server-side (avoid cookie overflow)
    c.execute("""
    CREATE TABLE IF NOT EXISTS RunSequences(
        run_id INTEGER PRIMARY KEY,
        sequence_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES Runs(id)
    );
    """)

    # --- Safe schema upgrades (no-op if present) ---
    try:
        c.execute("ALTER TABLE RunReflections ADD COLUMN choice_value TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE RunReflections ADD COLUMN choice_label TEXT")
    except Exception:
        pass

    conn.commit()
    conn.close()

# Initialize DB immediately (Flask 3.x has no before_first_request)
init_db()

# ===== Paths =====
SCENARIOS_JSON_PATHS = [
    os.path.join("static", "scenarios", "scenarios.json"),
    os.path.join("scenarios.json"),  # fallback for local testing
]
OUTCOMES_JSON_PATHS = [
    os.path.join("static", "scenarios", "outcomes_64_map.json"),
    os.path.join("outcomes_64_map.json"),  # optional fallback
]

# ===== JSON Loaders =====
def load_json_first(paths, required=True):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    if required:
        raise FileNotFoundError(f"None of these files found: {paths}")
    return {}

def validate_scenarios(schema):
    if not isinstance(schema, dict):
        raise ValueError("scenarios.json must be a top-level object keyed by sequence IDs.")
    if not set(schema.keys()).issuperset({"A", "B", "C", "D"}):
        raise ValueError("scenarios.json must include sequences: A, B, C, D.")
    for sid, seq in schema.items():
        if seq.get("id") != sid:
            raise ValueError(f"Sequence '{sid}' must have matching id field.")
        steps = seq.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError(f"Sequence '{sid}' must include a non-empty steps array.")
        for st in steps:
            missing = [k for k in ("id", "title", "options") if k not in st]
            if missing:
                raise ValueError(f"Sequence '{sid}' step missing {', '.join(missing)}.")
            if not st.get("situation"):
                raise ValueError(f"Sequence '{sid}' step {st.get('id')} must include 'situation'.")
            if not st.get("question"):
                raise ValueError(f"Sequence '{sid}' step {st.get('id')} must include 'question'.")
            opts = st["options"]
            if not isinstance(opts, list) or len(opts) == 0:
                raise ValueError(f"Sequence '{sid}' step {st.get('id')} has no options.")
            for opt in opts:
                if not all(k in opt for k in ("value", "label", "consequence")):
                    raise ValueError(
                        f"Sequence '{sid}' step {st['id']} option missing value/label/consequence."
                    )

def first_letter_path(progress):
    return "".join((p[0] if p else "?") for p in progress)

# ===== Load authored content (static) =====
SCENARIO_SEQUENCES = load_json_first(SCENARIOS_JSON_PATHS, required=True)
validate_scenarios(SCENARIO_SEQUENCES)

OUTCOME_MAP = load_json_first(OUTCOMES_JSON_PATHS, required=False)
if not OUTCOME_MAP:
    OUTCOME_MAP = {"AAA": "Outcome map missing; add static/scenarios/outcomes_64_map.json."}

# ===== Helpers =====
def build_label_map(steps):
    return {i+1: {o["value"]: o["label"] for o in st["options"]} for i, st in enumerate(steps)}

def derive_recaps(seq: dict, progress: list):
    """Rebuild per-step recap directly from sequence + chosen options."""
    recaps = []
    steps = seq.get("steps", [])
    for idx, choice in enumerate(progress[:len(steps)], start=1):
        st = steps[idx-1]
        opt = next((o for o in st.get("options", []) if o.get("value")==choice), None)
        if not opt:
            continue
        recaps.append({
            "situation": st.get("situation",""),
            "chosen_label": opt.get("label",""),
            "chosen_consequence": opt.get("consequence",""),
            "other_labels": [o.get("label","") for o in st.get("options", []) if o.get("value")!=choice],
            "step_title": st.get("title", f"Step {idx}"),
            "step_index": idx
        })
    return recaps

def story_from_progress(seq: dict, progress: list, upto_step_exclusive: int):
    """Intro + consequences of chosen options up to (but not including) 'upto_step_exclusive'."""
    chunks = [seq.get("intro","")]
    steps = seq.get("steps", [])
    upto = max(0, min(upto_step_exclusive, len(progress), len(steps)))
    for idx in range(upto):
        st = steps[idx]
        choice = progress[idx]
        opt = next((o for o in st.get("options", []) if o.get("value")==choice), None)
        if opt and opt.get("consequence"):
            chunks.append(opt["consequence"])
    return " ".join([c for c in chunks if c]).strip()

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

# ====== Run logging (Profile System) ======
def _start_run(run_type, sid, title, total_steps, prefs=None):
    """Create a Runs row and stash run_id in session."""
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

# ===== Heuristic analyzer for strengths/weaknesses =====
ETH_DIMENSIONS = ["Distinction", "Proportionality", "Necessity", "Precaution"]
KEYWORDS = {
    "Distinction": [
        "civilian", "non-combatant", "evacuate", "warning", "shelter", "medical", "protected",
        "avoid collateral", "separate", "identify", "verify target", "double-check"
    ],
    "Proportionality": [
        "minimize damage", "limited strike", "contain", "low-yield", "warning shot", "tight rules",
        "de-escalate", "mitigate", "reduce risk", "no heavy", "scale down"
    ],
    "Necessity": [
        "mission critical", "necessary", "objective", "requires", "no alternative", "immediate need",
        "essential", "decisive", "time-sensitive"
    ],
    "Precaution": [
        "recon", "reconnaissance", "intel", "delay", "pause", "observe", "surveillance", "route change",
        "divert", "seek confirmation", "assess", "caution"
    ],
}

def analyze_decisions(rows):
    scores = {d: 0 for d in ETH_DIMENSIONS}
    for r in rows:
        blob = f"{r['option_label']} {r['option_consequence']}".lower()
        for dim, kws in KEYWORDS.items():
            if any(kw in blob for kw in kws):
                scores[dim] += 1
        if any(k in blob for k in ["warn", "evacuate", "reschedule", "verify", "confirm"]):
            scores["Precaution"] += 0.5

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    strengths = [f"{k} (+{int(v)})" for k, v in ordered[:2]]
    weaknesses = [f"{k} (+{int(v)})" for k, v in ordered[-2:]]

    weakest_dims = [k for k, _ in ordered[-2:]]
    practice_tags = []
    if "Distinction" in weakest_dims:
        practice_tags.append("civilian-dense urban checkpoint")
    if "Proportionality" in weakest_dims:
        practice_tags.append("time-sensitive target near clinic")
    if "Necessity" in weakest_dims:
        practice_tags.append("non-critical target under high risk")
    if "Precaution" in weakest_dims:
        practice_tags.append("limited intel—recon vs. action")

    return scores, strengths, weaknesses, practice_tags

# ====== Survey / Reflection (MCQ + Text) ======
PRE_MCQ = [
    {
        "question": "How confident are you in understanding this situation?",
        "name": "pre_confidence",
        "choices": [("Very low","1"),("Low","2"),("Medium","3"),("High","4"),("Very high","5")]
    },
    {
        "question": "Which factor matters most before deciding?",
        "name": "pre_factor",
        "choices": [("Civilian safety","CIV"),("Mission success","MS"),("Time pressure","TP"),("Intel quality","IQ")]
    }
]

POST_MCQ = [
    {
        "question": "How satisfied are you with your decision?",
        "name": "post_satisfaction",
        "choices": [("Very low","1"),("Low","2"),("Medium","3"),("High","4"),("Very high","5")]
    },
    {
        "question": "Your choice most likely…",
        "name": "post_effect",
        "choices": [("Reduced risk","RR"),("Kept risk similar","KS"),("Increased risk","IR"),("Unsure","UN")]
    }
]

SURVEY_PRE_PROB = 0.30
SURVEY_POST_PROB = 0.30

def _score_from_mcq(name: str, value: str):
    if name.endswith("confidence") or name.endswith("satisfaction"):
        try:
            n = int(value)
            return (n - 3) / 2.0  # 1..5 -> -1..+1
        except Exception:
            return 0.0
    if name.endswith("effect"):
        table = {"RR": 0.6, "KS": 0.0, "IR": -0.6, "UN": 0.0}
        return table.get(value, 0.0)
    return 0.0

def analyze_sentiment(text: str):
    if not text or not text.strip():
        return 0.0, "neutral"
    t = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = [w for w in t.split() if w]
    POS_WORDS = {"confident","calm","clear","safe","protected","improved","better","satisfied","relieved","balanced","careful","responsible"}
    NEG_WORDS = {"uncertain","worried","afraid","unsafe","risky","bad","worse","regret","concerned","anxious","dangerous","reckless","harm"}
    pos = sum(1 for w in tokens if w in POS_WORDS)
    neg = sum(1 for w in tokens if w in NEG_WORDS)
    score = 0.0 if (pos+neg)==0 else (pos - neg) / (pos + neg)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return round(score, 3), label

def _save_reflection(run_id: int, step_index: int, phase: str,
                     question: str, response_text: str = "",
                     choice_value: str = "", choice_label: str = ""):
    text_score, _ = analyze_sentiment(response_text or "")
    name_key = "confidence" if "confident" in question.lower() else \
               "satisfaction" if "satisfied" in question.lower() else \
               "effect" if ("likely" in question.lower() or "choice most" in question.lower() or "effect" in question.lower()) else \
               "generic"
    mcq_score = _score_from_mcq(name_key, choice_value) if choice_value else 0.0
    final_score = round(max(min(text_score + mcq_score, 1.0), -1.0), 3)
    final_label = "positive" if final_score > 0.2 else "negative" if final_score < -0.2 else "neutral"

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO RunReflections(run_id, step_index, phase, question_text, response_text,
                                   sentiment_score, sentiment_label, choice_value, choice_label)
        VALUES(?,?,?,?,?,?,?,?,?)
    """, (run_id, step_index, phase, question, response_text, final_score, final_label, choice_value, choice_label))
    conn.commit()
    conn.close()

def _maybe_pick_pre_question(step: int):
    sp = session.setdefault("survey_pre", {})
    if str(step) in sp:
        return sp[str(step)]
    if random.random() < SURVEY_PRE_PROB:
        if random.random() < 0.4:
            q = random.choice(PRE_MCQ)
            data = {"kind":"mcq", "question": q["question"], "name": q["name"], "choices": q["choices"]}
        else:
            qtext = random.choice([
                "Before deciding, what concerns are top-of-mind and why?",
                "What additional intel would most reduce your uncertainty right now?"
            ])
            data = {"kind":"text", "question": qtext, "name":"pre_text"}
        sp[str(step)] = data
        session["survey_pre"] = sp
        return data
    return None

def _maybe_prepare_post_question(step: int):
    sp = session.setdefault("survey_post", {})
    if str(step) in sp:
        return sp[str(step)]
    if random.random() < SURVEY_POST_PROB:
        if random.random() < 0.5:
            q = random.choice(POST_MCQ)
            data = {"kind":"mcq", "question": q["question"], "name": q["name"], "choices": q["choices"]}
        else:
            qtext = random.choice([
                "After deciding, what possible harms would you monitor closely?",
                "If you could revise your choice, what would you change and why?"
            ])
            data = {"kind":"text", "question": qtext, "name":"post_text"}
        sp[str(step)] = data
        session["survey_post"] = sp
        return data
    return None

# ========= GPT client (for scenario + reflective journal) =========

def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            mid = parts[1]
            if mid.lower().startswith("json"):
                mid = mid.split("\n", 1)[-1]
            return mid
        return t.replace("```", "")
    return t

def _coerce_gpt_sequence(seq: dict) -> dict:
    if not isinstance(seq, dict):
        raise ValueError("Generated scenario is not a JSON object.")
    seq["id"] = "G"
    steps = seq.get("steps") or []
    fixed_steps = []
    for i, st in enumerate(steps[:5], start=1):
        st = dict(st) if isinstance(st, dict) else {}
        st["id"] = int(st.get("id", i))
        opts = st.get("options")
        if isinstance(opts, dict):
            norm = []
            for k, v in list(opts.items())[:4]:
                if isinstance(v, dict):
                    norm.append({"value": k, "label": v.get("label") or str(v.get("text") or k),
                                 "consequence": v.get("consequence", ""), "hint": v.get("hint", "")})
                else:
                    norm.append({"value": k, "label": str(v), "consequence": "", "hint": ""})
            opts = norm
        elif not isinstance(opts, list):
            opts = []
        while len(opts) < 4:
            opts.append({"value": "", "label": "—", "consequence": "", "hint": ""})
        opts = opts[:4]
        letters = ["A", "B", "C", "D"]
        fixed_opts = []
        for j, opt in enumerate(opts):
            label = (opt.get("label") or "").strip() or f"Option {letters[j]}"
            cons  = (opt.get("consequence") or "").strip()
            hint  = (opt.get("hint") or "").strip()
            val   = (opt.get("value") or "").strip().upper()
            m = re.match(r"([A-D])\d?", val)
            letter = m.group(1) if m else letters[j]
            fixed_val = f"{letter}{st['id']}"
            fixed_opts.append({"value": fixed_val, "label": label, "consequence": cons, "hint": hint})
        st["options"] = fixed_opts
        fixed_steps.append(st)
    seq["steps"] = fixed_steps
    seq["title"] = (seq.get("title") or "Ethics-in-War Scenario").strip()
    seq["intro"] = (seq.get("intro") or "").strip()
    return seq

def _validate_generated_sequence_gpt(seq: dict):
    if not isinstance(seq, dict):
        raise ValueError("Generated scenario is not a JSON object.")
    if seq.get("id") != "G":
        raise ValueError("Generated scenario must have id='G'.")
    if not seq.get("title"):
        raise ValueError("Generated scenario missing 'title'.")
    if "intro" not in seq or not isinstance(seq["intro"], str) or len(seq["intro"].strip()) == 0:
        raise ValueError("Generated scenario must include a non-empty 'intro'.")
    steps = seq.get("steps")
    if not isinstance(steps, list) or len(steps) != 5:
        raise ValueError("Generated scenario must include exactly 5 steps.")
    for st in steps:
        for k in ("id","title","situation","question","options"):
            if k not in st:
                raise ValueError(f"Step missing '{k}'.")
        if not isinstance(st["id"], int) or st["id"] not in (1,2,3,4,5):
            raise ValueError("Step id must be 1..5.")
        if not isinstance(st["situation"], str) or len(st["situation"].strip()) < 200:
            raise ValueError("Each step's 'situation' should be at least a paragraph (200+ chars).")
        opts = st["options"]
        if not isinstance(opts, list) or len(opts) != 4:
            raise ValueError("Each step must have exactly 4 options.")
        expected = {f"A{st['id']}", f"B{st['id']}", f"C{st['id']}", f"D{st['id']}"}
        seen = {opt.get("value") for opt in opts}
        if seen != expected:
            raise ValueError(f"Options must be exactly {sorted(expected)} for step {st['id']}.")

def generate_ethics_sequence_with_llm(prefs: dict) -> dict:
    if not _client:
        raise RuntimeError("OPENAI_API_KEY not set")
    system = (
        "You generate a 5-step ethics-in-war decision scenario as STRICT JSON only (no prose outside JSON).\n"
        "Schema:\n"
        "{\n"
        '  \"id\":\"G\", \"title\":\"...\", \"intro\":\"(120–200 words background)\",\n'
        '  \"steps\":[\n'
        '    {\"id\":1,\"title\":\"...\",\"situation\":\"(>=200 chars paragraph)\",\"question\":\"...\",\n'
        '     \"options\":[{\"value\":\"A1\",\"label\":\"...\",\"consequence\":\"(1–3 sentences)\",\"hint\":\"(short tip)\"},'
        '                 {\"value\":\"B1\",\"label\":\"...\",\"consequence\":\"...\",\"hint\":\"...\"},'
        '                 {\"value\":\"C1\",\"label\":\"...\",\"consequence\":\"...\",\"hint\":\"...\"},'
        '                 {\"value\":\"D1\",\"label\":\"...\",\"consequence\":\"...\",\"hint\":\"...\"}]},\n'
        '    {\"id\":2, ... A2/B2/C2/D2}, {\"id\":3, ...}, {\"id\":4, ...}, {\"id\":5, ...}\n'
        "  ]\n"
        "}\n"
        "Constraints: respectful, age-appropriate for teens, no graphic detail; avoid naming modern political actors; "
        "align with IHL/ROE; total length < 1800 words. Write fluent, neutral English.\n"
        "IMPORTANT: The top-level field id MUST be exactly the string \"G\"."
    )
    user = (
        "Preferences (JSON):\n" + json.dumps(prefs, ensure_ascii=False) +
        "\nUse the selected historical war/theatre as educational context without graphic detail. "
        "Do not include contemporary political propaganda. Output JSON only."
    )
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    content = resp.choices[0].message.content or "{}"
    content = _strip_json_fences(content)
    try:
        with open("static/generated/last_gpt_raw.json", "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass
    try:
        seq = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise ValueError("Model did not return valid JSON.")
        seq = json.loads(m.group(0))
    seq = _coerce_gpt_sequence(seq)
    try:
        with open("static/generated/last_gpt_fixed.json", "w", encoding="utf-8") as f:
            json.dump(seq, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _validate_generated_sequence_gpt(seq)
    return seq

# ===== Reflective Journal (GPT) =====
def _save_run_journal(run_id: int, journal: dict):
    conn = get_db()
    c = conn.cursor()
    c.execute("REPLACE INTO RunJournals(run_id, journal_json) VALUES(?,?)",
              (run_id, json.dumps(journal, ensure_ascii=False)))
    conn.commit()
    conn.close()

def _get_run_journal(run_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT journal_json FROM RunJournals WHERE run_id=?", (run_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row["journal_json"])
    except Exception:
        return None

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

def generate_reflective_journal_gpt(run_id: int, seq: dict, decisions_view: list):
    if not _client:
        # Return a stub journal if API key is not configured
        journal = {
            "narrative_summary": "Reflective journal generation skipped (no OPENAI_API_KEY).",
            "decision_rationale": [],
            "framework_self_assessment": {k:"N/A" for k in ["Distinction","Proportionality","Necessity","Precaution"]},
            "emotional_influences": "N/A",
            "growth_targets": [],
            "action_items": []
        }
        _save_run_journal(run_id, journal)
        return journal

    # fetch reflections tied to this run
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT step_index, phase, question_text, response_text, sentiment_score, sentiment_label,
               choice_value, choice_label
        FROM RunReflections
        WHERE run_id=?
        ORDER BY step_index, phase
    """, (run_id,))
    refl = [dict(r) for r in c.fetchall()]
    conn.close()

    payload = {
        "scenario_title": seq.get("title"),
        "intro": seq.get("intro"),
        "steps": [{
            "id": st.get("id"),
            "title": st.get("title"),
            "question": st.get("question"),
            "options": [{"label": o.get("label"), "consequence": o.get("consequence")} for o in st.get("options", [])]
        } for st in seq.get("steps", [])],
        "decisions": decisions_view,
        "reflections": refl,
        "dimensions": ["Distinction","Proportionality","Necessity","Precaution"]
    }

    sys = (
        "You are a tutor generating a concise reflective journal in STRICT JSON. "
        "Use IHL-aligned terms, be neutral and age-appropriate."
    )
    usr = (
        "Compile a reflective journal from this data.\n"
        "Return ONLY JSON with keys:\n"
        "{\n"
        '  "narrative_summary": "(120-180 words)",\n'
        '  "decision_rationale": [{"step":1,"why":"..."}, ...],\n'
        '  "framework_self_assessment": {"Distinction":"(1-2 sentences)","Proportionality":"...",'
        '                               "Necessity":"...","Precaution":"..."},\n'
        '  "emotional_influences": "(80-140 words; synthesize pre/post reflections + sentiments)",\n'
        '  "growth_targets": ["short, actionable bullets (3-5)"],\n'
        '  "action_items": ["practice ideas (3-5)"]\n'
        "}\n"
        "Do not include names of modern political actors. Avoid graphic details."
        "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False)
    )

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ]
        )
        content = resp.choices[0].message.content or "{}"
        content = _strip_json_fences(content)
        journal = json.loads(content)
    except Exception as e:
        journal = {
            "narrative_summary": "Reflective journal generation failed.",
            "decision_rationale": [],
            "framework_self_assessment": {k:"N/A" for k in ["Distinction","Proportionality","Necessity","Precaution"]},
            "emotional_influences": "N/A",
            "growth_targets": [],
            "action_items": [],
            "error": str(e)
        }

    _save_run_journal(run_id, journal)
    return journal

def _get_active_gpt_sequence():
    run_id = session.get("run_id")
    return _get_run_sequence(run_id)

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
        else:
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
            c.execute('INSERT INTO Users (username, password, gender, age_group, preferred_war, interest) VALUES (?,?,?,?,?,?)',
                      (username, password, gender, age_group, preferred_war, interest))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
    return render_template("register.html")

# ========= Scenario routing (STATIC: choose A / B / C / D) =========
@app.route("/scenario", methods=["GET"])
def scenario_choose():
    return redirect(url_for("scenario_start", sid="A"))

@app.route("/scenario/<sid>/start", methods=["GET"])
@login_required
def scenario_start(sid):
    if sid not in SCENARIO_SEQUENCES:
        return redirect(url_for("scenario_start", sid="A"))
    session["scenario_sid"] = sid
    session["scenario_progress"] = []            # keep tiny
    session["survey_pre"] = {}
    session["survey_post"] = {}
    seq = SCENARIO_SEQUENCES[sid]
    _start_run(run_type="static", sid=sid, title=seq.get("title", f"Scenario {sid}"),
               total_steps=len(seq["steps"]), prefs=None)
    return redirect(url_for("scenario_step", sid=sid, step=1))

@app.route("/scenario/<sid>/step/<int:step>", methods=["GET", "POST"])
@login_required
def scenario_step(sid, step):
    if sid not in SCENARIO_SEQUENCES:
        return redirect(url_for("scenario_start", sid="A"))
    seq = SCENARIO_SEQUENCES[sid]
    steps = seq["steps"]
    if step < 1 or step > len(steps):
        return redirect(url_for("scenario_start", sid=sid))

    progress = session.get("scenario_progress", [])
    current = steps[step - 1]

    labels_map = build_label_map(steps)
    decisions_so_far = []
    for idx, choice in enumerate(progress[:step-1], start=1):
        decisions_so_far.append(f"{steps[idx-1]['title']}: {labels_map[idx].get(choice, choice)}")

    if request.method == "POST":
        run_id = session.get("run_id")
        if run_id:
            if request.form.get("pre_kind") == "text" and request.form.get("pre_question"):
                _save_reflection(run_id, step, "pre",
                                 request.form.get("pre_question",""),
                                 response_text=request.form.get("pre_text","").strip())
            if request.form.get("pre_kind") == "mcq" and request.form.get("pre_question"):
                _save_reflection(run_id, step, "pre",
                                 request.form.get("pre_question",""),
                                 choice_value=request.form.get("pre_choice_value",""),
                                 choice_label=request.form.get("pre_choice_label",""))

        choice = request.form.get("choice")
        if not choice:
            flash("Please select one option.", "warning")
        else:
            if len(progress) >= step:
                progress[step - 1] = choice
            else:
                progress.append(choice)
            session["scenario_progress"] = progress

            chosen_opt = next(o for o in current["options"] if o["value"] == choice)
            run_id = session.get("run_id")
            if run_id:
                _log_decision(
                    run_id=run_id,
                    step_index=step,
                    opt_value=choice,
                    opt_label=chosen_opt["label"],
                    opt_consequence=chosen_opt.get("consequence", "")
                )
                if request.form.get("post_kind") == "text" and request.form.get("post_question"):
                    _save_reflection(run_id, step, "post",
                                     request.form.get("post_question",""),
                                     response_text=request.form.get("post_text","").strip())
                if request.form.get("post_kind") == "mcq" and request.form.get("post_question"):
                    _save_reflection(run_id, step, "post",
                                     request.form.get("post_question",""),
                                     choice_value=request.form.get("post_choice_value",""),
                                     choice_label=request.form.get("post_choice_label",""))

            if step < len(steps):
                return redirect(url_for("scenario_step", sid=sid, step=step + 1))
            else:
                return redirect(url_for("scenario_result", sid=sid))

    # derive previous recap & story without storing bulky text in session
    prev_recap = None
    if step > 1 and progress and len(progress) >= (step - 1):
        derived = derive_recaps(seq, progress)
        prev_recap = derived[step - 2] if len(derived) >= (step - 1) else None

    story_text = story_from_progress(seq, progress, step - 1)
    preselected = progress[step - 1] if len(progress) >= step else None
    pre_survey = _maybe_pick_pre_question(step)
    post_survey = _maybe_prepare_post_question(step)

    return render_template(
        "scenario_step.html",
        scenario_id=seq["id"],
        scenario_title=seq["title"],
        step=current,
        step_index=step,
        total_steps=len(steps),
        selected=preselected,
        decisions_so_far=decisions_so_far,
        situation_text=current.get("situation", ""),
        question_text=current.get("question", ""),
        prev_recap=prev_recap,
        story_so_far=story_text,
        pre_survey=pre_survey,
        post_survey=post_survey,
        is_last=(step == len(steps))
    )

@app.route("/scenario/<sid>/result", methods=["GET"])
@login_required
def scenario_result(sid):
    if sid not in SCENARIO_SEQUENCES:
        return redirect(url_for("scenario_start", sid="A"))

    seq = SCENARIO_SEQUENCES[sid]
    steps = seq["steps"]
    progress = session.get("scenario_progress", [])
    if len(progress) != len(steps):
        return redirect(url_for("scenario_start", sid=sid))

    decisions_view = []
    for idx, choice in enumerate(progress, start=1):
        st = steps[idx - 1]
        opt = next(o for o in st["options"] if o["value"] == choice)
        decisions_view.append({
            "step_id": idx,
            "title": st["title"],
            "label": opt["label"],
            "consequence": opt.get("consequence", "")
        })

    run_id = session.get("run_id")
    if run_id:
        _finish_run(run_id)

    path_letters = first_letter_path(progress)
    ending = OUTCOME_MAP.get(path_letters, f"No authored ending for path '{path_letters}'.")
    full_story = story_from_progress(seq, progress, len(steps))

    # No reflective journal for static runs
    run_journal = None

    return render_template(
        "scenario_result.html",
        scenario_id=seq["id"],
        scenario_title=seq["title"],
        scenario_intro=seq.get("intro", ""),
        decisions=decisions_view,
        full_story=full_story,
        ending=ending,
        path_letters=path_letters,
        timestamp=int(datetime.utcnow().timestamp()),
        run_journal=run_journal
    )
@app.route("/gpt-scenario", methods=["GET", "POST"])
@login_required
def gpt_scenario_prefs():
    if request.method == "POST":
        prefs = {
            "war": request.form.get("war", "").strip(),
            "theatre": request.form.get("theatre", "").strip(),
            "role": request.form.get("role", "field-commander").strip(),
            "civilians": request.form.get("civilians", "present").strip(),
            "tone": request.form.get("tone", "serious & age-appropriate").strip(),
            "goal": request.form.get("goal", "").strip()
        }
        try:
            seq = generate_ethics_sequence_with_llm(prefs)
        except Exception as e:
            flash(f"Scenario generation failed: {e}", "danger")
            return redirect(url_for("gpt_scenario_prefs"))

        # Start run & reset transient session state
        session["scenario_sid"] = "G"
        session["scenario_progress"] = []
        session["survey_pre"] = {}
        session["survey_post"] = {}

        run_id = _start_run(
            run_type="gpt",
            sid="G",
            title=seq.get("title", "GPT Scenario"),
            total_steps=len(seq.get("steps", [])),
            prefs=prefs
        )

        # Save initial sequence
        _save_run_sequence(run_id, seq)

        # Optional: generate watercolor cover image
        make_image = request.form.get("make_image") == "1"
        if make_image:
            img_rel_path = generate_watercolor_image_for_run(run_id, prefs, seq.get("title", ""))
            if img_rel_path:
                # Attach to sequence for templates to use as {{ seq.cover_image }}
                seq_with_img = dict(seq)
                seq_with_img["cover_image"] = img_rel_path  # e.g., "generated/run_12_cover.png"
                _save_run_sequence(run_id, seq_with_img)

        return redirect(url_for("gpt_scenario_step", step=1))

    # GET
    return render_template("gpt_scenario_prefs.html")
@app.route("/gpt-scenario/step/<int:step>", methods=["GET", "POST"])
@login_required
def gpt_scenario_step(step):
    seq = _get_active_gpt_sequence()
    if not seq:
        return redirect(url_for("gpt_scenario_prefs"))

    steps = seq["steps"]
    if step < 1 or step > len(steps):
        return redirect(url_for("gpt_scenario_prefs"))

    progress = session.get("scenario_progress", [])
    current = steps[step - 1]

    labels_map = build_label_map(steps)
    decisions_so_far = []
    for idx, choice in enumerate(progress[:step-1], start=1):
        decisions_so_far.append(f"{steps[idx-1]['title']}: {labels_map[idx].get(choice, choice)}")

    if request.method == "POST":
        run_id = session.get("run_id")

        # Save PRE reflection if any
        if run_id:
            if request.form.get("pre_kind") == "text" and request.form.get("pre_question"):
                _save_reflection(run_id, step, "pre",
                                 request.form.get("pre_question",""),
                                 response_text=request.form.get("pre_text","").strip())
            if request.form.get("pre_kind") == "mcq" and request.form.get("pre_question"):
                _save_reflection(run_id, step, "pre",
                                 request.form.get("pre_question",""),
                                 choice_value=request.form.get("pre_choice_value",""),
                                 choice_label=request.form.get("pre_choice_label",""))

        # Handle decision
        choice = request.form.get("choice")
        if not choice:
            flash("Please select one option.", "warning")
        else:
            if len(progress) >= step:
                progress[step - 1] = choice
            else:
                progress.append(choice)
            session["scenario_progress"] = progress
            session.modified = True  # ensure persistence before redirect

            chosen_opt = next(o for o in current["options"] if o["value"] == choice)
            if run_id:
                _log_decision(
                    run_id=run_id,
                    step_index=step,
                    opt_value=choice,
                    opt_label=chosen_opt["label"],
                    opt_consequence=chosen_opt.get("consequence", "")
                )
                # Save POST reflection if any
                if request.form.get("post_kind") == "text" and request.form.get("post_question"):
                    _save_reflection(run_id, step, "post",
                                     request.form.get("post_question",""),
                                     response_text=request.form.get("post_text","").strip())
                if request.form.get("post_kind") == "mcq" and request.form.get("post_question"):
                    _save_reflection(run_id, step, "post",
                                     request.form.get("post_question",""),
                                     choice_value=request.form.get("post_choice_value",""),
                                     choice_label=request.form.get("post_choice_label",""))

            if step < len(steps):
                return redirect(url_for("gpt_scenario_step", step=step + 1))
            else:
                return redirect(url_for("gpt_scenario_result"))

    # sidebar recap/story
    prev_recap = None
    if step > 1 and progress and len(progress) >= (step - 1):
        derived = derive_recaps(seq, progress)
        prev_recap = derived[step - 2] if len(derived) >= (step - 1) else None

    story_text = story_from_progress(seq, progress, step - 1)
    preselected = progress[step - 1] if len(progress) >= step else None
    pre_survey = _maybe_pick_pre_question(step)
    post_survey = _maybe_prepare_post_question(step)

    return render_template(
        "scenario_step.html",
        scenario_id=seq["id"],
        scenario_title=seq["title"],
        step=current,
        step_index=step,
        total_steps=len(steps),
        selected=preselected,
        decisions_so_far=decisions_so_far,
        situation_text=current.get("situation", ""),
        question_text=current.get("question", ""),
        prev_recap=prev_recap,
        story_so_far=story_text,
        pre_survey=pre_survey,
        post_survey=post_survey,
        is_last=(step == len(steps)),
        # NEW:
        cover_image=seq.get("cover_image")
    )

@app.route("/gpt-scenario/result", methods=["GET"])
@login_required
def gpt_scenario_result():
    run_id = session.get("run_id")
    if not run_id:
        return redirect(url_for("gpt_scenario_prefs"))

    seq = _get_run_sequence(run_id)
    if not seq:
        return redirect(url_for("gpt_scenario_prefs"))

    steps = seq["steps"]
    progress = session.get("scenario_progress", [])

    # Resume if incomplete
    if len(progress) < len(steps):
        next_step = len(progress) + 1
        return redirect(url_for("gpt_scenario_step", step=next_step))

    decisions_view = []
    for idx, choice in enumerate(progress, start=1):
        st = steps[idx - 1]
        opt = next(o for o in st["options"] if o["value"] == choice)
        decisions_view.append({
            "step_id": idx,
            "title": st["title"],
            "label": opt["label"],
            "consequence": opt.get("consequence", "")
        })

    _finish_run(run_id)

    full_story = story_from_progress(seq, progress, len(steps))
    ending = "Your decisions balanced mission objectives with proportionality and distinction under IHL."

    run_journal = _get_run_journal(run_id)
    if not run_journal:
        run_journal = generate_reflective_journal_gpt(run_id, seq, decisions_view)

    return render_template(
        "scenario_result.html",
        scenario_id=seq["id"],
        scenario_title=seq["title"],
        scenario_intro=seq.get("intro", ""),
        decisions=decisions_view,
        full_story=full_story,
        ending=ending,
        path_letters="".join(c[0] for c in progress),
        timestamp=int(datetime.utcnow().timestamp()),
        run_journal=run_journal,
        # NEW:
        cover_image=seq.get("cover_image")
    )

def _load_sequence_for_run(run_row):
    """
    Return the sequence dict for this run (static or GPT).
    For static, read from in-memory SCENARIO_SEQUENCES by sid.
    For GPT, read from RunSequences (DB).
    """
    if run_row["run_type"] == "static":
        seq = SCENARIO_SEQUENCES.get(run_row["sid"])
        return seq or {"title": run_row["title"], "intro": "", "steps": []}
    # GPT
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT sequence_json FROM RunSequences WHERE run_id=?", (run_row["id"],))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"title": run_row["title"], "intro": "", "steps": []}
    try:
        return json.loads(row["sequence_json"])
    except Exception:
        return {"title": run_row["title"], "intro": "", "steps": []}

def _build_run_decisions_view(run_row, seq, decisions_rows, reflections_rows):
    """
    Build the decisions list used by the profile template, and compute path letters.
    Each item: { step_index, step_title, chosen_value, chosen_label, chosen_consequence, all_options[] }
    """
    steps = seq.get("steps", [])
    # quick map for reflections by step
    refl_by_step = {}
    for rf in reflections_rows:
        refl_by_step.setdefault(rf["step_index"], []).append(dict(rf))

    view = []
    letters = []
    for d in decisions_rows:
        idx = int(d["step_index"])
        st = steps[idx-1] if 1 <= idx <= len(steps) else {}
        chosen_value = d["option_value"] or ""
        chosen_label = d["option_label"] or ""
        chosen_cons  = d["option_consequence"] or ""
        letters.append((chosen_value[:1] or "?").upper())

        all_opts = []
        for o in st.get("options", []):
            all_opts.append({
                "value": o.get("value",""),
                "label": o.get("label",""),
                "consequence": o.get("consequence","")
            })

        view.append({
            "step_index": idx,
            "step_title": st.get("title", f"Step {idx}"),
            "chosen_value": chosen_value,
            "chosen_label": chosen_label,
            "chosen_consequence": chosen_cons,
            "all_options": all_opts,
            "reflections": refl_by_step.get(idx, [])
        })

    path_letters = "".join(letters)
    return view, path_letters
# --- utils_profile_stats.py (or inline above your route) ---------------------
import re
from collections import Counter, defaultdict

_DIM_KEYS = ["Distinction","Proportionality","Necessity","Precaution"]
_DIM_PATTERNS = {
    "Distinction":      re.compile(r"\b(civili(an|ans)|non[- ]combatant|distinction|collateral)\b", re.I),
    "Proportionality":  re.compile(r"\b(proportion(al|ality)|escalat(e|ion)|minimum|force)\b", re.I),
    "Necessity":        re.compile(r"\b(necess(ar|ity)|essential|unavoidable|must)\b", re.I),
    "Precaution":       re.compile(r"\b(precaution|warn(ing)?|verify|double[- ]check|confirm)\b", re.I),
}

# Map common labels to compact codes expected by charts
_PRE_FACTOR_CODES = {
    "civilian": "CIV", "civ": "CIV",
    "mission": "MS", "success": "MS", "ms": "MS",
    "time": "TP", "pressure": "TP", "tp": "TP",
    "intel": "IQ", "intelligence": "IQ", "iq": "IQ",
}
_POST_EFFECT_CODES = {
    "reduced": "RR", "lower": "RR", "decrease": "RR", "rr": "RR",
    "similar": "KS", "kept": "KS", "same": "KS", "ks": "KS",
    "increased": "IR", "higher": "IR", "increase": "IR", "ir": "IR",
    "unsure": "UN", "unknown": "UN", "not sure": "UN", "un": "UN",
}

def _kw_hit_count(text: str) -> dict:
    """Count keyword hits for each ethics dimension in a piece of text."""
    hits = {k: 0 for k in _DIM_KEYS}
    if not text:
        return hits
    for k, pat in _DIM_PATTERNS.items():
        hits[k] = len(pat.findall(text))
    return hits

def _norm_dim_counts(raw_counts: dict) -> dict:
    total = sum(raw_counts.values()) or 1
    return {k: raw_counts.get(k, 0) / total for k in _DIM_KEYS}

def _safe_avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals)/len(vals), 3) if vals else 0.0

def _label_has_any(label: str, mapping: dict) -> str | None:
    if not label:
        return None
    s = label.lower()
    # exact code passthrough
    if s.upper() in mapping.values():
        return s.upper()
    for key, code in mapping.items():
        if key in s:
            return code
    return None

def _compute_stats_for_user(username: str) -> dict:
    """Builds the 'stats' structure used by profile.html charts."""
    conn = get_db()
    c = conn.cursor()

    # --- counts
    c.execute("SELECT COUNT(*) FROM Runs WHERE username=?", (username,))
    runs_cnt = c.fetchone()[0]

    c.execute("""SELECT COUNT(*)
                 FROM RunDecisions d JOIN Runs r ON r.id=d.run_id
                 WHERE r.username=?""", (username,))
    decisions_cnt = c.fetchone()[0]

    c.execute("""SELECT COUNT(*)
                 FROM RunReflections rf JOIN Runs r ON r.id=rf.run_id
                 WHERE r.username=?""", (username,))
    reflections_cnt = c.fetchone()[0]

    # --- ethics dimensions from decisions: accumulate option labels+consequences
    c.execute("""
      SELECT d.option_label, d.option_consequence
      FROM RunDecisions d JOIN Runs r ON r.id=d.run_id
      WHERE r.username=?
    """, (username,))
    dim_raw = Counter({k: 0 for k in _DIM_KEYS})
    for opt_label, opt_cons in c.fetchall():
        chunk = " ".join([opt_label or "", opt_cons or ""])
        hits = _kw_hit_count(chunk)
        for k, v in hits.items():
            dim_raw[k] += v
    dim_raw_dict = {k: int(dim_raw[k]) for k in _DIM_KEYS}
    dim_norm_dict = _norm_dim_counts(dim_raw_dict)

    # --- reflections: survey-like fields & sentiment
    # We assume RunReflections has: phase (pre/post), sentiment_score (-1..1), sentiment_label,
    # and optionally numeric survey scores in response_text or choice_* columns.
    c.execute("""
      SELECT phase, sentiment_score, sentiment_label, choice_value, choice_label, response_text
      FROM RunReflections rf JOIN Runs r ON r.id=rf.run_id
      WHERE r.username=?
    """, (username,))
    pre_conf_scores = []     # numeric 1..5 if present
    post_sat_scores = []     # numeric 1..5 if present
    pre_sent, post_sent = [], []
    sent_dist = Counter({"positive":0, "neutral":0, "negative":0})
    pre_factor_dist = Counter({"CIV":0,"MS":0,"TP":0,"IQ":0})
    post_effect_dist = Counter({"RR":0,"KS":0,"IR":0,"UN":0})

    for phase, s_score, s_label, choice_val, choice_label, resp in c.fetchall():
        # sentiment
        if s_label:
            key = s_label.lower()
            if key.startswith("pos"): sent_dist["positive"] += 1
            elif key.startswith("neg"): sent_dist["negative"] += 1
            else: sent_dist["neutral"] += 1
        if s_score is not None:
            (pre_sent if phase == "pre" else post_sent).append(float(s_score))

        # numeric survey values: try choice_value first, fallback to parsing response_text
        if phase == "pre":
            # confidence (1..5) may be stored in choice_value or like "confidence=3"
            val = None
            if isinstance(choice_val, (int, float)):
                val = float(choice_val)
            elif resp:
                m = re.search(r"(confidence|conf)\s*=\s*([1-5])", resp, re.I)
                if m: val = float(m.group(2))
            if val is not None: pre_conf_scores.append(val)

            code = _label_has_any(choice_label or "", _PRE_FACTOR_CODES)
            if code: pre_factor_dist[code] += 1

        elif phase == "post":
            # satisfaction (1..5)
            val = None
            if isinstance(choice_val, (int, float)):
                val = float(choice_val)
            elif resp:
                m = re.search(r"(satisf(action)?|sat)\s*=\s*([1-5])", resp, re.I)
                if m: val = float(m.group(3))
            if val is not None: post_sat_scores.append(val)

            code = _label_has_any(choice_label or "", _POST_EFFECT_CODES)
            if code: post_effect_dist[code] += 1

    conn.close()

    stats = {
      "counts": {
        "runs": runs_cnt,
        "decisions": decisions_cnt,
        "reflections": reflections_cnt
      },
      "dimensions_raw": dim_raw_dict,
      "dimensions_norm": dim_norm_dict,
      "survey": {
        "pre_confidence_avg": _safe_avg(pre_conf_scores),
        "post_satisfaction_avg": _safe_avg(post_sat_scores),
        "pre_factor_dist": {k: pre_factor_dist[k] for k in ["CIV","MS","TP","IQ"]},
        "post_effect_dist": {k: post_effect_dist[k] for k in ["RR","KS","IR","UN"]},
      },
      "sentiment": {
        "pre_avg": _safe_avg(pre_sent),
        "post_avg": _safe_avg(post_sent),
        "dist": dict(sent_dist)
      }
    }
    return stats

# ---------- Updated /profile route (populates chosen decisions + stats) ----------

@app.route("/profile", methods=["GET"])
@login_required
def profile():
    username = session["username"]

    conn = get_db()
    c = conn.cursor()

    # Recent runs
    c.execute("""
      SELECT id, run_type, sid, title, total_steps, started_at, finished_at
      FROM Runs
      WHERE username=?
      ORDER BY COALESCE(finished_at, started_at) DESC
      LIMIT 12
    """, (username,))
    runs_rows = [dict(r) for r in c.fetchall()]

    runs = []
    for r in runs_rows:
        # decisions & reflections for this run
        c.execute("SELECT * FROM RunDecisions WHERE run_id=? ORDER BY step_index", (r["id"],))
        decisions_rows = [dict(x) for x in c.fetchall()]

        c.execute("""
          SELECT step_index, phase, question_text, response_text, sentiment_score, sentiment_label,
                 choice_value, choice_label, created_at
          FROM RunReflections
          WHERE run_id=?
          ORDER BY created_at
        """, (r["id"],))
        reflections_rows = [dict(x) for x in c.fetchall()]

        seq = _load_sequence_for_run(r)
        # intro excerpt
        intro = (seq.get("intro") or "").strip()
        intro_excerpt = (intro[:180] + "…") if len(intro) > 180 else intro

        decisions_view, path_letters = _build_run_decisions_view(r, seq, decisions_rows, reflections_rows)

        r_out = dict(r)
        r_out["scenario_intro_excerpt"] = intro_excerpt
        r_out["decisions"] = decisions_view
        r_out["reflections"] = reflections_rows
        r_out["path_letters"] = path_letters
        runs.append(r_out)

    conn.close()

    # Build stats for charts (no “Communication Traits” card needed)
    stats = _compute_stats_for_user(username)

    # You can still compute strengths/weaknesses for the small pills if you wish to keep them
    # (The new template you asked for removed the traits card; if you keep pills elsewhere, use below lines.)
    c2 = get_db()
    dcur = c2.cursor()
    dcur.execute("""
        SELECT d.option_label, d.option_consequence
        FROM Runs r
        JOIN RunDecisions d ON d.run_id = r.id
        WHERE r.username=?
    """, (username,))
    all_decisions_for_pills = dcur.fetchall()
    c2.close()
    dim_raw_pills, strengths, weaknesses, _ = analyze_decisions(all_decisions_for_pills)

    return render_template(
        "profile.html",
        runs=runs,
        stats=stats,
        # strengths/weaknesses are optional; safe to pass even if not used by the template
        strengths=strengths,
        weaknesses=weaknesses,
        scores=dim_raw_pills
    )
import base64

def _prompt_from_prefs(prefs: dict, title: str = "") -> str:
    # Build a short, safe, non-graphic prompt
    war     = prefs.get("war") or "historical conflict"
    theatre = prefs.get("theatre") or "relevant theatre/front"
    role    = prefs.get("role") or "field commander"
    tone    = prefs.get("tone") or "serious & age-appropriate"

    parts = [
        "Watercolor illustration, minimal detail, soft palette, gentle edges.",
        "Age-appropriate, respectful, NO graphic content, NO depictions of injury. NO TEXT AT ALL",
        f"Context: {war}, theatre/front: {theatre}, perspective: {role}.",
        "Focus on nonviolent elements: terrain, map marks, tents, radios, supply lines, vehicles at a distance, neutral symbols.",
        "Avoid faces close-up, avoid flags/emblems tied to modern politics, avoid combat action.",
    ]
    if title:
        parts.append(f"Title cue: {title}")
    # Keep it short so generation is snappy
    return " ".join(parts)
def generate_watercolor_image_for_run(run_id: int, prefs: dict, title: str = "") -> str | None:
    if not _client:
        return None
    try:
        prompt = _prompt_from_prefs(prefs, title)
        resp = _client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        b64 = resp.data[0].b64_json
        img = base64.b64decode(b64)

        out_dir = os.path.join("static", "generated")
        os.makedirs(out_dir, exist_ok=True)

        # absolute filesystem path for writing
        out_path = os.path.join(out_dir, f"run_{run_id}_cover.png")
        with open(out_path, "wb") as f:
            f.write(img)

        # IMPORTANT: return a path RELATIVE TO /static for url_for('static', filename=...)
        rel_path = f"generated/run_{run_id}_cover.png"
        return rel_path
    except Exception:
        return None

# ===== Dev server =====
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=8080)
