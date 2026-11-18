import os
import io
import re
import csv
import json
import math
import time
import base64
import random
import hashlib
import hmac
import difflib
import uuid
from collections import Counter
from datetime import date, timedelta
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ================= CONFIG =================

def _secret(name, default=""):
    try:
        v = st.secrets[name]
        if v:
            return str(v)
    except Exception:
        pass
    return os.getenv(name, default)

groq_key = _secret("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None
GROQ_MODEL = _secret("GROQ_MODEL")
TEACHER_PASSWORD = _secret("TEACHER_PASSWORD")

RAW = "data/raw"
USERS_FILE = "data/users.json"
USERS_DIR = "data/users"
LOG_FILE = "data/logs.jsonl"
VERIFIED_FILE = "data/verified.json"
DOUBTS_FILE = "data/doubts.json"
SYLLABUS_FILE = "data/syllabus.json"
PATTERN_FILE = "data/pattern.json"
MSG_FILE = "data/messages.json"
os.makedirs(RAW, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

st.set_page_config(page_title="AI Study Companion", page_icon="📚", layout="wide")

# ================= THEME =================
THEMES = {
    "Default": None,
    "🌙 Dark Blue": {"bg": "#0F172A", "side": "#1E293B", "text": "#E2E8F0", "inp": "#1E293B", "accent": "#60A5FA", "dark": True},
    "🌿 Soft Green": {"bg": "#F0FDF4", "side": "#DCFCE7", "text": "#14532D", "inp": "#DCFCE7", "accent": "#16A34A", "dark": False},
    "💜 Lavender": {"bg": "#F5F3FF", "side": "#EDE9FE", "text": "#2E1065", "inp": "#EDE9FE", "accent": "#7C3AED", "dark": False},
    "☀️ Warm Cream": {"bg": "#FFFBEB", "side": "#FEF3C7", "text": "#451A03", "inp": "#FEF3C7", "accent": "#D97706", "dark": False},
    "🌌 Aurora (animated)": {"bg": "linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1b6b5a)", "side": "rgba(15,32,39,0.88)", "text": "#E8F5F3", "inp": "#1c3a44", "accent": "#2DD4BF", "dark": True, "anim": True},
    "🌅 Sunset (animated)": {"bg": "linear-gradient(-45deg,#ff9a9e,#fad0c4,#fbc2eb,#a6c1ee)", "side": "rgba(255,255,255,0.6)", "text": "#3B1F2B", "inp": "#FFF1F5", "accent": "#E11D48", "dark": False, "anim": True},
    "🌊 Ocean": {"bg": "linear-gradient(160deg,#e0f2fe,#bae6fd,#a5f3fc)", "side": "rgba(255,255,255,0.65)", "text": "#0C4A6E", "inp": "#F0F9FF", "accent": "#0284C7", "dark": False},
    "💎 Neon Night": {"bg": "linear-gradient(135deg,#0b0b1e,#1a1040,#2a0a4a)", "side": "rgba(20,10,50,0.88)", "text": "#F5F3FF", "inp": "#241552", "accent": "#F0ABFC", "dark": True},
    "🌲 Forest": {"bg": "linear-gradient(160deg,#052e16,#14532d,#166534)", "side": "rgba(5,46,22,0.88)", "text": "#ECFDF5", "inp": "#14532d", "accent": "#4ADE80", "dark": True},
    "🌸 Rose Gold": {"bg": "linear-gradient(135deg,#fff1f2,#fecdd3,#fde68a)", "side": "rgba(255,255,255,0.65)", "text": "#4C0519", "inp": "#FFF7F8", "accent": "#BE123C", "dark": False},
    "📜 Paper (sepia)": {"bg": "#F4ECD8", "side": "#E9DDC2", "text": "#3B2F1E", "inp": "#FBF5E6", "accent": "#92400E", "dark": False},
    "🖤 AMOLED Black": {"bg": "#000000", "side": "#0A0A0A", "text": "#F5F5F5", "inp": "#141414", "accent": "#22D3EE", "dark": True},
    "🎨 Custom (apna rang)": "custom",
}
theme_name = st.sidebar.selectbox("🎨 Theme", list(THEMES.keys()))
_t = THEMES[theme_name]


def _is_dark(hexcolor):
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 140


if _t == "custom":
    _c1 = st.sidebar.color_picker("Background 1", "#1E1B4B")
    _c2 = st.sidebar.color_picker("Background 2", "#312E81")
    _ct = st.sidebar.color_picker("Text color", "#F8FAFC")
    _cs = st.sidebar.color_picker("Sidebar color", "#1E1B4B")
    _ca = st.sidebar.color_picker("Accent (buttons)", "#A78BFA")
    _t = {"bg": f"linear-gradient(135deg,{_c1},{_c2})", "side": _cs, "text": _ct,
          "inp": _cs, "accent": _ca, "dark": _is_dark(_c1)}

if _t:
    _glass = st.sidebar.checkbox("🪟 Glass effect (cards)", value=True)
    _dark = _t.get("dark", True)
    _accent = _t.get("accent", "#6C63FF")
    _inp = _t.get("inp", _t["side"])
    _anim = (
        "background-size: 300% 300%; animation: gradmove 18s ease infinite;"
        if _t.get("anim") else ""
    )
    _card = "rgba(255,255,255,0.07)" if _dark else "rgba(255,255,255,0.55)"
    _line = "rgba(255,255,255,0.18)" if _dark else "rgba(0,0,0,0.10)"
    _glass_css = f"""
        [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"], [data-testid="stChatMessage"],
        [data-testid="stForm"], [data-testid="stAlert"] {{
            background: {_card} !important; border: 1px solid {_line} !important;
            border-radius: 16px !important; backdrop-filter: blur(10px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.12);
        }}
        [data-testid="stMetric"] {{ padding: 10px 14px; }}
    """ if _glass else ""
    st.markdown(
        f"""<style>
        @keyframes gradmove {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .stApp {{ background: {_t['bg']}; background-attachment: fixed; color: {_t['text']}; {_anim} }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {_t['side']}; backdrop-filter: blur(12px); }}
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp span, .stApp li, .stApp caption, .stApp div[data-testid="stMetricValue"],
        .stApp div[data-testid="stMetricLabel"] {{ color: {_t['text']}; }}
        textarea, input {{ color: {_t['text']} !important; }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div, [data-testid="stChatInput"] {{
            background-color: {_inp} !important; border-radius: 12px;
        }}
        .stButton > button, .stDownloadButton > button {{
            border: 1px solid {_accent}; border-radius: 12px; transition: all .15s ease;
            background: {_card}; color: {_t['text']};
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: {_accent}; color: #fff; border-color: {_accent};
            transform: translateY(-1px);
        }}
        a {{ color: {_accent} !important; }}
        {_glass_css}
        </style>""",
        unsafe_allow_html=True,
    )


# ================= LOGIN =================
def clean_user(u):
    return re.sub(r"[^a-z0-9_]", "", (u or "").strip().lower())[:20]


def load_users():
    try:
        with open(USERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(u):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(u, f)


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 120000).hex()


def do_signup(name, pw):
    name = clean_user(name)
    if len(name) < 3:
        return False, "Username kam se kam 3 akshar ka rakho (a-z, 0-9, _)."
    if name in ("teacher", "guest"):
        return False, "Ye username reserved hai."
    if len(pw) < 4:
        return False, "Password kam se kam 4 akshar ka rakho."
    users = load_users()
    if name in users:
        return False, "Ye username pehle se hai."
    salt = os.urandom(16).hex()
    users[name] = {"salt": salt, "hash": hash_pw(pw, salt)}
    save_users(users)
    return True, name


def check_login(name, pw):
    name = clean_user(name)
    if name == "teacher":
        if TEACHER_PASSWORD and hmac.compare_digest(pw.encode(), TEACHER_PASSWORD.encode()):
            return name, "teacher"
        return None, None
    u = load_users().get(name)
    if u and hmac.compare_digest(hash_pw(pw, u["salt"]), u["hash"]):
        return name, "student"
    return None, None


if "user" not in st.session_state:
    st.title("📚 AI Study Companion")
    st.caption("Your personal AI-powered learning platform")
    tab1, tab2, tab3 = st.tabs(["🔑 Login", "📝 Sign up", "👤 Guest"])
    with tab1:
        with st.form("login_form"):
            lu = st.text_input("Username")
            lp = st.text_input("Password", type="password")
            ok = st.form_submit_button("Login")
        if ok:
            nm, rl = check_login(lu, lp)
            if nm:
                st.session_state["user"] = nm
                st.session_state["role"] = rl
                st.rerun()
            else:
                st.error("Username ya password galat hai.")
        st.caption("Teacher ke liye username: teacher")
    with tab2:
        with st.form("signup_form"):
            su = st.text_input("Naya username")
            sp = st.text_input("Naya password", type="password")
            ok2 = st.form_submit_button("Account banao")
        if ok2:
            good, msg = do_signup(su, sp)
            if good:
                st.session_state["user"] = msg
                st.session_state["role"] = "student"
                st.rerun()
            else:
                st.error(msg)
    with tab3:
        st.write("Bina account ke try karo. Guest ka progress sab guests mein shared hota hai.")
        if st.button("Guest ke roop mein chalo"):
            st.session_state["user"] = "guest"
            st.session_state["role"] = "student"
            st.rerun()
    st.stop()

user = st.session_state["user"]
role = st.session_state.get("role", "student")


# ================= MODELS =================
@st.cache_resource
def load():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path="db").get_or_create_collection("study")
    return embedder, col


embedder, col = load()


@st.cache_resource
def get_ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception:
        return None


def ocr_page(page):
    engine = get_ocr()
    if engine is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=150)
        result, _ = engine(pix.tobytes("png"))
        if not result:
            return ""
        return " ".join(r[1] for r in result)
    except Exception:
        return ""


@st.cache_resource
def get_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


# ================= FILE READING =================
def chunk(text, size=800, overlap=100):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def read_pages(path):
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        doc = fitz.open(path)
        pages = []
        for i, p in enumerate(doc, 1):
            text = " ".join(p.get_text().split())
            if len(text) < 20:  # scanned page -> OCR
                text = " ".join(ocr_page(p).split())
            pages.append((i, text))
        return pages
    if ext == "txt":
        with open(path, encoding="utf-8", errors="ignore") as f:
            return [(1, " ".join(f.read().split()))]
    if ext == "docx":
        import docx
        d = docx.Document(path)
        text = " ".join(p.text for p in d.paragraphs)
        return [(1, " ".join(text.split()))]
    return []


def ingest_file(path):
    name = os.path.basename(path)
    count = 0
    for page, text in read_pages(path):
        for j, c in enumerate(chunk(text)):
            if len(c) < 50:
                continue
            col.upsert(
                ids=[f"{name}-{page}-{j}"],
                documents=[c],
                embeddings=[embedder.encode(c).tolist()],
                metadatas=[{"file": name, "page": page}],
            )
            count += 1
    build_index.clear()
    return count


def list_files():
    return sorted(
        f for f in os.listdir(RAW)
        if f.lower().endswith((".pdf", ".txt", ".docx"))
    )


def is_ingested(name):
    return len(col.get(where={"file": name}, limit=1)["ids"]) > 0


# ================= HYBRID SEARCH =================
def tokenize(text):
    return re.findall(r"\w+", text.lower())


@st.cache_resource
def build_index():
    data = col.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    toks = [tokenize(d) for d in docs]
    tfs = [Counter(t) for t in toks]
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    avgdl = sum(len(t) for t in toks) / max(len(toks), 1)
    return {"ids": ids, "docs": docs, "metas": metas, "toks": toks,
            "tfs": tfs, "df": df, "avgdl": avgdl, "n": len(ids)}


def bm25_search(query, selected, k):
    idx = build_index()
    q = set(tokenize(query))
    if not q or idx["n"] == 0:
        return []
    N, df, avgdl = idx["n"], idx["df"], idx["avgdl"] or 1
    k1, b = 1.5, 0.75
    scores = []
    for i, t in enumerate(idx["toks"]):
        if selected != "Sab files" and idx["metas"][i]["file"] != selected:
            continue
        tf, s = idx["tfs"][i], 0.0
        for w in q:
            f = tf.get(w, 0)
            if not f:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
        if s > 0:
            scores.append((s, i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:k]]


def search(query, selected="Sab files", k=6, rerank=False):
    total = col.count()
    if total == 0:
        return []
    kwargs = {"where": {"file": selected}} if selected != "Sab files" else {}
    try:
        res = col.query(
            query_embeddings=[embedder.encode(query).tolist()],
            n_results=min(k * 2, total),
            **kwargs,
        )
        v_ids, v_docs, v_metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
    except Exception:
        v_ids, v_docs, v_metas = [], [], []
    idx = build_index()
    fused, info = {}, {}
    for r, (i, d, m) in enumerate(zip(v_ids, v_docs, v_metas)):
        fused[i] = fused.get(i, 0) + 1 / (60 + r)
        info[i] = (d, m)
    for r, j in enumerate(bm25_search(query, selected, k * 2)):
        i = idx["ids"][j]
        fused[i] = fused.get(i, 0) + 1 / (60 + r)
        info[i] = (idx["docs"][j], idx["metas"][j])
    ranked = sorted(fused, key=lambda x: -fused[x])
    cand = ranked[: (k * 2 if rerank else k)]
    items = [
        {"id": i, "text": info[i][0], "file": info[i][1]["file"], "page": info[i][1]["page"]}
        for i in cand
    ]
    if rerank and items:
        rr = get_reranker()
        if rr is not None:
            try:
                sc = rr.predict([(query, it["text"]) for it in items])
                items = [it for _, it in sorted(zip(sc, items), key=lambda x: -x[0])]
            except Exception:
                pass
        items = items[:k]
    return items


# ================= STARTUP SYNC =================
if "synced" not in st.session_state:
    for fname in list_files():
        if not is_ingested(fname):
            try:
                ingest_file(os.path.join(RAW, fname))
            except Exception:
                pass
    st.session_state["synced"] = True


# ================= LLM CALLS =================
@st.cache_resource(ttl=600)
def all_model_ids():
    if not groq_client:
        return []
    try:
        return [m.id for m in groq_client.models.list().data]
    except Exception:
        return []


def get_models():
    ids = all_model_ids()
    skip = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard")
    chat = [i for i in ids if not any(s in i.lower() for s in skip)]
    order = []
    if GROQ_MODEL and GROQ_MODEL in chat:
        order.append(GROQ_MODEL)
    for pref in ("llama-3.3-70b", "70b", "llama", "gpt-oss", "qwen"):
        for i in chat:
            if pref in i.lower() and i not in order:
                order.append(i)
    for i in chat:
        if i not in order:
            order.append(i)
    return order


def vision_models():
    return [
        i for i in all_model_ids()
        if any(s in i.lower() for s in ("scout", "maverick", "vision")) and "guard" not in i.lower()
    ]


def whisper_models():
    ids = [i for i in all_model_ids() if "whisper" in i.lower()]
    ids.sort(key=lambda x: 0 if "turbo" in x.lower() else 1)
    return ids


def generate(prompt):
    if not groq_client:
        raise Exception("GROQ_API_KEY nahi mili. .env file check karo.")
    models = get_models()
    if not models:
        raise Exception("Groq se koi model nahi mila. API key check karo.")
    errors = []
    for m in models[:4]:
        try:
            r = groq_client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content
        except Exception as e:
            errors.append(f"{m}: {e}")
    raise Exception(" | ".join(errors))


def safe_generate(prompt):
    last = ""
    for wait in (1, 2):
        try:
            return generate(prompt)
        except Exception as e:
            last = str(e)
            time.sleep(wait)
    return f"Error aaya: {last}"


def parse_json(raw):
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def prep_image(raw):
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate_vision(prompt, img_bytes):
    if not groq_client:
        return "GROQ_API_KEY nahi mili."
    models = vision_models()
    if not models:
        return "Is account par abhi koi image (vision) model available nahi hai."
    b64 = base64.b64encode(prep_image(img_bytes)).decode()
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    errors = []
    for m in models[:3]:
        try:
            r = groq_client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": content}],
            )
            return r.choices[0].message.content
        except Exception as e:
            errors.append(f"{m}: {e}")
    return "Error aaya: " + " | ".join(errors)


def transcribe(audio_bytes):
    if not groq_client:
        raise Exception("GROQ_API_KEY nahi mili.")
    models = whisper_models()
    if not models:
        raise Exception("Voice (whisper) model available nahi hai.")
    errors = []
    for m in models[:2]:
        try:
            r = groq_client.audio.transcriptions.create(file=("audio.wav", audio_bytes), model=m)
            return r.text.strip()
        except Exception as e:
            errors.append(f"{m}: {e}")
    raise Exception(" | ".join(errors))


def tts_code(text, lang):
    ranges = [
        ("ऀ", "ॿ", "hi"), ("਀", "੿", "pa"),
        ("઀", "૿", "gu"), ("ঀ", "৿", "bn"),
        ("஀", "௿", "ta"), ("ఀ", "౿", "te"),
    ]
    for ch in text:
        for a, b, c in ranges:
            if a <= ch <= b:
                return "mr" if (c == "hi" and lang.startswith("Marathi")) else c
    return "en"


def speak(text, lang):
    from gtts import gTTS
    clean = re.sub(r"[*#`_>|$]", "", text.split("**Source:**")[0])[:1500]
    buf = io.BytesIO()
    gTTS(clean, lang=tts_code(clean, lang)).write_to_fp(buf)
    return buf.getvalue()


def play_audio(data, autoplay=False):
    try:
        st.audio(data, format="audio/mp3", autoplay=autoplay)
    except TypeError:
        st.audio(data, format="audio/mp3")


# ================= LANGUAGE / LEVEL / STYLE =================
LANGS = [
    "Same as question", "English", "Hindi (हिंदी)", "Hinglish",
    "Punjabi (ਪੰਜਾਬੀ)", "Marathi (मराठी)", "Gujarati (ગુજરાતી)",
    "Bengali (বাংলা)", "Tamil (தமிழ்)", "Telugu (తెలుగు)",
]


def lang_rule(lang):
    if lang == "Same as question":
        return "Reply in the same language as the question."
    if lang == "Hinglish":
        return (
            "Write the whole answer in easy Hinglish (Hindi written in English letters). "
            "Keep technical terms and formulas in English."
        )
    name = lang.split(" (")[0]
    return (
        f"Write the whole answer in {name}. "
        "Keep technical terms and formulas in English (you may add them in brackets). "
        "Even the sentence 'I couldn't find this in the study material' must be in this language."
    )


def level_rule():
    lv = st.session_state.get("level_sel", "Intermediate")
    aud = (st.session_state.get("aud") or "").strip()
    base = {
        "Beginner": "Audience level: BEGINNER. Use very simple words, everyday analogies and small examples. Avoid jargon.",
        "Intermediate": "Audience level: INTERMEDIATE. Balanced explanation, define key terms briefly.",
        "Advanced": "Audience level: ADVANCED. Be precise and technical; include formulas, assumptions and nuances.",
    }[lv]
    if aud:
        base += f" Target audience: {aud}."
    return base


def ls(lang):
    return lang_rule(lang) + " " + level_rule()


MODES = {
    "Simple": "Explain in very simple words, like to a beginner.",
    "Detailed": "Give a detailed, well-structured explanation with headings or bullet points.",
    "Exam-style": "Answer like a good exam answer: definition, key points, formula if any, short conclusion.",
    "😂 Funny (Bollywood style)": "Explain correctly, but in a funny way: use Bollywood dialogues, desi analogies (chai, cricket, traffic, mummy ka gussa) and light jokes. The facts must stay accurate. Keep it clean and friendly.",
    "🔥 Roast mode": "Explain correctly, but with a playful, friendly roast of how everyone confuses this topic. Be witty, never insulting. The facts must stay accurate.",
}


# ================= PROGRESS + GAMIFICATION =================
DEFAULT_PROGRESS = {
    "weak": {}, "history": [], "xp": 0, "streak": 0, "best_streak": 0,
    "last_day": "", "badges": [], "topics": {}, "stats": {}, "activity": {},
    "mistakes": [], "srs": {}, "syllabus_done": [], "my_syllabus": [],
    "exam_date": "", "buddy": False,
}


def progress_path(u):
    return os.path.join(USERS_DIR, u, "progress.json")


def load_progress(u):
    try:
        with open(progress_path(u), encoding="utf-8") as f:
            p = json.load(f)
    except Exception:
        p = {}
    for k, v in DEFAULT_PROGRESS.items():
        p.setdefault(k, json.loads(json.dumps(v)))
    return p


def save_progress(u, p):
    os.makedirs(os.path.join(USERS_DIR, u), exist_ok=True)
    with open(progress_path(u), "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


def level_of(xp):
    return xp // 100 + 1


BADGES = [
    ("🌱 First Step", "Pehla sawal ya quiz", lambda p: p["stats"].get("questions", 0) + len(p["history"]) >= 1),
    ("🧠 Quiz Starter", "Pehla quiz/test diya", lambda p: len(p["history"]) >= 1),
    ("🎯 Sharpshooter", "Kisi test mein 80%+", lambda p: any(h.get("pct", 0) >= 80 for h in p["history"])),
    ("💯 Perfectionist", "Kisi test mein 100%", lambda p: any(h.get("pct", 0) >= 100 for h in p["history"])),
    ("📚 Curious Mind", "25 sawal poochhe", lambda p: p["stats"].get("questions", 0) >= 25),
    ("🃏 Card Master", "20 flashcards yaad kiye", lambda p: p["stats"].get("cards", 0) >= 20),
    ("✍️ Writer", "5 answers evaluate karwaye", lambda p: p["stats"].get("evals", 0) >= 5),
    ("🔥 3-Day Streak", "3 din lagatar padhai", lambda p: p["best_streak"] >= 3),
    ("🔥 7-Day Streak", "7 din lagatar padhai", lambda p: p["best_streak"] >= 7),
    ("🏆 Level 5", "Level 5 tak pahunche", lambda p: level_of(p["xp"]) >= 5),
    ("🧪 Data Lab Explorer", "Data Lab mein regression chalaya", lambda p: p["stats"].get("labs", 0) >= 1),
    ("🧐 Pattern Hunter", "Exam pattern analyse kiya", lambda p: p["stats"].get("patterns", 0) >= 1),
    ("📚 Syllabus Star", "5 syllabus topics complete", lambda p: p["stats"].get("syllabus", 0) >= 5),
]


def _bump(p, amount, stat=None, n=1):
    today = date.today().isoformat()
    if p["last_day"] != today:
        y = (date.today() - timedelta(days=1)).isoformat()
        p["streak"] = p["streak"] + 1 if p["last_day"] == y else 1
        p["last_day"] = today
        p["best_streak"] = max(p["best_streak"], p["streak"])
    p["xp"] += int(amount)
    if stat:
        p["stats"][stat] = p["stats"].get(stat, 0) + n
    p["activity"][today] = p["activity"].get(today, 0) + 1
    new = []
    for name, _, fn in BADGES:
        if name not in p["badges"]:
            try:
                ok = fn(p)
            except Exception:
                ok = False
            if ok:
                p["badges"].append(name)
                new.append(name)
    return new


def award(amount, stat=None, n=1):
    p = load_progress(user)
    new = _bump(p, amount, stat, n)
    save_progress(user, p)
    for b in new:
        st.toast(f"🏅 Naya badge: {b}")


def record_result(u, label, total, points, wrong_topics, right_topics, wrong_items=(), right_qs=()):
    p = load_progress(u)
    weak = p["weak"]
    for t in wrong_topics:
        k = t.strip().lower()
        if k:
            weak[k] = weak.get(k, 0) + 1
            r = p["topics"].setdefault(k, [0, 0])
            r[1] += 1
            sr = p["srs"].setdefault(k, {"last": "", "stage": 0})
            sr["stage"] = 0
            sr["last"] = date.today().isoformat()
    for t in right_topics:
        k = t.strip().lower()
        if not k:
            continue
        r = p["topics"].setdefault(k, [0, 0])
        r[0] += 1
        r[1] += 1
        sr = p["srs"].setdefault(k, {"last": "", "stage": 0})
        sr["stage"] = sr.get("stage", 0) + 1
        sr["last"] = date.today().isoformat()
        if k in weak:
            weak[k] -= 1
            if weak[k] <= 0:
                del weak[k]
    for item, yours in wrong_items:
        qkey = item["q"]
        m = next((x for x in p["mistakes"] if x["q"] == qkey), None)
        if m is None:
            m = {"q": qkey, "type": item.get("type", "mcq"), "options": item.get("options"),
                 "answer": item["answer"], "topic": item.get("topic", "general"),
                 "explanation": item.get("explanation", ""), "count": 0}
            p["mistakes"].append(m)
        m["count"] += 1
        m["your"] = yours
        m["last"] = date.today().isoformat()
    if right_qs:
        solved = set(right_qs)
        p["mistakes"] = [x for x in p["mistakes"] if x["q"] not in solved]
    p["mistakes"] = p["mistakes"][-200:]
    pct = round(100 * points / total, 1) if total else 0
    p["history"].append(
        {"time": time.strftime("%d-%m-%Y %H:%M"), "test": label,
         "score": f"{points:g}/{total}", "pct": pct}
    )
    p["history"] = p["history"][-50:]
    xp = max(int(points * 10), 0) + (20 if pct >= 80 else 0)
    new = _bump(p, xp, "quizzes")
    save_progress(u, p)
    for b in new:
        st.toast(f"🏅 Naya badge: {b}")
    return xp


# ================= LOGS (teacher + doubt detection) =================
def log_event(ev):
    ev["time"] = time.strftime("%d-%m-%Y %H:%M")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_logs():
    out = []
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


# ================= MATERIAL HELPERS =================
def get_file_chunks(name):
    data = col.get(where={"file": name})
    items = list(zip(data["metadatas"], data["documents"]))
    items.sort(key=lambda x: x[0]["page"])
    return [d for _, d in items]


def material_from_files(names, limit=8000):
    chunks = []
    for n in names:
        chunks += get_file_chunks(n)
    random.shuffle(chunks)
    out, total = [], 0
    for c in chunks:
        if total > limit:
            break
        out.append(c)
        total += len(c)
    return "\n".join(out)


def material_from_topics(topics, limit=8000):
    out, total = [], 0
    for t in topics:
        for it in search(t, "Sab files", 3):
            if total > limit:
                break
            out.append(it["text"])
            total += len(it["text"])
    return "\n".join(out)


def ordered_text(names, limit=12000):
    chunks = []
    for n in names:
        chunks += get_file_chunks(n)
    text = " ".join(chunks)
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...\n" + text[-limit // 2:]


def ctx_text(items):
    return "\n\n".join(f"[{it['file']} p{it['page']}] {it['text']}" for it in items)


# ================= AI FUNCTIONS =================
def ask(question, selected, mode, lang, history, rerank):
    v = find_verified(question)
    if v:
        return (v["a"] + f"\n\n✅ **Teacher-verified answer** (by {v.get('by', 'teacher')})"), [], True
    locked = st.session_state.get("locked", True)
    if col.count() == 0 and locked:
        return "Abhi koi study material upload nahi hua hai.", [], False
    eq = question
    if len(question.split()) < 6 and history:  # short follow-up
        prev = [h["q"] for h in history if h["role"] == "assistant" and h.get("q")]
        if prev:
            eq = prev[-1] + " " + question
    items = search(eq, selected, 6, rerank) if col.count() else []
    if not items and locked:
        return "Ye study material mein nahi mila.", [], False
    hist = "\n".join(
        f"{h['role']}: {h['msg'].split('**Source:**')[0].strip()}" for h in history[-6:]
    )
    weak = list(load_progress(user)["weak"].keys())[:5]
    memory = (f"The student's known weak topics: {', '.join(weak)}. "
              "If the question touches one of them, add one short extra tip (1-2 lines)." if weak else "")
    if locked:
        rule = ("Answer ONLY using the context below. "
                "If the answer is not in the context, say \"I couldn't find this in the study material.\"")
    else:
        rule = ("Use the context below first. If it is not enough, you may add general knowledge, "
                "but mark that part clearly as \"🌐 General knowledge (not in your material)\".")
    context = "\n".join(it["text"] for it in items) or "(none)"
    prompt = f"""You are a study assistant. {rule}
{MODES[mode]}
{ls(lang)}
{memory}

Earlier conversation (for follow-up questions):
{hist}

Context:
{context}

Question: {question}"""
    answer = safe_generate(prompt)
    found = ("couldn't find" not in answer.lower()) and not answer.startswith("Error aaya")
    return answer, items, found


def ask_image(question, img_bytes, selected, mode, lang, rerank):
    items = search(question, selected, 4, rerank) if (question and col.count()) else []
    context = "\n\n".join(it["text"] for it in items)
    prompt = f"""You are a study assistant. Look at the image carefully and answer the student's question.
Use the study material context below too, if it is relevant.
{MODES[mode]}
{ls(lang)}

Study material context:
{context or '(none)'}

Question: {question or 'Explain what is shown in this image.'}"""
    return generate_vision(prompt, img_bytes), items


def summarize(name, style, lang):
    chunks = get_file_chunks(name)
    if not chunks:
        return "Is file mein text nahi mila."
    text = " ".join(chunks)
    ask_style = {
        "Short summary": "Write a short summary in 8-10 lines.",
        "Key points": "List the key points as clear bullet points.",
        "Important terms": "List important terms and formulas with one-line meanings.",
    }[style]
    ask_style = f"{ask_style} {ls(lang)}"
    parts = [text[i:i + 9000] for i in range(0, len(text), 9000)][:6]
    if len(parts) == 1:
        return safe_generate(f"{ask_style}\n\nStudy material:\n{parts[0]}")
    partial = [safe_generate(f"Summarize briefly in 6 lines in English:\n\n{p}") for p in parts]
    return safe_generate(f"{ask_style}\n\nNotes:\n" + "\n\n".join(partial))


# ---------- Quiz generation (MCQ / True-False / Fill / Descriptive) ----------
QTYPES = {
    "MCQ": "mcq",
    "True / False": "tf",
    "Fill in the blank": "fill",
    "Descriptive": "desc",
}
QTAG = {"mcq": "MCQ", "tf": "True/False", "fill": "Fill in the blank", "desc": "Descriptive"}
QSPEC = {
    "mcq": '{"type":"mcq","q":"question","options":["A","B","C","D"],"answer":0,"topic":"topic","explanation":"short reason"}',
    "tf": '{"type":"tf","q":"a statement","options":["True","False"],"answer":0,"topic":"topic","explanation":"short reason"}',
    "fill": '{"type":"fill","q":"sentence with ____ in place of the missing word","answer":"missing word or phrase","topic":"topic","explanation":"short reason"}',
    "desc": '{"type":"desc","q":"open question","answer":"model answer in 2-4 sentences","topic":"topic","explanation":"key points expected"}',
}


def valid_q(q):
    try:
        t = q.get("type")
        if not q.get("q"):
            return False
        if t == "mcq":
            return len(q["options"]) == 4 and 0 <= int(q["answer"]) <= 3
        if t == "tf":
            return len(q["options"]) == 2 and int(q["answer"]) in (0, 1)
        if t in ("fill", "desc"):
            return bool(str(q.get("answer", "")).strip())
    except Exception:
        return False
    return False


def make_quiz(material, n, level, lang, types=("mcq",), topics=None):
    if not material.strip():
        return None, "Material nahi mila. Pehle file upload karo."
    types = list(types) or ["mcq"]
    if topics:
        topic_rule = (
            "Cover these topics: " + ", ".join(topics) + ". "
            'In the "topic" field use exactly one of these topic names.'
        )
    else:
        topic_rule = 'In the "topic" field give a short 2-4 word topic name (in English).'
    formats = "\n".join(f"- {QSPEC[t]}" for t in types)
    prompt = f"""Create {n} questions ({level} level) ONLY from the study material below.
Use these question types, spread roughly evenly: {', '.join(types)}.
{ls(lang)} This applies to question text, options, answers and explanations.
{topic_rule}
Return ONLY valid JSON, no extra text, in the form {{"questions":[ ... ]}}. Keys stay in English.
Each question must follow exactly one of these formats:
{formats}
"answer" for mcq and tf is the index of the correct option.

Study material:
{material}"""
    raw = safe_generate(prompt)
    try:
        data = parse_json(raw)
        qs = [q for q in data["questions"] if valid_q(q)]
        if not qs:
            return None, "Quiz ban nahi paya. Dobara try karo."
        return qs, ""
    except Exception:
        return None, f"Quiz format galat aaya, dobara try karo.\n\n{raw[:300]}"


# ---------- Answer evaluation ----------
def evaluate_answer(question, student, context, max_marks, lang):
    prompt = f"""You are a strict but fair examiner. Evaluate the student's answer using ONLY the reference material.
Maximum marks: {max_marks}. Give marks as a number between 0 and {max_marks} (decimals allowed).
{ls(lang)} (This applies to the text values only.)
Return ONLY valid JSON with English keys:
{{"score": 0, "concept_understanding": "1-2 lines", "accuracy": "1-2 lines",
"missing_points": ["..."], "suggestions": ["..."], "ideal_answer": "short model answer"}}

Question: {question}

Student's answer: {student}

Reference material:
{context}"""
    raw = safe_generate(prompt)
    try:
        d = parse_json(raw)
        d["score"] = max(0.0, min(float(d.get("score", 0)), float(max_marks)))
        return d
    except Exception:
        return {"score": 0.0, "concept_understanding": raw[:300], "accuracy": "",
                "missing_points": [], "suggestions": [], "ideal_answer": ""}


def show_evaluation(ev, max_marks):
    st.subheader(f"Marks: {ev['score']:g} / {max_marks}")
    st.progress(min(ev["score"] / max_marks, 1.0))
    st.markdown(f"**🧠 Concept understanding:** {ev.get('concept_understanding', '')}")
    st.markdown(f"**🎯 Accuracy:** {ev.get('accuracy', '')}")
    if ev.get("missing_points"):
        st.markdown("**❗ Missing points**")
        for m in ev["missing_points"]:
            st.write(f"- {m}")
    if ev.get("suggestions"):
        st.markdown("**💡 Suggestions**")
        for s in ev["suggestions"]:
            st.write(f"- {s}")
    if ev.get("ideal_answer"):
        with st.expander("✅ Ideal answer dekho"):
            st.write(ev["ideal_answer"])


# ---------- Flashcards ----------
def make_flashcards(material, n, lang):
    if not material.strip():
        return None, "Material nahi mila. Pehle file upload karo."
    prompt = f"""Create {n} flashcards ONLY from the study material below.
Each card has a short question or term on the front and a short, clear answer on the back.
{ls(lang)}
Return ONLY valid JSON, no extra text. Keys must stay in English exactly as shown:
{{"cards":[{{"front":"question or term","back":"short answer"}}]}}

Study material:
{material}"""
    raw = safe_generate(prompt)
    try:
        data = parse_json(raw)
        cards = [
            {"front": str(c["front"]), "back": str(c["back"])}
            for c in data["cards"] if c.get("front") and c.get("back")
        ]
        if not cards:
            return None, "Cards ban nahi paye. Dobara try karo."
        return cards, ""
    except Exception:
        return None, f"Cards ka format galat aaya, dobara try karo.\n\n{raw[:300]}"


# ---------- Mind map ----------
def make_mindmap(material, topic, lang):
    if not material.strip():
        return None, "Material nahi mila."
    prompt = f"""Build a concept map (mind map) ONLY from the study material below{f' focusing on: {topic}' if topic else ''}.
Use 4-7 main branches, each with 2-5 short children (max 6 words each).
Add up to 6 cross-links between related concepts, each with a short label.
{lang_rule(lang)} Keep node names short.
Return ONLY valid JSON, English keys:
{{"center":"main topic","branches":[{{"name":"branch","children":["child 1","child 2"]}}],
"links":[{{"from":"node name","to":"node name","label":"relation"}}]}}

Study material:
{material}"""
    raw = safe_generate(prompt)
    try:
        m = parse_json(raw)
        if not m.get("center") or not m.get("branches"):
            return None, "Mind map ban nahi paya. Dobara try karo."
        return m, ""
    except Exception:
        return None, f"Mind map ka format galat aaya, dobara try karo.\n\n{raw[:300]}"


def _esc(s):
    return str(s).replace("\\", " ").replace('"', "'").replace("\n", " ")[:70]


def mindmap_dot(m):
    ids, lines = {}, []
    lines.append("digraph G {")
    lines.append('rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.7;')
    lines.append('node [shape=box, style="rounded,filled", fillcolor="#E6EAFF", color="#6C63FF", fontname="Helvetica", fontcolor="#1F2340"];')
    lines.append('edge [color="#6C63FF"];')

    def nid(label):
        key = str(label).strip().lower()
        if key not in ids:
            ids[key] = f"n{len(ids)}"
        return ids[key], _esc(label)

    cid, clab = nid(m["center"])
    lines.append(f'{cid} [label="{clab}", shape=ellipse, fillcolor="#6C63FF", fontcolor="white"];')
    for b in m.get("branches", []):
        bid, blab = nid(b.get("name", "?"))
        lines.append(f'{bid} [label="{blab}", fillcolor="#C7D2FE"];')
        lines.append(f"{cid} -> {bid};")
        for ch in b.get("children", []):
            xid, xlab = nid(ch)
            lines.append(f'{xid} [label="{xlab}", fillcolor="#FFFFFF"];')
            lines.append(f"{bid} -> {xid};")
    for l in m.get("links", []):
        a = ids.get(str(l.get("from", "")).strip().lower())
        b = ids.get(str(l.get("to", "")).strip().lower())
        if a and b and a != b:
            lines.append(f'{a} -> {b} [style=dashed, color="#F97316", label="{_esc(l.get("label", ""))}", fontsize=10];')
    lines.append("}")
    return "\n".join(lines)


def mindmap_outline(m):
    out = [f"**{m['center']}**"]
    for b in m.get("branches", []):
        out.append(f"- **{b.get('name', '')}**")
        for ch in b.get("children", []):
            out.append(f"    - {ch}")
    return "\n".join(out)


# ---------- Compare / formulas / planner / research / case study ----------
def compare_topics(a, b, selected, lang, rerank):
    ia = search(a, selected, 4, rerank)
    ib = search(b, selected, 4, rerank)
    if not ia and not ib:
        return "Study material mein ye topics nahi mile."
    prompt = f"""Compare "{a}" and "{b}" using ONLY the study material below.
If something is missing in the material, write "not in material" for it instead of guessing.
Output:
1. A Markdown table with rows: Definition, Key idea, Cause, Detection / test, Effect or problem, Solution / remedy (skip rows that are not relevant).
2. "Main differences" as 3 short bullet points.
3. "Exam tip" in one line.
{ls(lang)}

Material about {a}:
{ctx_text(ia)}

Material about {b}:
{ctx_text(ib)}"""
    return safe_generate(prompt)


def make_formula_sheet(names, lang):
    chunks = []
    for n in names:
        chunks += get_file_chunks(n)
    if not chunks:
        return "Chuni hui files mein text nahi mila."
    text = " ".join(chunks)
    parts = [text[i:i + 9000] for i in range(0, len(text), 9000)][:4]
    rules = (
        "List every formula, equation and rule found. For each give: **Name**, the formula in LaTeX "
        "between $ signs, what each symbol means, and when it is used. Use ONLY the material given."
    )
    res = [safe_generate(f"{rules}\n\nMaterial:\n{p}") for p in parts]
    if len(res) == 1:
        return safe_generate(f"Clean this formula list. {ls(lang)}\n\n{res[0]}")
    return safe_generate(
        "Merge these formula lists into one clean sheet, remove duplicates, "
        f"group by topic, keep LaTeX in $ signs. {ls(lang)}\n\n" + "\n\n".join(res)
    )


def make_plan(files, exam_date, hours, lang, subjects_note=""):
    days = max((exam_date - date.today()).days, 1)
    overview = []
    for f in files:
        chunks = get_file_chunks(f)
        if not chunks:
            continue
        step = max(1, len(chunks) // 4)
        sample = " ".join(chunks[::step][:4])[:2000]
        overview.append(f"FILE: {f}\nSAMPLE CONTENT: {sample}")
    if not overview and not subjects_note:
        return "Chuni hui files mein text nahi mila."
    weak = list(load_progress(user)["weak"].keys())[:8]
    weak_txt = ", ".join(weak) if weak else "none yet"
    prompt = f"""Make a realistic day-by-day revision timetable for a student.
Today: {date.today().strftime('%d-%m-%Y')}. Exam date: {exam_date.strftime('%d-%m-%Y')} ({days} days left).
Study time per day: {hours} hours.
Extra subjects / notes from the student: {subjects_note or 'none'}.
Weak topics to revise more often: {weak_txt}.
Rules:
- Cover all files and subjects below, work out the main topics yourself from the sample content.
- Put harder or weak topics earlier and repeat them, keep the last 1-2 days for full revision and a mock test.
- If there are more than 21 days, group the plan week by week instead of day by day.
- Output a Markdown table with columns: Day/Date | Topics | What to do | Hours.
- After the table add 3 short study tips.
{lang_rule(lang)}

{chr(10).join(overview)}"""
    return safe_generate(prompt)


def research_assistant(name, lang):
    text = ordered_text([name], 14000)
    if not text.strip():
        return "Is file mein text nahi mila."
    prompt = f"""You are a research paper assistant. Using ONLY the paper text below, produce a structured analysis.
If a section is not stated in the paper, write "not stated in the paper".
Sections (Markdown headings):
## 1. Title and objective
## 2. Research question / hypothesis
## 3. Data and sample
## 4. Methodology (model, variables, tests)
## 5. Key findings
## 6. Limitations
## 7. Future research directions
## 8. One-paragraph plain-language summary
## 9. Critical review (2-3 strengths, 2-3 weaknesses)
{ls(lang)}

Paper text:
{text}"""
    return safe_generate(prompt)


def make_case(material, topic, lang):
    if not material.strip():
        return None, "Material nahi mila."
    prompt = f"""Create a realistic practical business case study{f' about: {topic}' if topic else ''} that tests the concepts in the study material below.
The case should be 150-220 words with realistic (made-up) numbers. Then write 3 analytical questions (application based, not memory based).
{ls(lang)}
Return ONLY valid JSON, English keys:
{{"title":"...","case":"...","questions":["q1","q2","q3"]}}

Study material:
{material}"""
    raw = safe_generate(prompt)
    try:
        d = parse_json(raw)
        if not d.get("case") or not d.get("questions"):
            return None, "Case study ban nahi paya. Dobara try karo."
        return d, ""
    except Exception:
        return None, f"Case study ka format galat aaya, dobara try karo.\n\n{raw[:300]}"


def debate_turn(d, lang):
    transcript = "\n".join(
        f"{'AI' if m['role'] == 'assistant' else 'Student'}: {m['content']}" for m in d["msgs"]
    )
    prompt = f"""You are a sharp but friendly debate coach testing a student's conceptual understanding.
Topic: {d['topic']}
Student's stance: {d['stance'] or 'not given'}
Use ONLY this study material for facts:
{d['ctx']}
{ls(lang)}

Conversation so far:
{transcript or '(not started)'}

Task: {"Start the debate by asking ONE challenging counter-question about the topic." if not d['msgs'] else "In 1-2 lines say specifically what was right or wrong in the student's last reply, then ask ONE tougher counter-question."}
Keep it under 90 words."""
    return safe_generate(prompt)


def debate_score(d, lang):
    transcript = "\n".join(
        f"{'AI' if m['role'] == 'assistant' else 'Student'}: {m['content']}" for m in d["msgs"]
    )
    prompt = f"""Judge the student's performance in this debate on "{d['topic']}" using ONLY this study material:
{d['ctx']}

Debate:
{transcript}

{ls(lang)} (text values only)
Return ONLY valid JSON, English keys:
{{"score": 0, "strengths": ["..."], "gaps": ["..."], "advice": "2-3 lines"}}
"score" is out of 10."""
    raw = safe_generate(prompt)
    try:
        r = parse_json(raw)
        r["score"] = max(0.0, min(float(r.get("score", 0)), 10.0))
        return r
    except Exception:
        return {"score": 0.0, "strengths": [], "gaps": [], "advice": raw[:300]}


# ================= UNIQUE FEATURES: STORAGE + HELPERS =================
def jload(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- Teacher-verified answers + doubt queue ----------
def norm_q(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (q or "").lower())).strip()


def find_verified(q):
    nq = norm_q(q)
    if not nq:
        return None
    best, bs = None, 0.0
    for v in jload(VERIFIED_FILE, []):
        r = difflib.SequenceMatcher(None, nq, norm_q(v.get("q", ""))).ratio()
        if r > bs:
            best, bs = v, r
    return best if bs >= 0.88 else None


def save_verified(q, a, by):
    items = [v for v in jload(VERIFIED_FILE, []) if norm_q(v.get("q", "")) != norm_q(q)]
    items.append({"q": q, "a": a, "by": by, "time": time.strftime("%d-%m-%Y %H:%M")})
    jsave(VERIFIED_FILE, items)


def add_doubt(u, q, bot_answer):
    items = jload(DOUBTS_FILE, [])
    items.append({"id": uuid.uuid4().hex[:10], "user": u, "q": q,
                  "bot": bot_answer.split("**Source:**")[0].strip()[:600],
                  "status": "open", "answer": "", "by": "",
                  "time": time.strftime("%d-%m-%Y %H:%M")})
    jsave(DOUBTS_FILE, items)
    log_event({"type": "doubt", "user": u, "q": q})


# ---------- Syllabus ----------
def get_syllabus(u):
    shared = jload(SYLLABUS_FILE, {}).get("topics", [])
    if shared:
        return shared, "teacher"
    return load_progress(u).get("my_syllabus", []), "personal"


def topic_match(topics, name):
    n = name.strip().lower()
    r = tot = 0
    for k, (rr, tt) in topics.items():
        if not tt:
            continue
        if k == n or k in n or n in k or difflib.SequenceMatcher(None, k, n).ratio() >= 0.65:
            r += rr
            tot += tt
    return r, tot


def coverage(prog, syl):
    if not syl:
        return 0.0
    done = set(prog["syllabus_done"])
    n = 0
    for t in syl:
        r, tot = topic_match(prog["topics"], t)
        if t in done or (tot >= 2 and r / tot >= 0.6):
            n += 1
    return n / len(syl)


def extract_syllabus(text):
    raw = safe_generate(
        "List the main syllabus topics covered in the study material below as short names "
        "(max 25, no numbering).\n"
        'Return ONLY valid JSON: {"topics":["topic 1","topic 2"]}\n\nMaterial:\n' + text
    )
    try:
        return [str(t).strip() for t in parse_json(raw)["topics"] if str(t).strip()]
    except Exception:
        return []


# ---------- Forgetting curve ----------
INTERVALS = [1, 3, 7, 14, 30]


def due_topics(p):
    out = []
    for t, s in p.get("srs", {}).items():
        try:
            last = date.fromisoformat(s["last"])
        except Exception:
            continue
        days = (date.today() - last).days
        need = INTERVALS[min(s.get("stage", 0), len(INTERVALS) - 1)]
        if days >= need:
            out.append((t, days, s.get("stage", 0)))
    out.sort(key=lambda x: -x[1])
    return out


# ---------- Study buddy ----------
def topic_acc(pr, topic):
    r = tot = 0
    for k, (rr, tt) in pr["topics"].items():
        if k == topic or difflib.SequenceMatcher(None, k, topic).ratio() >= 0.75:
            r += rr
            tot += tt
    return r, tot


# ---------- Exam pattern ----------
def analyse_pattern(text):
    prompt = f"""You are analysing past exam papers. Using ONLY the papers' text below, return ONLY valid JSON with English keys:
{{"sections":[{{"name":"Section A","type":"MCQ / short / long / numerical / case","count":10,"marks_each":1}}],
"total_marks":100,"duration":"3 hours",
"hot_topics":[{{"topic":"name","times":3}}],
"style_notes":"how questions are worded, difficulty, internal choice, typical command words"}}
If something is unknown, give a best estimate and say so in style_notes.

Papers:
{text}"""
    raw = safe_generate(prompt)
    try:
        d = parse_json(raw)
        if not d.get("sections"):
            return None, "Pattern samajh nahi aaya. Papers ka text check karo."
        return d, ""
    except Exception:
        return None, f"Pattern ka format galat aaya, dobara try karo.\n\n{raw[:300]}"


def make_paper(pattern, material, hot, lang):
    boost = "Give extra weight to the hot topics listed in the pattern." if hot else ""
    prompt = f"""Create a new mock exam paper following EXACTLY this pattern:
{json.dumps(pattern, ensure_ascii=False)}
Use ONLY the study material below for content. {boost}
Output Markdown: title, time, total marks, instructions, then each section with numbered questions and the marks in brackets.
After a line with --- write "## Marking scheme / Answer key" with short model answers and the marks split.
{ls(lang)}

Study material:
{material}"""
    return safe_generate(prompt)


# ---------- Data Lab (live regression tests) ----------
def sample_hetero_df():
    rng = np.random.default_rng(42)
    n = 120
    income = rng.uniform(10, 100, n)
    family = rng.integers(1, 7, n).astype(float)
    noise = rng.normal(0, 1, n) * (0.05 * income + 0.5) * 3
    spend = 5 + 0.6 * income + 1.5 * family + noise
    return pd.DataFrame({"income": income.round(1), "family_size": family, "spending": spend.round(1)})


def _aux_r2(Z, target):
    coef = np.linalg.pinv(Z.T @ Z) @ Z.T @ target
    res = target - Z @ coef
    tss = ((target - target.mean()) ** 2).sum()
    return float(1 - (res @ res) / tss) if tss > 0 else 0.0


def run_regression(df, yc, xs):
    from scipy import stats as sps
    d = df[[yc] + list(xs)].dropna()
    n, k = len(d), len(xs) + 1
    if n < k + 5:
        return None, "Data bahut kam hai (kam se kam variables + 5 rows chahiye)."
    y = d[yc].to_numpy(float)
    Xv = d[list(xs)].to_numpy(float)
    X = np.column_stack([np.ones(n), Xv])
    names = ["const"] + list(xs)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    fitted = X @ beta
    e = y - fitted
    dfree = n - k
    sse = float(e @ e)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 0.0
    adj = 1 - (1 - r2) * (n - 1) / dfree
    s2 = sse / dfree
    se = np.sqrt(np.diag(XtX_inv) * s2)
    tval = beta / se
    pval = 2 * sps.t.sf(np.abs(tval), dfree)
    meat = X.T @ (X * (e ** 2)[:, None])
    V = XtX_inv @ meat @ XtX_inv * n / dfree
    se_r = np.sqrt(np.diag(V))
    p_r = 2 * sps.t.sf(np.abs(beta / se_r), dfree)
    fstat = ((sst - sse) / (k - 1)) / s2 if s2 > 0 else float("nan")
    f_p = float(sps.f.sf(fstat, k - 1, dfree))

    e2 = e ** 2
    bp_lm = n * _aux_r2(X, e2)
    bp_p = float(sps.chi2.sf(bp_lm, k - 1))

    cols = [Xv[:, i] for i in range(Xv.shape[1])]
    sq = [Xv[:, i] ** 2 for i in range(Xv.shape[1])]
    cross = [Xv[:, i] * Xv[:, j] for i in range(Xv.shape[1]) for j in range(i + 1, Xv.shape[1])]
    Zw = np.column_stack([np.ones(n)] + cols + sq + cross)
    rk = int(np.linalg.matrix_rank(Zw))
    white = None
    if rk - 1 >= 1 and n > rk + 2:
        w_lm = n * _aux_r2(Zw, e2)
        white = (float(w_lm), float(sps.chi2.sf(w_lm, rk - 1)))

    vif = {}
    for i, nm in enumerate(xs):
        if len(xs) == 1:
            vif[nm] = 1.0
            continue
        others = np.column_stack([np.ones(n), np.delete(Xv, i, axis=1)])
        r2j = _aux_r2(others, Xv[:, i])
        vif[nm] = float(1 / (1 - r2j)) if r2j < 0.999999 else 999.0

    dw = float(np.sum(np.diff(e) ** 2) / sse) if sse > 0 else float("nan")
    jb, jb_p = sps.jarque_bera(e)

    coef = pd.DataFrame({
        "Variable": names, "Coef": beta, "Std Err": se, "t": tval, "p-value": pval,
        "Robust SE (HC1)": se_r, "Robust p": p_r,
    })
    return {
        "n": n, "k": k, "r2": float(r2), "adj": float(adj), "f": float(fstat), "f_p": f_p,
        "coef": coef, "bp": (float(bp_lm), bp_p), "white": white, "vif": vif, "dw": dw,
        "jb": (float(jb), float(jb_p)), "y": yc, "xs": list(xs),
        "resid": pd.DataFrame({"Fitted": fitted, "Residual": e}),
    }, ""


REMEDY = {
    "hetero": "Robust (HC1/White) standard errors use karo, WLS try karo, ya Y ka log lo.",
    "multi": "Zyada correlated variable hatao ya milao, ridge regression, ya zyada data lo.",
    "auto": "Newey-West SE ya GLS use karo, lagged variable jodo.",
    "normal": "Bade sample mein zyada chinta nahi; outliers check karo ya variables transform karo.",
}


def assumption_table(res):
    rows = []
    bp_p = res["bp"][1]
    rows.append({"Assumption": "Homoscedasticity", "Test": "Breusch-Pagan",
                 "Value": f"LM={res['bp'][0]:.2f}, p={bp_p:.4f}",
                 "Verdict": "✅ OK" if bp_p >= 0.05 else "❌ Violated (heteroscedasticity)",
                 "Remedy": "-" if bp_p >= 0.05 else REMEDY["hetero"]})
    if res["white"]:
        wp = res["white"][1]
        rows.append({"Assumption": "Homoscedasticity", "Test": "White",
                     "Value": f"LM={res['white'][0]:.2f}, p={wp:.4f}",
                     "Verdict": "✅ OK" if wp >= 0.05 else "❌ Violated (heteroscedasticity)",
                     "Remedy": "-" if wp >= 0.05 else REMEDY["hetero"]})
    worst = max(res["vif"].values()) if res["vif"] else 1.0
    rows.append({"Assumption": "No multicollinearity", "Test": "VIF (max)",
                 "Value": f"{worst:.2f}",
                 "Verdict": "✅ OK" if worst < 5 else ("⚠️ Moderate" if worst < 10 else "❌ Severe"),
                 "Remedy": "-" if worst < 5 else REMEDY["multi"]})
    dw = res["dw"]
    ok = 1.5 <= dw <= 2.5
    rows.append({"Assumption": "No autocorrelation", "Test": "Durbin-Watson",
                 "Value": f"{dw:.2f}",
                 "Verdict": "✅ OK" if ok else "⚠️ Possible autocorrelation",
                 "Remedy": "-" if ok else REMEDY["auto"]})
    jp = res["jb"][1]
    rows.append({"Assumption": "Normal residuals", "Test": "Jarque-Bera",
                 "Value": f"JB={res['jb'][0]:.2f}, p={jp:.4f}",
                 "Verdict": "✅ OK" if jp >= 0.05 else "⚠️ Not normal",
                 "Remedy": "-" if jp >= 0.05 else REMEDY["normal"]})
    return rows


def lab_summary_text(res):
    c = res["coef"]
    lines = [f"Model: {res['y']} ~ {' + '.join(res['xs'])}, n={res['n']}, R2={res['r2']:.3f}, "
             f"Adj R2={res['adj']:.3f}, F p={res['f_p']:.4f}"]
    for _, r in c.iterrows():
        lines.append(f"{r['Variable']}: coef={r['Coef']:.4f}, p={r['p-value']:.4f}, robust p={r['Robust p']:.4f}")
    for a in assumption_table(res):
        lines.append(f"{a['Test']}: {a['Value']} -> {a['Verdict']}")
    lines.append("VIF: " + ", ".join(f"{k}={v:.2f}" for k, v in res["vif"].items()))
    return "\n".join(lines)


# ================= FUN STUFF =================
JOKES = [
    "Heteroscedasticity: jab data ka mood swing hota hai, kabhi shant, kabhi hungama. 😅",
    "OLS ne kaha: 'Main BLUE hoon.' Residuals bole: 'Humein toh tum udaas lagte ho.' 😂",
    "Multicollinearity: jab do variables itne close ho jaayein ki regression bhi confuse ho jaye ki 'Ye wala kaun?' 👀",
    "Exam se ek raat pehle syllabus dekhna: 'Ye chapter kab padhaya tha?' 😭",
    "Student ka plan: kal subah 5 baje uthunga. Student ka alarm: 'Hum 5 baar baje, tum 5 baar snooze.' ⏰",
    "Notes banana: 2 ghante. Notes padhna: 2 minute. Notes kho jaana: 0 second. 📒",
    "P-value 0.06 dekh ke researcher: 'Bas thoda sa aur data daal dete hain...' 🙈",
    "Group study ka matlab: 5 minute padhai, 55 minute gossip. ☕",
    "Statistician aur doctor mein fark: doctor kehta hai 'aap theek ho jaoge', statistician kehta hai '95% confidence se'. 🩺",
    "Mean, median, mode: teeno alag nikle. Teacher: 'Chalo, kisi ek ko chun lo.' 🤷",
    "Library mein silent rehna: sabse mushkil exam wahi hota hai. 🤫",
    "Regression line: 'Main sabko saath le kar chalti hoon.' Outlier: 'Mujhe mat bulao.' 📈",
]
LOADING = [
    "Chai bana raha hoon, tab tak soch raha hoon... ☕",
    "Dimaag ke gears ghoom rahe hain... ⚙️",
    "Notes mein se jawab dhoondh raha hoon... 🔎",
    "Professor ki tarah gambheer ho raha hoon... 🧐",
    "Google se nahi, tumhare notes se bata raha hoon... 😎",
    "Ek minute, calculator dhoondh raha hoon... 🧮",
    "Mummy bol rahi hai padhai karo, wahi kar raha hoon... 📚",
]
SCORE_COMMENTS = [
    (90, "Topper alert! 🏆 Professor ko tumhara naam yaad rahega."),
    (70, "Zabardast! 🔥 Thoda aur aur tum legend ho."),
    (50, "Theek-thaak! 👍 Chai pi ke dobara try karo."),
    (30, "Arre yaar! 😅 Notes ek baar aur padh lo, himmat mat haro."),
    (0, "Koi baat nahi, galtiyon se hi seekhte hain. Agla quiz tumhara hai! 💪"),
]


def fun_spin():
    return random.choice(LOADING)


def score_comment(pct):
    for lim, txt in SCORE_COMMENTS:
        if pct >= lim:
            return txt
    return SCORE_COMMENTS[-1][1]


# ================= TIMER + QUIZ UI =================
def _timer_box(end_key, done_key):
    left = int(st.session_state.get(end_key, 0) - time.time())
    if left <= 0:
        st.session_state[done_key] = True
        st.rerun()
    m, s = divmod(left, 60)
    st.metric("⏱️ Bacha hua time", f"{m:02d}:{s:02d}")


try:
    timer_box = st.fragment(run_every=1)(_timer_box)
except Exception:
    timer_box = _timer_box


def start_quiz(p, data, label, end=None):
    st.session_state[f"{p}_data"] = data
    st.session_state[f"{p}_id"] = st.session_state.get(f"{p}_id", 0) + 1
    st.session_state[f"{p}_done"] = False
    st.session_state[f"{p}_result"] = None
    st.session_state[f"{p}_label"] = label
    if end:
        st.session_state[f"{p}_end"] = end


def render_q(p, qid, i, item, done):
    t = item.get("type", "mcq")
    key = f"{p}_{qid}_{i}"
    st.markdown(f"**Q{i + 1}. [{QTAG.get(t, 'MCQ')}] {item['q']}**")
    if t in ("mcq", "tf"):
        st.radio("Option chuno", item["options"], index=None, key=key,
                 label_visibility="collapsed", disabled=done)
    elif t == "fill":
        st.text_input("Jawab", key=key, label_visibility="collapsed", disabled=done)
    else:
        st.text_area("Apna jawab likho", key=key, disabled=done, height=120)


def _norm(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def grade_quiz(p, qid, quiz, negative):
    stt = {"points": 0.0, "right": 0, "wrong": 0, "skipped": 0}
    wrong_topics, right_topics, msgs, wrong_items, right_qs = [], [], [], [], []

    def good(item, topic, text, pts=1.0):
        stt["right"] += 1
        stt["points"] += pts
        right_topics.append(topic)
        right_qs.append(item["q"])
        msgs.append(("success", text))

    def bad(item, topic, text, yours, pts=0.0):
        stt["wrong"] += 1
        stt["points"] += pts
        wrong_topics.append(topic)
        wrong_items.append((item, yours))
        msgs.append(("error", text))

    def skip(text):
        stt["skipped"] += 1
        msgs.append(("warning", text))

    neg = -0.25 if negative else 0.0
    for i, item in enumerate(quiz):
        t = item.get("type", "mcq")
        ans = st.session_state.get(f"{p}_{qid}_{i}")
        topic = item.get("topic") or "general"
        expl = item.get("explanation", "")
        if t in ("mcq", "tf"):
            correct = item["options"][int(item["answer"])]
            if ans is None:
                skip(f"Q{i + 1}: Chhoda gaya ⏭️  Sahi jawab: {correct}. {expl}")
            elif ans == correct:
                good(item, topic, f"Q{i + 1}: Sahi ✅  {expl}")
            else:
                bad(item, topic, f"Q{i + 1}: Galat ❌  Sahi jawab: {correct}. {expl}", ans, neg)
        elif t == "fill":
            correct = str(item["answer"]).strip()
            if not (ans or "").strip():
                skip(f"Q{i + 1}: Chhoda gaya ⏭️  Sahi jawab: {correct}. {expl}")
            else:
                a, c = _norm(ans), _norm(correct)
                okf = a == c or (len(c) > 3 and c in a) or difflib.SequenceMatcher(None, a, c).ratio() >= 0.85
                if okf:
                    good(item, topic, f"Q{i + 1}: Sahi ✅  {expl}")
                else:
                    bad(item, topic, f"Q{i + 1}: Galat ❌  Sahi jawab: {correct}. {expl}", ans, neg)
        else:  # descriptive -> AI evaluation
            if not (ans or "").strip():
                skip(f"Q{i + 1}: Chhoda gaya ⏭️  Model answer: {item.get('answer', '')}")
            else:
                ev = evaluate_answer(item["q"], ans, str(item.get("answer", "")) + "\n" + expl, 5, lang)
                frac = ev["score"] / 5
                miss = "; ".join(ev.get("missing_points", [])[:3])
                text = (f"Q{i + 1}: {ev['score']:g}/5 — {ev.get('accuracy', '')} "
                        f"Missing: {miss or 'none'}. Model answer: {item.get('answer', '')}")
                if frac >= 0.6:
                    good(item, topic, text, frac)
                else:
                    bad(item, topic, text, ans, frac)
    total = len(quiz)
    xp = record_result(user, st.session_state.get(f"{p}_label", "Quiz"), total, stt["points"],
                       wrong_topics, right_topics, wrong_items, right_qs)
    return {"qid": qid, "points": stt["points"], "right": stt["right"], "wrong": stt["wrong"],
            "skipped": stt["skipped"], "msgs": msgs, "total": total, "xp": xp}


def run_quiz_ui(p, negative=False, timer=False):
    quiz = st.session_state.get(f"{p}_data")
    if not quiz:
        return
    qid = st.session_state[f"{p}_id"]
    done = st.session_state.get(f"{p}_done", False)

    if timer and not done:
        timer_box(f"{p}_end", f"{p}_done")
        done = st.session_state.get(f"{p}_done", False)

    for i, item in enumerate(quiz):
        render_q(p, qid, i, item, done)

    if not done:
        if st.button("Submit karo", key=f"{p}_submit"):
            st.session_state[f"{p}_done"] = True
            st.rerun()
        return

    res = st.session_state.get(f"{p}_result")
    if not res or res.get("qid") != qid:
        with st.spinner("Jawab check ho rahe hain..."):
            res = grade_quiz(p, qid, quiz, negative)
        st.session_state[f"{p}_result"] = res

    st.subheader(f"Score: {res['points']:g} / {res['total']}")
    st.caption(f"Sahi: {res['right']}  |  Galat: {res['wrong']}  |  Chhode: {res['skipped']}  |  ⭐ +{res['xp']} XP")
    _pct = 100 * res["points"] / res["total"] if res["total"] else 0
    st.info(score_comment(_pct))
    if _pct >= 80 and not res.get("fun_done"):
        st.balloons()
        res["fun_done"] = True
    if negative:
        st.caption("Negative marking: har galat objective jawab par -0.25")
    for kind, text in res["msgs"]:
        getattr(st, kind)(text)


# ================= SIDEBAR + NAVIGATION =================
P_DASH = "🏠 Dashboard"
P_CHAT = "💬 AI Tutor (Chat)"
P_VOICE = "🎙️ Voice Tutor"
P_QUIZ = "🧠 Quiz Generator"
P_EVAL = "✍️ Answer Evaluation"
P_CARDS = "🃏 Flashcards"
P_MAP = "🗺️ Mind Map"
P_MOCK = "⏱️ Mock Exam"
P_WEAK = "🎯 Weak Topics"
P_DOUBT = "🔍 Doubt Detection"
P_PLAN = "📅 Study Planner"
P_STATS = "📊 My Analytics"
P_BADGE = "🏅 Achievements"
P_PAPER = "📄 Research Paper Assistant"
P_CASE = "💼 Case Study Solver"
P_DEBATE = "⚔️ Debate Mode"
P_COMPARE = "⚖️ Compare Topics"
P_FORM = "🧮 Formula Sheet"
P_SUM = "📝 Summary"
P_SYL = "📚 Syllabus Tracker"
P_MIST = "📓 Mistake Notebook"
P_KIT = "🧾 Exam Revision Kit"
P_BUDDY = "🤝 Study Buddy"
P_LAB = "📈 Data Lab"
P_PATTERN = "🧐 Exam Pattern Trainer"
P_FUN = "🎭 Fun Zone"
P_TEACH = "👩‍🏫 Teacher Dashboard"

PAGES = [P_DASH, P_CHAT, P_VOICE, P_QUIZ, P_EVAL, P_CARDS, P_MAP, P_MOCK, P_WEAK, P_DOUBT,
         P_PLAN, P_STATS, P_BADGE, P_PAPER, P_CASE, P_DEBATE, P_COMPARE, P_FORM, P_SUM,
         P_SYL, P_MIST, P_KIT, P_BUDDY, P_LAB, P_PATTERN, P_FUN]
if role == "teacher":
    PAGES.append(P_TEACH)


def go_page(p):
    st.session_state["nav"] = p


st.sidebar.title("📚 AI Study Companion")
_prog = load_progress(user)
st.sidebar.write(f"👤 **{user}**" + (" (teacher)" if role == "teacher" else ""))
st.sidebar.caption(f"⭐ Level {level_of(_prog['xp'])}  |  {_prog['xp']} XP  |  🔥 {_prog['streak']} din")
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.radio("Kya karna hai?", PAGES, key="nav")
lang = st.sidebar.selectbox("🌐 Language", LANGS)
st.sidebar.selectbox("🎓 Explanation level", ["Beginner", "Intermediate", "Advanced"],
                     index=1, key="level_sel")
st.sidebar.text_input("Audience (optional)", key="aud", placeholder="e.g. MBA student")
st.sidebar.checkbox("🔒 Source-locked (sirf notes se jawab)", value=True, key="locked")

st.sidebar.divider()
st.sidebar.subheader("📂 Upload (apne aap add hoga)")

if "processed" not in st.session_state:
    st.session_state["processed"] = set()

files = st.sidebar.file_uploader(
    "PDF / TXT / DOCX", type=["pdf", "txt", "docx"], accept_multiple_files=True,
)

if files:
    for f in files:
        key = f"{f.name}-{f.size}"
        if key in st.session_state["processed"]:
            continue
        path = os.path.join(RAW, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        with st.spinner(f"{f.name} add ho rahi hai (scanned ho to time lagega)..."):
            n = ingest_file(path)
        st.session_state["processed"].add(key)
        if n == 0:
            st.sidebar.warning(f"{f.name}: text nahi mila (scanned hai aur OCR install nahi hai?)")
        else:
            st.sidebar.success(f"✅ {f.name}: add ho gayi")

all_files = list_files()

st.sidebar.divider()
st.sidebar.subheader("📄 Added files")
if all_files:
    for name in all_files:
        mark = "✅" if is_ingested(name) else "⚠️ (text nahi mila)"
        st.sidebar.write(f"{mark} {name}")
else:
    st.sidebar.caption("Abhi koi file nahi hai.")

st.sidebar.divider()
st.sidebar.subheader("🗑️ File delete")
if all_files:
    to_delete = st.sidebar.selectbox("File chuno", all_files, key="del_sel")
    if st.sidebar.button("Delete karo"):
        col.delete(where={"file": to_delete})
        build_index.clear()
        os.remove(os.path.join(RAW, to_delete))
        st.session_state["processed"] = {
            k for k in st.session_state["processed"]
            if not k.startswith(f"{to_delete}-")
        }
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []


def need_files():
    if not all_files:
        st.info("Pehle sidebar se koi file upload karo.")
        return True
    return False


# ================= PAGE: DASHBOARD =================
if page == P_DASH:
    st.title("📚 AI Study Companion")
    st.caption("Your personal AI-powered learning platform")
    prog = load_progress(user)
    lvl = level_of(prog["xp"])
    hist = prog["history"]
    avg = round(sum(h.get("pct", 0) for h in hist) / len(hist), 1) if hist else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("⭐ Level", lvl)
    m2.metric("XP", prog["xp"])
    m3.metric("🔥 Streak", f"{prog['streak']} din")
    m4.metric("Tests", len(hist))
    m5.metric("Avg accuracy", f"{avg}%")

    st.write(f"**Level {lvl} → {lvl + 1}**")
    st.progress((prog["xp"] % 100) / 100)
    st.write("**Learning progress (average accuracy)**")
    st.progress(min(avg / 100, 1.0))

    syl_d, _src_d = get_syllabus(user)
    if syl_d:
        st.write("**Syllabus coverage**")
        st.progress(coverage(prog, syl_d))

    st.subheader("Kahan jaana hai?")
    cards = [
        (P_CHAT, "Ask AI Tutor", "Notes se sawal poochho"),
        (P_QUIZ, "Generate Quiz", "Apna knowledge test karo"),
        (P_MAP, "Mind Maps", "Concepts visualize karo"),
        (P_MOCK, "Mock Exam", "Timed practice exam"),
        (P_PLAN, "Study Planner", "Preparation plan banao"),
        (P_STATS, "My Analytics", "Performance track karo"),
    ]
    cols = st.columns(3)
    for i, (pg, title, sub) in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(sub)
                st.button("Kholo", key=f"dash_{i}", on_click=go_page, args=(pg,))

    due = due_topics(prog)
    weak_now = list(dict(sorted(prog["weak"].items(), key=lambda x: -x[1])).keys())[:5]
    if weak_now or due:
        st.subheader("🧠 Bot ko tumhare baare mein ye yaad hai")
        if weak_now:
            st.write("**Kamzor topics:** " + ", ".join(t.title() for t in weak_now))
        if due:
            st.warning("⏰ **Revise karne ka time:** " + ", ".join(f"{t.title()} ({d} din pehle)" for t, d, _ in due[:6]))
        topics_q = list(dict.fromkeys(weak_now + [t for t, _, _ in due]))[:5]
        if st.button("⚡ 3 quick revision sawal"):
            with st.spinner(fun_spin()):
                qs_r, err_r = make_quiz(material_from_topics(topics_q), 3, "easy", lang, ("mcq", "tf"), topics_q)
            if qs_r:
                start_quiz("recall", qs_r, "Quick revision")
            else:
                st.error(err_r)
        run_quiz_ui("recall")

    st.subheader("😂 Aaj ka joke")
    st.info(JOKES[date.today().toordinal() % len(JOKES)])

    if prog["badges"]:
        st.subheader("Tumhare badges")
        st.write("  ".join(prog["badges"]))

# ================= PAGE: CHAT =================
elif page == P_CHAT:
    st.title("💬 AI Tutor")
    c1, c2, c3 = st.columns([2, 2, 2])
    selected = c1.selectbox("Kis file se jawab chahiye?", ["Sab files"] + all_files)
    mode = c2.selectbox("Jawab ka style", list(MODES.keys()))
    rerank = c3.checkbox("🔎 Better search (thoda slow)", value=False)

    def show_source(src):
        if not src:
            return
        with st.expander("📖 Source text dekho (jawab kahan se aaya)"):
            for s in src:
                st.markdown(f"**{s['file']} — page {s['page']}**")
                st.caption(s["text"])

    def add_exchange(q, answer, items, found):
        sources = sorted({f"{s['file']} (page {s['page']})" for s in items})
        full = answer + ("\n\n**Source:** " + ", ".join(sources) if sources else "")
        st.session_state.history.append({"role": "user", "msg": q, "q": q})
        st.session_state.history.append(
            {"role": "assistant", "msg": full, "src": items, "q": q, "found": found}
        )
        log_event({"type": "question", "user": user, "q": q, "file": selected,
                   "found": found, "pages": sources})
        award(2, "questions")

    voice_q = None
    with st.expander("🎤 Bolkar poochho"):
        if hasattr(st, "audio_input"):
            audio = st.audio_input("Record karo aur roko")
            if audio is not None:
                data = audio.getvalue()
                h = hashlib.md5(data).hexdigest()
                if st.session_state.get("last_audio") != h:
                    st.session_state["last_audio"] = h
                    with st.spinner("Sun raha hoon..."):
                        try:
                            voice_q = transcribe(data)
                        except Exception as e:
                            st.error(f"Awaaz samajh nahi aayi: {e}")
        else:
            st.caption("Voice ke liye terminal mein chalao: pip install -U streamlit")

    with st.expander("🖼️ Photo / diagram se poochho"):
        img = st.file_uploader("Image chuno", type=["png", "jpg", "jpeg"], key="img_up")
        iq = st.text_input("Is image ke baare mein sawal", key="img_q")
        if img is not None:
            st.image(img, width=300)
        if st.button("Image se jawab do"):
            if img is None:
                st.error("Pehle image upload karo.")
            else:
                with st.spinner("Image dekh raha hoon..."):
                    ans, items = ask_image(iq, img.getvalue(), selected, mode, lang, rerank)
                add_exchange("🖼️ " + (iq or "Image samjhao"), ans, items, True)
                st.rerun()

    for i, h in enumerate(st.session_state.history):
        with st.chat_message(h["role"]):
            st.write(h["msg"])
            if h["role"] == "assistant":
                show_source(h.get("src"))
                if h.get("fb"):
                    st.caption("Aapka feedback: " + ("👍" if h["fb"] == "up" else "👎"))
                else:
                    f1, f2, _ = st.columns([1, 1, 8])
                    if f1.button("👍", key=f"up_{i}"):
                        h["fb"] = "up"
                        log_event({"type": "feedback", "user": user, "q": h.get("q"), "vote": "up"})
                        st.rerun()
                    if f2.button("👎", key=f"down_{i}"):
                        h["fb"] = "down"
                        log_event({"type": "feedback", "user": user, "q": h.get("q"), "vote": "down"})
                        st.rerun()
                if role == "teacher":
                    with st.expander("✅ Verify / ✏️ Correct (teacher)"):
                        fixed = st.text_area("Sahi jawab", value=h["msg"].split("**Source:**")[0].strip(),
                                             key=f"vf_{i}", height=140)
                        if st.button("Verified answer save karo", key=f"vfb_{i}"):
                            save_verified(h.get("q") or "", fixed, user)
                            st.success("Save ho gaya. Ab ye sawal poochhne par yehi jawab milega.")
                elif not h.get("asked"):
                    if st.button("🙋 Teacher se poochho", key=f"td_{i}"):
                        add_doubt(user, h.get("q") or "", h["msg"])
                        h["asked"] = True
                        st.rerun()
                else:
                    st.caption("🙋 Teacher ko bheja gaya")
                if st.button("🔊 Sunao", key=f"tts_{i}"):
                    try:
                        with st.spinner("Awaaz ban rahi hai..."):
                            play_audio(speak(h["msg"], lang))
                    except ImportError:
                        st.info("Voice answer ke liye terminal mein chalao: pip install gtts")
                    except Exception as e:
                        st.error(f"Voice nahi ban paayi: {e}")

    mine_d = [d for d in jload(DOUBTS_FILE, []) if d.get("user") == user]
    if mine_d:
        n_ans = sum(1 for d in mine_d if d["status"] == "answered")
        with st.expander(f"📬 Mere doubts ({n_ans} jawab aaye)"):
            for d in mine_d[::-1]:
                st.markdown(f"**Q:** {d['q']}")
                if d["status"] == "answered":
                    st.success(f"👩‍🏫 {d.get('by', 'teacher')}: {d['answer']}")
                else:
                    st.caption("⏳ Teacher ke jawab ka intezaar")

    typed = st.chat_input("Apna sawal poochho...")
    q = typed or voice_q
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.spinner(fun_spin()):
            answer, items, found = ask(q, selected, mode, lang, st.session_state.history, rerank)
        add_exchange(q, answer, items, found)
        st.rerun()

    if st.session_state.history:
        b1, b2 = st.columns(2)
        text = "\n\n".join(f"{h['role'].upper()}: {h['msg']}" for h in st.session_state.history)
        b1.download_button("⬇️ Chat download karo", text, file_name="chat.txt")
        if b2.button("🧹 Chat saaf karo"):
            st.session_state.history = []
            st.rerun()

# ================= PAGE: VOICE TUTOR =================
elif page == P_VOICE:
    st.title("🎙️ Voice Tutor")
    st.caption("Bolkar poochho, aur jawab suno.")
    if not need_files():
        selected = st.selectbox("Kis file se?", ["Sab files"] + all_files)
        if not hasattr(st, "audio_input"):
            st.warning("Voice ke liye terminal mein chalao: pip install -U streamlit")
        else:
            audio = st.audio_input("Record karo aur roko")
            if audio is not None:
                data = audio.getvalue()
                h = hashlib.md5(data).hexdigest()
                if st.session_state.get("vt_audio") != h:
                    st.session_state["vt_audio"] = h
                    try:
                        with st.spinner("Sun raha hoon..."):
                            qv = transcribe(data)
                        with st.spinner("Soch raha hoon..."):
                            ans, items, found = ask(qv, selected, "Simple", lang, [], False)
                        audio_out = None
                        try:
                            audio_out = speak(ans, lang)
                        except Exception:
                            pass
                        st.session_state["vt"] = {"q": qv, "a": ans, "audio": audio_out, "items": items}
                        log_event({"type": "question", "user": user, "q": qv, "file": selected, "found": found,
                                   "pages": sorted({f"{s['file']} (page {s['page']})" for s in items})})
                        award(2, "questions")
                    except Exception as e:
                        st.error(f"Error: {e}")
            vt = st.session_state.get("vt")
            if vt:
                st.markdown(f"**🗣️ Tumne poochha:** {vt['q']}")
                st.markdown(vt["a"])
                if vt["audio"]:
                    play_audio(vt["audio"], autoplay=True)
                else:
                    st.caption("Voice answer ke liye terminal mein chalao: pip install gtts")
                if vt["items"]:
                    with st.expander("📖 Source"):
                        for s in vt["items"]:
                            st.markdown(f"**{s['file']} — page {s['page']}**")
                            st.caption(s["text"])

# ================= PAGE: QUIZ =================
elif page == P_QUIZ:
    st.title("🧠 AI Quiz Generator")
    if not need_files():
        f = st.selectbox("File chuno", all_files)
        c1, c2 = st.columns(2)
        n = c1.slider("Kitne sawal?", 3, 15, 5)
        level = c2.selectbox("Difficulty", ["easy", "medium", "hard"])
        tsel = st.multiselect("Question types", list(QTYPES.keys()), default=["MCQ", "True / False"])
        if st.button("Quiz banao"):
            if not tsel:
                st.error("Kam se kam ek type chuno.")
            else:
                with st.spinner("Quiz ban raha hai..."):
                    qs, err = make_quiz(material_from_files([f]), n, level, lang,
                                        [QTYPES[t] for t in tsel])
                if qs:
                    start_quiz("quiz", qs, f"Quiz: {f[:30]}")
                else:
                    st.error(err)
        run_quiz_ui("quiz")

# ================= PAGE: ANSWER EVALUATION =================
elif page == P_EVAL:
    st.title("✍️ Automatic Answer Evaluation")
    st.caption("Apna jawab likho, AI marks aur feedback dega.")
    if not need_files():
        sel = st.selectbox("Kis file se?", ["Sab files"] + all_files)
        c1, c2 = st.columns([3, 1])
        maxm = c2.selectbox("Marks", [5, 10])
        if c1.button("🎲 Mujhe ek sawal do"):
            mat = material_from_files(all_files if sel == "Sab files" else [sel], 5000)
            raw = safe_generate(
                f"Write ONE descriptive exam question worth {maxm} marks from this material. "
                f"{ls(lang)} Return only the question text.\n\n{mat}")
            st.session_state["ev_q"] = raw.strip()
        question = st.text_area("Sawal", key="ev_q", height=80)
        answer = st.text_area("Tumhara jawab", key="ev_a", height=200)
        if st.button("Evaluate karo"):
            if not question.strip() or not answer.strip():
                st.error("Sawal aur jawab dono likho.")
            else:
                with st.spinner("Check ho raha hai..."):
                    items = search(question, sel, 5)
                    ev = evaluate_answer(question, answer, ctx_text(items), maxm, lang)
                st.session_state["ev_res"] = (ev, maxm)
                award(5, "evals")
        if st.session_state.get("ev_res"):
            show_evaluation(*st.session_state["ev_res"])

# ================= PAGE: FLASHCARDS =================
elif page == P_CARDS:
    st.title("🃏 AI Flashcards")
    if not need_files():
        sel = st.multiselect("Kin files se cards chahiye?", all_files, default=all_files[:1])
        n = st.slider("Kitne cards?", 5, 20, 10)
        if st.button("Cards banao"):
            if not sel:
                st.error("Kam se kam ek file chuno.")
            else:
                with st.spinner("Cards ban rahe hain..."):
                    cards, err = make_flashcards(material_from_files(sel), n, lang)
                if cards:
                    st.session_state["fc_cards"] = cards
                    st.session_state["fc_i"] = 0
                    st.session_state["fc_show"] = False
                    st.session_state["fc_known"] = set()
                else:
                    st.error(err)

        cards = st.session_state.get("fc_cards")
        if cards:
            i = st.session_state["fc_i"]
            known = st.session_state["fc_known"]
            st.progress((i + 1) / len(cards))
            st.caption(f"Card {i + 1} / {len(cards)}   |   Yaad hain: {len(known)}")
            st.markdown(f"### ❓ {cards[i]['front']}")
            if st.session_state["fc_show"]:
                st.success(cards[i]["back"])

            def go(step):
                st.session_state["fc_i"] = min(max(i + step, 0), len(cards) - 1)
                st.session_state["fc_show"] = False

            b1, b2, b3, b4, b5 = st.columns(5)
            if b1.button("⬅️ Pichla"):
                go(-1)
                st.rerun()
            if b2.button("👁️ Answer"):
                st.session_state["fc_show"] = not st.session_state["fc_show"]
                st.rerun()
            if b3.button("➡️ Agla"):
                go(1)
                st.rerun()
            if b4.button("✅ Yaad hai"):
                if i not in known:
                    known.add(i)
                    award(2, "cards")
                go(1)
                st.rerun()
            if b5.button("❌ Dobara"):
                known.discard(i)
                go(1)
                st.rerun()

            if len(known) == len(cards):
                st.balloons()
                st.success("Saare cards yaad ho gaye! 🎉")

            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["Front", "Back"])
            for c in cards:
                w.writerow([c["front"], c["back"]])
            st.download_button("⬇️ Cards CSV download karo (Anki/Excel)", buf.getvalue(),
                               file_name="flashcards.csv")

# ================= PAGE: MIND MAP =================
elif page == P_MAP:
    st.title("🗺️ Mind Map Generator")
    if not need_files():
        sel = st.multiselect("Kin files / chapters se?", all_files, default=all_files[:1])
        topic = st.text_input("Focus topic (optional)", placeholder="Heteroscedasticity")
        if st.button("Mind map banao"):
            if not sel:
                st.error("Kam se kam ek file chuno.")
            else:
                with st.spinner("Mind map ban raha hai..."):
                    mat = material_from_topics([topic]) if topic.strip() else ordered_text(sel, 9000)
                    mm, err = make_mindmap(mat, topic.strip(), lang)
                if mm:
                    st.session_state["mindmap"] = mm
                    award(3, "maps")
                else:
                    st.error(err)
        mm = st.session_state.get("mindmap")
        if mm:
            dot = mindmap_dot(mm)
            st.graphviz_chart(dot, use_container_width=True)
            with st.expander("📋 Text outline"):
                st.markdown(mindmap_outline(mm))
            st.download_button("⬇️ Mind map (DOT) download karo", dot, file_name="mindmap.dot")

# ================= PAGE: MOCK EXAM =================
elif page == P_MOCK:
    st.title("⏱️ Exam Simulation")
    if not need_files():
        sel = st.multiselect("Kin files se test chahiye?", all_files, default=all_files[:1])
        c1, c2, c3 = st.columns(3)
        n = c1.slider("Kitne sawal?", 5, 20, 10)
        minutes = c2.slider("Kitne minute?", 5, 90, 15)
        level = c3.selectbox("Level", ["easy", "medium", "hard"], index=1)
        tsel = st.multiselect("Question types", list(QTYPES.keys()), default=["MCQ", "True / False"])
        neg = st.checkbox("Negative marking (-0.25 har galat objective par)", value=True)
        if st.button("▶️ Test shuru karo"):
            if not sel or not tsel:
                st.error("Files aur question types chuno.")
            else:
                with st.spinner("Test ban raha hai..."):
                    qs, err = make_quiz(material_from_files(sel), n, level, lang,
                                        [QTYPES[t] for t in tsel])
                if qs:
                    start_quiz("mock", qs, "Mock Exam", end=time.time() + minutes * 60)
                else:
                    st.error(err)
        run_quiz_ui("mock", negative=neg, timer=True)

# ================= PAGE: WEAK TOPICS =================
elif page == P_WEAK:
    st.title("🎯 Weak Topics")
    prog = load_progress(user)
    weak = dict(sorted(prog["weak"].items(), key=lambda x: -x[1]))
    if not weak:
        st.info("Abhi koi weak topic nahi hai. Quiz ya Mock Exam do, galat jawab yahan save honge.")
    else:
        st.subheader("Jin topics par galti hui")
        for t, c in weak.items():
            st.write(f"• **{t.title()}** — {c} baar galat")
        c1, c2 = st.columns(2)
        n = c1.slider("Kitne sawal?", 3, 10, 5, key="weak_n")
        level = c2.selectbox("Level", ["easy", "medium", "hard"], key="weak_level")
        if st.button("Weak topics par practice quiz banao"):
            topics = list(weak.keys())[:5]
            with st.spinner("Practice quiz ban raha hai..."):
                qs, err = make_quiz(material_from_topics(topics), n, level, lang, ("mcq", "tf"), topics)
            if qs:
                start_quiz("weak", qs, "Weak topics practice")
            else:
                st.error(err)
        run_quiz_ui("weak")

    if prog["history"]:
        st.divider()
        st.subheader("Test history")
        st.dataframe(prog["history"][::-1])

    if weak or prog["history"]:
        if st.button("🧹 Weak topics aur history saaf karo"):
            p = load_progress(user)
            p["weak"], p["history"], p["topics"] = {}, [], {}
            save_progress(user, p)
            for k in ("weak_data", "weak_done", "weak_result"):
                st.session_state.pop(k, None)
            st.rerun()

# ================= PAGE: DOUBT DETECTION =================
elif page == P_DOUBT:
    st.title("🔍 Doubt Detection")
    st.caption("Wo concepts jinme tum baar-baar atakte ho.")
    logs = [l for l in read_logs() if l.get("type") == "question" and l.get("user") == user]
    prog = load_progress(user)
    if not logs and not prog["weak"]:
        st.info("Abhi data nahi hai. Kuch sawal poochho aur quiz do, phir yahan analysis dikhega.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total sawal", len(logs))
        c2.metric("Jawab nahi mila", sum(1 for l in logs if not l.get("found")))
        c3.metric("Weak topics", len(prog["weak"]))

        rep = Counter((l.get("q") or "").strip().lower() for l in logs if l.get("q"))
        rep = [(q, c) for q, c in rep.most_common(10) if c >= 2]
        st.subheader("Baar-baar poochhe gaye sawal")
        if rep:
            st.dataframe([{"Sawal": q, "Baar": c} for q, c in rep])
        else:
            st.caption("Abhi koi sawal 2 baar se zyada nahi poochha gaya.")

        pages = Counter(p for l in logs for p in (l.get("pages") or []))
        st.subheader("Jin pages par sabse zyada wapas aaye")
        if pages:
            st.dataframe([{"Page": p, "Baar": c} for p, c in pages.most_common(8)])
        else:
            st.caption("Abhi data nahi.")

        if st.button("🧠 AI se doubt analysis karao"):
            qs_txt = "\n".join(f"- {l.get('q')}" for l in logs[-40:])
            weak_txt = ", ".join(prog["weak"].keys()) or "none"
            with st.spinner("Analysis ho raha hai..."):
                st.session_state["doubt"] = safe_generate(
                    f"""A student asked these questions to a study chatbot:
{qs_txt}
Quiz weak topics: {weak_txt}
Group them into the concepts the student is struggling with. For each concept give: why it is likely confusing,
a 2-line quick fix, and what to practise. Finish with a priority list of 3 concepts. {ls(lang)}""")
        if st.session_state.get("doubt"):
            st.markdown(st.session_state["doubt"])

        st.divider()
        st.subheader("Concept dobara samjho")
        opts = list(prog["weak"].keys())
        pick = st.selectbox("Weak topic chuno (ya neeche likho)", [""] + opts)
        custom = st.text_input("Ya apna concept likho")
        concept = custom.strip() or pick
        if st.button("📘 Mini-lesson do") and concept:
            with st.spinner("Lesson ban raha hai..."):
                items = search(concept, "Sab files", 5)
                st.session_state["mini"] = safe_generate(
                    f"""Teach the concept "{concept}" ONLY from the material below: simple explanation, one worked example,
a common mistake, and 3 quick practice questions with answers. {ls(lang)}

Material:
{ctx_text(items)}""")
        if st.session_state.get("mini"):
            st.markdown(st.session_state["mini"])

# ================= PAGE: STUDY PLANNER =================
elif page == P_PLAN:
    st.title("📅 Smart Study Planner")
    sel = st.multiselect("Kaunse chapters / files?", all_files, default=all_files)
    note = st.text_input("Extra subjects / topics (optional)", placeholder="Statistics, Marketing...")
    c1, c2 = st.columns(2)
    exam = c1.date_input("Exam ki date", value=date.today() + timedelta(days=14),
                         min_value=date.today())
    hours = c2.slider("Roz kitne ghante padhoge?", 1, 10, 3)
    if st.button("Plan banao"):
        if not sel and not note.strip():
            st.error("Kam se kam ek file chuno ya subjects likho.")
        else:
            with st.spinner("Plan ban raha hai..."):
                st.session_state["plan"] = make_plan(sel, exam, hours, lang, note)
    if st.session_state.get("plan"):
        st.markdown(st.session_state["plan"])
        st.download_button("⬇️ Plan download karo", st.session_state["plan"],
                           file_name="revision_plan.txt")

# ================= PAGE: ANALYTICS =================
elif page == P_STATS:
    st.title("📊 Student Performance Analytics")
    prog = load_progress(user)
    hist = prog["history"]
    if not hist:
        st.info("Abhi koi quiz/test nahi diya. Ek quiz do, phir analytics dikhega.")
    else:
        df = pd.DataFrame(hist)
        df["Test #"] = range(1, len(df) + 1)
        accs = df["pct"].tolist()
        first = sum(accs[:3]) / len(accs[:3])
        last = sum(accs[-3:]) / len(accs[-3:])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tests", len(df))
        m2.metric("Average accuracy", f"{sum(accs) / len(accs):.1f}%")
        m3.metric("Best score", f"{max(accs):.0f}%")
        m4.metric("Improvement", f"{last - first:+.1f}%")

        st.subheader("Score over time")
        st.line_chart(df.set_index("Test #")["pct"])

        topics = prog["topics"]
        if topics:
            tdf = pd.DataFrame(
                [{"Topic": t.title(), "Right": r, "Total": tot,
                  "Accuracy %": round(100 * r / tot, 1)} for t, (r, tot) in topics.items() if tot]
            )
            st.subheader("Topic-wise accuracy")
            st.bar_chart(tdf.set_index("Topic")["Accuracy %"])
            strong = tdf[(tdf["Accuracy %"] >= 70) & (tdf["Total"] >= 2)]
            weakdf = tdf[tdf["Accuracy %"] < 60]
            c1, c2 = st.columns(2)
            c1.subheader("💪 Strong topics")
            c1.write(", ".join(strong["Topic"]) if len(strong) else "Abhi koi nahi (kam se kam 2 sawal chahiye).")
            c2.subheader("⚠️ Weak topics")
            c2.write(", ".join(weakdf["Topic"]) if len(weakdf) else "Koi nahi 🎉")
            with st.expander("Poori table"):
                st.dataframe(tdf.sort_values("Accuracy %"))

        act = prog["activity"]
        if act:
            days = [(date.today() - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
            adf = pd.DataFrame({"Din": days, "Activities": [act.get(d, 0) for d in days]})
            st.subheader("Pichhle 14 din ki activity")
            st.bar_chart(adf.set_index("Din")["Activities"])

        with st.expander("Test history"):
            st.dataframe(df[["time", "test", "score", "pct"]][::-1])

# ================= PAGE: ACHIEVEMENTS =================
elif page == P_BADGE:
    st.title("🏅 Achievements")
    prog = load_progress(user)
    lvl = level_of(prog["xp"])
    c1, c2, c3 = st.columns(3)
    c1.metric("⭐ Level", lvl)
    c2.metric("XP", prog["xp"])
    c3.metric("🔥 Best streak", f"{prog['best_streak']} din")
    st.progress((prog["xp"] % 100) / 100, text=f"Level {lvl + 1} tak {100 - prog['xp'] % 100} XP baaki")
    st.subheader("Badges")
    cols = st.columns(2)
    for i, (name, desc, _) in enumerate(BADGES):
        got = name in prog["badges"]
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"{'✅' if got else '🔒'} **{name}**")
                st.caption(desc)
    st.caption("XP kaise milta hai: sawal +2, quiz sahi jawab +10, 80%+ par +20 bonus, answer evaluation +5, flashcard +2, mind map +3.")

# ================= PAGE: RESEARCH PAPER =================
elif page == P_PAPER:
    st.title("📄 Research Paper Assistant")
    if not need_files():
        f = st.selectbox("Paper chuno", all_files)
        if st.button("Paper analyse karo"):
            with st.spinner("Paper padh raha hoon..."):
                st.session_state["paper"] = research_assistant(f, lang)
            award(3, "papers")
        if st.session_state.get("paper"):
            st.markdown(st.session_state["paper"])
            st.download_button("⬇️ Download karo", st.session_state["paper"], file_name="paper_analysis.md")
        st.divider()
        pq = st.text_input("Is paper ke baare mein koi sawal")
        if st.button("Paper se poochho") and pq.strip():
            with st.spinner("Soch raha hoon..."):
                ans, items, _ = ask(pq, f, "Detailed", lang, [], False)
            st.markdown(ans)

# ================= PAGE: CASE STUDY =================
elif page == P_CASE:
    st.title("💼 Case Study Solver")
    if not need_files():
        sel = st.multiselect("Kin files se?", all_files, default=all_files[:1])
        topic = st.text_input("Topic (optional)", placeholder="Heteroscedasticity in wage data")
        if st.button("Case study banao"):
            if not sel:
                st.error("Kam se kam ek file chuno.")
            else:
                with st.spinner("Case study ban rahi hai..."):
                    mat = material_from_topics([topic]) if topic.strip() else material_from_files(sel, 6000)
                    cs, err = make_case(mat, topic.strip(), lang)
                if cs:
                    st.session_state["case"] = cs
                    st.session_state["case_mat"] = mat
                    st.session_state["case_id"] = st.session_state.get("case_id", 0) + 1
                    st.session_state["case_res"] = None
                else:
                    st.error(err)
        cs = st.session_state.get("case")
        if cs:
            cid = st.session_state["case_id"]
            st.subheader(cs.get("title", "Case Study"))
            st.write(cs["case"])
            qs = cs["questions"][:3]
            for i, q in enumerate(qs):
                st.markdown(f"**Q{i + 1}. {q}**")
                st.text_area("Jawab", key=f"case_{cid}_{i}", height=110, label_visibility="collapsed")
            if st.button("Mere jawab evaluate karo"):
                res = []
                with st.spinner("Check ho raha hai..."):
                    for i, q in enumerate(qs):
                        a = st.session_state.get(f"case_{cid}_{i}", "")
                        if a.strip():
                            ctxt = cs["case"] + "\n\n" + st.session_state.get("case_mat", "")
                            res.append((q, evaluate_answer(q, a, ctxt, 10, lang)))
                        else:
                            res.append((q, None))
                st.session_state["case_res"] = res
                award(8, "evals", len([r for r in res if r[1]]))
            if st.session_state.get("case_res"):
                total = 0
                for i, (q, ev) in enumerate(st.session_state["case_res"]):
                    st.markdown(f"### Q{i + 1}")
                    if ev is None:
                        st.warning("Jawab nahi likha gaya.")
                    else:
                        total += ev["score"]
                        show_evaluation(ev, 10)
                st.success(f"Total: {total:g} / {10 * len(qs)}")

# ================= PAGE: DEBATE =================
elif page == P_DEBATE:
    st.title("⚔️ AI Debate Mode")
    st.caption("AI tumhare jawab ko challenge karega, taaki concept pakka ho.")
    if not need_files():
        deb = st.session_state.get("debate")
        if not deb:
            topic = st.text_input("Debate topic / concept",
                                  placeholder="Heteroscedasticity makes OLS biased")
            stance = st.text_input("Tumhara stance (optional)")
            if st.button("Debate shuru karo"):
                if not topic.strip():
                    st.error("Topic likho.")
                else:
                    items = search(topic, "Sab files", 5)
                    d = {"topic": topic.strip(), "stance": stance.strip(),
                         "ctx": ctx_text(items), "msgs": [], "done": False, "result": None}
                    with st.spinner("AI pehla sawal soch raha hai..."):
                        d["msgs"].append({"role": "assistant", "content": debate_turn(d, lang)})
                    st.session_state["debate"] = d
                    st.rerun()
        else:
            st.markdown(f"**Topic:** {deb['topic']}")
            for m in deb["msgs"]:
                with st.chat_message(m["role"]):
                    st.write(m["content"])
            if not deb["done"]:
                reply = st.chat_input("Apna argument likho...")
                if reply:
                    deb["msgs"].append({"role": "user", "content": reply})
                    with st.spinner("AI jawab de raha hai..."):
                        deb["msgs"].append({"role": "assistant", "content": debate_turn(deb, lang)})
                    st.rerun()
                b1, b2 = st.columns(2)
                if b1.button("🏁 Debate khatam karo aur score do"):
                    with st.spinner("Score ban raha hai..."):
                        deb["result"] = debate_score(deb, lang)
                    deb["done"] = True
                    award(int(deb["result"]["score"]) * 2, "debates")
                    st.rerun()
                if b2.button("↩️ Naya debate"):
                    st.session_state.pop("debate")
                    st.rerun()
            else:
                r = deb["result"]
                st.subheader(f"Score: {r['score']:g} / 10")
                st.progress(r["score"] / 10)
                if r.get("strengths"):
                    st.markdown("**💪 Strengths**")
                    for s in r["strengths"]:
                        st.write(f"- {s}")
                if r.get("gaps"):
                    st.markdown("**❗ Gaps**")
                    for s in r["gaps"]:
                        st.write(f"- {s}")
                st.info(r.get("advice", ""))
                if st.button("↩️ Naya debate"):
                    st.session_state.pop("debate")
                    st.rerun()

# ================= PAGE: COMPARE =================
elif page == P_COMPARE:
    st.title("⚖️ Compare Topics")
    if not need_files():
        sel = st.selectbox("Kis file mein dhundhna hai?", ["Sab files"] + all_files)
        c1, c2 = st.columns(2)
        ta = c1.text_input("Topic 1", placeholder="Heteroscedasticity")
        tb = c2.text_input("Topic 2", placeholder="Multicollinearity")
        better = st.checkbox("🔎 Better search (thoda slow)")
        if st.button("Compare karo"):
            if not ta.strip() or not tb.strip():
                st.error("Dono topics likho.")
            else:
                with st.spinner("Compare ho raha hai..."):
                    st.session_state["cmp"] = compare_topics(ta, tb, sel, lang, better)
        if st.session_state.get("cmp"):
            st.markdown(st.session_state["cmp"])
            st.download_button("⬇️ Download karo", st.session_state["cmp"], file_name="compare.md")

# ================= PAGE: FORMULA SHEET =================
elif page == P_FORM:
    st.title("🧮 Formula Sheet")
    if not need_files():
        sel = st.multiselect("Kin files se formulas?", all_files, default=all_files)
        if st.button("Formula sheet banao"):
            if not sel:
                st.error("Kam se kam ek file chuno.")
            else:
                with st.spinner("Formulas nikal raha hoon..."):
                    st.session_state["formulas"] = make_formula_sheet(sel, lang)
        if st.session_state.get("formulas"):
            st.markdown(st.session_state["formulas"])
            st.download_button("⬇️ Download karo", st.session_state["formulas"],
                               file_name="formula_sheet.md")

# ================= PAGE: SUMMARY =================
elif page == P_SUM:
    st.title("📝 Summary")
    if not need_files():
        f = st.selectbox("File chuno", all_files)
        style = st.selectbox("Kya chahiye?", ["Short summary", "Key points", "Important terms"])
        if st.button("Summary banao"):
            with st.spinner("Summary ban rahi hai..."):
                st.session_state["summary"] = summarize(f, style, lang)
        if st.session_state.get("summary"):
            st.markdown(st.session_state["summary"])
            st.download_button("⬇️ Summary download karo", st.session_state["summary"],
                               file_name="summary.txt")

# ================= PAGE: SYLLABUS TRACKER =================
elif page == P_SYL:
    st.title("📚 Syllabus Coverage Tracker")
    prog = load_progress(user)
    syl, src = get_syllabus(user)
    can_edit = role == "teacher" or src == "personal"

    if syl:
        cov = coverage(prog, syl)
        st.metric("Syllabus coverage", f"{cov * 100:.0f}%")
        st.progress(cov)
        st.caption("Source: teacher ka syllabus" if src == "teacher" else "Source: tumhara apna syllabus")
        done = set(prog["syllabus_done"])
        new_done = set()
        for i, tpc in enumerate(syl):
            r, tot = topic_match(prog["topics"], tpc)
            c1, c2 = st.columns([3, 2])
            if c1.checkbox(tpc, value=tpc in done, key=f"syl_{i}"):
                new_done.add(tpc)
            if tot:
                c2.caption(f"🧪 Quiz accuracy: {100 * r / tot:.0f}% ({tot} sawal)")
            else:
                c2.caption("⬜ Abhi quiz nahi diya")
        if new_done != done:
            gained = len(new_done - done)
            p = load_progress(user)
            p["syllabus_done"] = [t for t in syl if t in new_done]
            new = _bump(p, 3 * gained, "syllabus", gained) if gained > 0 else []
            save_progress(user, p)
            for b in new:
                st.toast(f"🏅 Naya badge: {b}")
            st.rerun()
        left = [t for t in syl if t not in done and topic_match(prog["topics"], t)[1] == 0]
        if left:
            st.info("Abhi tak shuru nahi kiye: " + ", ".join(left[:8]))
    else:
        st.info("Abhi syllabus set nahi hai." + (" Neeche se set karo." if can_edit else " Teacher ke set karne ka intezaar karo."))

    if can_edit:
        with st.expander("✏️ Syllabus set / edit karo", expanded=not syl):
            if "syl_text" not in st.session_state:
                st.session_state["syl_text"] = "\n".join(syl)
            if all_files and st.button("🪄 Files se topics apne aap nikalo"):
                with st.spinner(fun_spin()):
                    found_t = extract_syllabus(ordered_text(all_files, 9000))
                if found_t:
                    st.session_state["syl_text"] = "\n".join(found_t)
                else:
                    st.error("Topics nahi nikal paye. Haath se likho.")
            txt = st.text_area("Har line mein ek topic", key="syl_text", height=220)
            if st.button("💾 Syllabus save karo"):
                topics_new = [t.strip() for t in txt.splitlines() if t.strip()]
                if role == "teacher":
                    jsave(SYLLABUS_FILE, {"topics": topics_new, "by": user})
                else:
                    p = load_progress(user)
                    p["my_syllabus"] = topics_new
                    save_progress(user, p)
                st.success("Save ho gaya.")
                st.rerun()

# ================= PAGE: MISTAKE NOTEBOOK =================
elif page == P_MIST:
    st.title("📓 Mistake Notebook")
    st.caption("Tumhari har galti yahan save hoti hai. Sahi jawab dene par wo list se hat jati hai.")
    prog = load_progress(user)
    ms = prog["mistakes"]
    if not ms:
        st.info("Abhi koi galti save nahi hai. Quiz ya Mock Exam do, galat jawab yahan aayenge.")
    else:
        topics_m = sorted({m.get("topic", "general") for m in ms})
        c1, c2 = st.columns(2)
        c1.metric("Total galtiyan", len(ms))
        flt = c2.selectbox("Topic filter", ["Sab"] + topics_m)
        view = [m for m in ms if flt == "Sab" or m.get("topic") == flt]
        for m in sorted(view, key=lambda x: -x.get("count", 1)):
            if m["type"] in ("mcq", "tf") and m.get("options"):
                correct = m["options"][int(m["answer"])]
            else:
                correct = str(m["answer"])
            with st.expander(f"❌ {m['q'][:90]}  (×{m.get('count', 1)})"):
                st.markdown(f"**Topic:** {m.get('topic', 'general')}")
                st.markdown(f"**Tumhara jawab:** {m.get('your', '-')}")
                st.markdown(f"**Sahi jawab:** {correct}")
                if m.get("explanation"):
                    st.caption(m["explanation"])
        st.divider()
        nret = 1
        if len(view) > 1:
            nret = st.slider("Kitne sawal dobara try karne hain?", 1, min(10, len(view)), min(5, len(view)))
        if view and st.button("🔁 In galtiyon ka retry quiz"):
            pick = random.sample(view, min(nret, len(view)))
            items = [{"type": m["type"], "q": m["q"], "options": m.get("options"), "answer": m["answer"],
                      "topic": m.get("topic", "general"), "explanation": m.get("explanation", "")} for m in pick]
            start_quiz("mist", items, "Mistake retry")
        run_quiz_ui("mist")
        if st.button("🧹 Notebook saaf karo"):
            p = load_progress(user)
            p["mistakes"] = []
            save_progress(user, p)
            st.rerun()

# ================= PAGE: EXAM REVISION KIT =================
elif page == P_KIT:
    st.title("🧾 Smart Exam Revision Kit")
    prog = load_progress(user)
    try:
        saved = date.fromisoformat(prog.get("exam_date", "")) if prog.get("exam_date") else date.today() + timedelta(days=7)
    except Exception:
        saved = date.today() + timedelta(days=7)
    exam = st.date_input("Exam ki date", value=max(saved, date.today()), min_value=date.today())
    if exam.isoformat() != prog.get("exam_date"):
        p = load_progress(user)
        p["exam_date"] = exam.isoformat()
        save_progress(user, p)
    left_days = (exam - date.today()).days
    c1, c2, c3 = st.columns(3)
    c1.metric("⏳ Din baaki", left_days)
    due = due_topics(prog)
    weak_sorted = list(dict(sorted(prog["weak"].items(), key=lambda x: -x[1])).keys())
    c2.metric("⚠️ Weak topics", len(weak_sorted))
    c3.metric("📓 Galtiyan", len(prog["mistakes"]))
    if left_days <= 3:
        st.warning("🔥 Last 3 din! Sirf kamzor topics aur galtiyan revise karo, naya chapter mat shuru karo.")
    if due:
        st.write("**⏰ Forgetting-curve alert (ye topics bhoolne wale hain):**")
        st.write(", ".join(f"{t.title()} ({d} din pehle)" for t, d, _ in due[:10]))
    if weak_sorted:
        st.write("**Kamzor topics:** " + ", ".join(t.title() for t in weak_sorted[:10]))

    if st.button("📄 1-page cheat sheet banao"):
        topics_k = list(dict.fromkeys(weak_sorted[:6] + [t for t, _, _ in due[:4]]))
        with st.spinner(fun_spin()):
            if topics_k:
                mat = material_from_topics(topics_k)
            elif all_files:
                mat = ordered_text(all_files, 8000)
            else:
                mat = ""
            if not mat.strip():
                st.session_state["kit"] = "Material nahi mila. Pehle file upload karo."
            else:
                mist = "\n".join(f"- {m['q']}" for m in prog["mistakes"][-6:])
                st.session_state["kit"] = safe_generate(
                    f"""Create a ONE-PAGE last-minute revision cheat sheet for a student, ONLY from the material below.
Focus topics: {', '.join(topics_k) or 'all main topics'}.
Include: key definitions, important formulas (LaTeX between $), common mistakes, and 5 quick-check questions with answers.
The student's recent mistakes (address them): 
{mist or 'none'}
{ls(lang)}

Material:
{mat}""")
    if st.session_state.get("kit"):
        st.markdown(st.session_state["kit"])
        st.download_button("⬇️ Cheat sheet download karo", st.session_state["kit"], file_name="cheat_sheet.md")

# ================= PAGE: STUDY BUDDY =================
elif page == P_BUDDY:
    st.title("🤝 Study Buddy Matching")
    if user == "guest":
        st.info("Study Buddy ke liye account banao (Guest ke liye available nahi hai).")
    else:
        me = load_progress(user)
        opt = st.checkbox("Mujhe buddy suggestions mein dikhao (mera username doosre dekh payenge)",
                          value=bool(me.get("buddy")))
        if opt != bool(me.get("buddy")):
            me["buddy"] = opt
            save_progress(user, me)
            st.rerun()
        if not opt:
            st.info("Buddy dhoondhne ke liye upar opt-in karo.")
        else:
            my_weak = list(me["weak"].keys())[:8]
            my_strong = [t for t, (r, tot) in me["topics"].items() if tot >= 2 and r / tot >= 0.7]
            helpers, needers = {}, {}
            for nm in load_users():
                if nm == user:
                    continue
                pr = load_progress(nm)
                if not pr.get("buddy"):
                    continue
                for w in my_weak:
                    r, tot = topic_acc(pr, w)
                    if tot >= 2 and r / tot >= 0.7:
                        helpers.setdefault(w, []).append(f"{nm} ({100 * r / tot:.0f}%)")
                for s in my_strong:
                    if any(difflib.SequenceMatcher(None, s, k).ratio() >= 0.75 for k in pr["weak"]):
                        needers.setdefault(s, []).append(nm)
            st.subheader("🙋 Ye log tumhari madad kar sakte hain")
            if helpers:
                st.dataframe([{"Tumhara weak topic": t.title(), "Buddy (accuracy)": ", ".join(v)} for t, v in helpers.items()])
            else:
                st.caption("Abhi koi match nahi mila (aur students ke opt-in aur quiz dene ka intezaar).")
            st.subheader("💪 Tum inki madad kar sakte ho")
            if needers:
                st.dataframe([{"Tumhara strong topic": t.title(), "Students": ", ".join(v)} for t, v in needers.items()])
            else:
                st.caption("Abhi koi match nahi.")

            msgs = jload(MSG_FILE, [])
            st.subheader("✉️ Message bhejo")
            pool = sorted({nm for nm in load_users() if nm != user and load_progress(nm).get("buddy")})
            if pool:
                to = st.selectbox("Kisko?", pool)
                body = st.text_input("Message", placeholder="Heteroscedasticity samjha doge?")
                if st.button("Bhejo") and body.strip():
                    msgs.append({"from": user, "to": to, "text": body.strip()[:300],
                                 "time": time.strftime("%d-%m-%Y %H:%M")})
                    jsave(MSG_FILE, msgs)
                    st.success("Message bhej diya.")
            else:
                st.caption("Abhi koi aur student opt-in nahi hai.")
            st.subheader("📥 Inbox")
            mine = [m for m in msgs if m["to"] == user]
            if mine:
                for m in mine[::-1][:15]:
                    st.info(f"**{m['from']}** ({m['time']}): {m['text']}")
            else:
                st.caption("Koi message nahi.")

# ================= PAGE: DATA LAB =================
elif page == P_LAB:
    st.title("📈 Data Lab — Live Tests")
    st.caption("Dataset upload karo, regression chalao, assumptions check karo aur notes ke saath samjho.")
    try:
        from scipy import stats as _sps
        have_sps = True
    except Exception:
        have_sps = False
    if not have_sps:
        st.error("scipy nahi mila. Terminal mein chalao: pip install scipy")
    else:
        c1, c2 = st.columns([3, 1])
        up = c1.file_uploader("CSV file upload karo", type=["csv"], key="lab_up")
        if c2.button("📦 Sample data try karo"):
            st.session_state["lab_df"] = sample_hetero_df()
            st.session_state["lab_res"] = None
        if up is not None:
            ukey = f"{up.name}-{up.size}"
            if st.session_state.get("lab_key") != ukey:
                try:
                    st.session_state["lab_df"] = pd.read_csv(up)
                    st.session_state["lab_key"] = ukey
                    st.session_state["lab_res"] = None
                except Exception as e:
                    st.error(f"CSV padh nahi paya: {e}")
        df = st.session_state.get("lab_df")
        if df is None:
            st.info("CSV upload karo ya 'Sample data' dabao (isme heteroscedasticity jaan-boojh kar rakhi gayi hai).")
        else:
            st.dataframe(df.head(10))
            num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num) < 2:
                st.error("Kam se kam 2 numeric columns chahiye.")
            else:
                yc = st.selectbox("Dependent variable (Y)", num, index=len(num) - 1)
                xopts = [c for c in num if c != yc]
                xs = st.multiselect("Independent variables (X)", xopts, default=xopts[:3])
                if st.button("▶️ Regression chalao"):
                    if not xs:
                        st.error("Kam se kam ek X variable chuno.")
                    else:
                        try:
                            res, err = run_regression(df, yc, xs)
                        except Exception as e:
                            res, err = None, f"Regression fail hua: {e}"
                        if res:
                            st.session_state["lab_res"] = res
                            st.session_state["lab_expl"] = ""
                            idx = random.randrange(len(res["xs"]))
                            st.session_state["lab_target"] = idx
                            award(5, "labs")
                        else:
                            st.error(err)

        res = st.session_state.get("lab_res")
        if res:
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("n", res["n"])
            m2.metric("R²", f"{res['r2']:.3f}")
            m3.metric("Adj R²", f"{res['adj']:.3f}")
            m4.metric("F-test p", f"{res['f_p']:.4f}")
            st.subheader("Regression output")
            st.dataframe(res["coef"].round(4))
            st.subheader("Residuals vs Fitted")
            try:
                st.scatter_chart(res["resid"], x="Fitted", y="Residual")
            except Exception:
                st.line_chart(res["resid"].set_index("Fitted")["Residual"])
            st.subheader("✅ Assumption checker")
            st.dataframe(assumption_table(res))
            if res["bp"][1] < 0.05:
                st.warning("Heteroscedasticity mili. 'Robust SE (HC1)' wale columns dekho: standard errors ko sahi karte hain, coefficients wahi rehte hain.")
            if st.button("📖 Notes ke saath samjhao"):
                with st.spinner(fun_spin()):
                    items = search("heteroscedasticity multicollinearity autocorrelation regression assumptions remedies", "Sab files", 6)
                    st.session_state["lab_expl"] = safe_generate(
                        f"""Explain these regression results to a student using the study material for concepts.
Cover: model fit, which variables are significant, which assumptions are violated and the remedies, and what to do next.
Mention the page of the notes where relevant. {ls(lang)}

Results:
{lab_summary_text(res)}

Study material:
{ctx_text(items) if items else '(no notes uploaded)'}""")
            if st.session_state.get("lab_expl"):
                st.markdown(st.session_state["lab_expl"])

            st.divider()
            st.subheader("🎓 Interpretation trainer")
            tgt = st.session_state.get("lab_target", 0) % len(res["xs"])
            row = res["coef"].iloc[tgt + 1]
            qtxt = (f"Regression output mein '{row['Variable']}' ka coefficient {row['Coef']:.3f} hai "
                    f"(p = {row['p-value']:.4f}). Iska matlab apne shabdon mein samjhao.")
            st.write(qtxt)
            ans = st.text_area("Tumhari interpretation", key="lab_ans", height=120)
            if st.button("Meri interpretation check karo"):
                if not ans.strip():
                    st.error("Pehle likho.")
                else:
                    ref = (f"Correct interpretation: holding the other variables constant, a one-unit increase in "
                           f"{row['Variable']} changes {res['y']} by about {row['Coef']:.3f} units on average. "
                           f"p-value {row['p-value']:.4f}: "
                           f"{'statistically significant at 5%' if row['p-value'] < 0.05 else 'not significant at 5%, so we cannot conclude a real effect'}. "
                           "Good answers mention units, ceteris paribus, direction (sign), and statistical significance.")
                    with st.spinner(fun_spin()):
                        ev = evaluate_answer(qtxt, ans, ref, 5, lang)
                    st.session_state["lab_ev"] = ev
                    award(5, "evals")
            if st.session_state.get("lab_ev"):
                show_evaluation(st.session_state["lab_ev"], 5)
            if st.button("🎲 Dusra coefficient do"):
                st.session_state["lab_target"] = random.randrange(len(res["xs"]))
                st.session_state["lab_ev"] = None
                st.rerun()

# ================= PAGE: EXAM PATTERN TRAINER =================
elif page == P_PATTERN:
    st.title("🧐 Exam Pattern Trainer")
    st.caption("Purane papers upload karo (sidebar se). Bot seekhega ki tumhare professor ka paper kaisa aata hai.")
    if not need_files():
        old = st.multiselect("Kaunsi files purane papers hain?", all_files)
        if st.button("📊 Pattern analyse karo"):
            if not old:
                st.error("Kam se kam ek purana paper chuno.")
            else:
                with st.spinner(fun_spin()):
                    pat, err = analyse_pattern(ordered_text(old, 14000))
                if pat:
                    jsave(PATTERN_FILE, pat)
                    st.session_state["pattern"] = pat
                    award(4, "patterns")
                else:
                    st.error(err)
        pat = st.session_state.get("pattern") or jload(PATTERN_FILE, None)
        if pat:
            st.subheader("Paper ka structure")
            st.dataframe(pat.get("sections", []))
            m1, m2 = st.columns(2)
            m1.metric("Total marks", pat.get("total_marks", "?"))
            m2.metric("Duration", pat.get("duration", "?"))
            hot = pat.get("hot_topics") or []
            if hot:
                st.subheader("🔥 Baar-baar aane wale topics")
                hdf = pd.DataFrame(hot)
                if {"topic", "times"} <= set(hdf.columns):
                    st.bar_chart(hdf.set_index("topic")["times"])
            if pat.get("style_notes"):
                st.info("**Question style:** " + str(pat["style_notes"]))

            st.divider()
            st.subheader("📝 Isi pattern ka naya mock paper")
            src_files = st.multiselect("Content kin files se?", all_files, default=[f for f in all_files if f not in old][:2] or all_files[:1])
            boost = st.checkbox("Hot topics ko zyada weight do", value=True)
            if st.button("🧪 Mock paper banao"):
                if not src_files:
                    st.error("Content files chuno.")
                else:
                    with st.spinner(fun_spin()):
                        st.session_state["paper_out"] = make_paper(pat, ordered_text(src_files, 10000), boost, lang)
            if st.session_state.get("paper_out"):
                st.markdown(st.session_state["paper_out"])
                st.download_button("⬇️ Paper download karo", st.session_state["paper_out"], file_name="mock_paper.md")

# ================= PAGE: FUN ZONE =================
elif page == P_FUN:
    st.title("🎭 Fun Zone")
    st.caption("Padhai se break lo, thoda hasao, phir wapas padho! 😄")

    t1, t2, t3, t4 = st.tabs(["😂 Joke", "🎵 Yaad karne ka funny trick", "🔥 Study habits roast", "💬 Motivation"])

    with t1:
        if st.button("Ek aur joke!"):
            st.session_state["joke"] = random.choice(JOKES)
        st.success(st.session_state.get("joke", random.choice(JOKES)))
        topic = st.text_input("Apne notes ke topic par joke chahiye?", placeholder="Heteroscedasticity")
        if st.button("🤣 Topic par joke banao"):
            if not topic.strip():
                st.error("Topic likho.")
            else:
                with st.spinner(fun_spin()):
                    items = search(topic, "Sab files", 3)
                    st.session_state["ai_joke"] = safe_generate(
                        f"""Write 3 short, clean, funny jokes or one-liners (Hinglish) about the study topic "{topic}".
Each must still hint at the real concept correctly. Material for reference:
{ctx_text(items)}""")
        if st.session_state.get("ai_joke"):
            st.markdown(st.session_state["ai_joke"])

    with t2:
        st.write("Mushkil concept ko funny rhyme ya mnemonic mein badlo, taaki exam mein yaad rahe.")
        mt = st.text_input("Concept", placeholder="Properties of OLS estimators", key="mn_topic")
        kind = st.radio("Kya chahiye?", ["Funny rhyme", "Mnemonic (short form)", "Filmy dialogue"], horizontal=True)
        if st.button("✨ Trick banao"):
            if not mt.strip():
                st.error("Concept likho.")
            else:
                with st.spinner(fun_spin()):
                    items = search(mt, "Sab files", 4)
                    st.session_state["mnemonic"] = safe_generate(
                        f"""Create a {kind} in Hinglish to help a student remember "{mt}".
It must be factually correct according to the material below and easy to memorise. After it, give a 2-line plain explanation.
Material:
{ctx_text(items)}""")
        if st.session_state.get("mnemonic"):
            st.markdown(st.session_state["mnemonic"])

    with t3:
        st.write("Apni padhai ki aadat batao, AI pyaar se roast karega (aur ek asli tip bhi dega).")
        c1, c2 = st.columns(2)
        hrs = c1.slider("Roz kitne ghante padhte ho?", 0, 12, 2)
        phone = c2.slider("Phone par kitne ghante?", 0, 12, 5)
        hab = st.text_input("Koi special aadat?", placeholder="Exam se ek raat pehle padhta hoon")
        if st.button("🔥 Roast karo"):
            with st.spinner(fun_spin()):
                st.session_state["roast"] = safe_generate(
                    f"""Playfully and kindly roast a student in Hinglish (clean, no insults about family, appearance or identity).
Study hours per day: {hrs}. Phone hours per day: {phone}. Habit: {hab or 'none'}.
Keep it to 4-5 lines and end with one genuinely useful study tip.""")
        if st.session_state.get("roast"):
            st.warning(st.session_state["roast"])

    with t4:
        quotes = [
            "Aaj ka thoda padha, kal ka bada fayda. 📚",
            "Topper bhi kabhi 'Chapter 1' par hi tha. 🌱",
            "Ek aur quiz, ek aur step. Chalo shuru karte hain! 🚀",
            "Neend zaroori hai, par pehle ye chapter khatam kar lo. 😴",
            "Mistakes ka matlab: dimaag seekh raha hai. 🧠",
            "Jo aaj mehnat karta hai, wo exam hall mein muskurata hai. 😎",
        ]
        if st.button("💬 Motivation do"):
            st.session_state["quote"] = random.choice(quotes)
        st.success(st.session_state.get("quote", random.choice(quotes)))
        if st.button("🎉 Celebrate karo"):
            st.balloons()
            award(1)

# ================= PAGE: TEACHER DASHBOARD =================
else:
    st.title("👩‍🏫 Teacher Dashboard")
    t_over, t_heat, t_doubt, t_ver = st.tabs(
        ["📊 Overview", "🔥 Class Heatmap", "🙋 Doubt Queue", "✅ Verified Answers"])
    with t_over:
        logs = read_logs()
        qs = [l for l in logs if l.get("type") == "question"]
        fbs = [l for l in logs if l.get("type") == "feedback"]
        ups = sum(1 for l in fbs if l.get("vote") == "up")
        downs = sum(1 for l in fbs if l.get("vote") == "down")
        unanswered = [l for l in qs if not l.get("found")]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Students", len(load_users()))
        m2.metric("Total sawal", len(qs))
        m3.metric("Jawab nahi mila", len(unanswered))
        m4.metric("👍", ups)
        m5.metric("👎", downs)

        st.subheader("Sabse zyada poochhe gaye sawal")
        top = Counter((l.get("q") or "").strip().lower() for l in qs if l.get("q")).most_common(10)
        if top:
            st.dataframe([{"Sawal": q, "Baar": c} for q, c in top])
        else:
            st.caption("Abhi koi sawal nahi.")

        st.subheader("Jin sawalon ka jawab material mein nahi mila")
        if unanswered:
            st.dataframe([
                {"Time": l.get("time"), "Student": l.get("user"), "Sawal": l.get("q")}
                for l in unanswered[::-1][:30]
            ])
        else:
            st.caption("Koi nahi.")

        st.subheader("Jin jawabon ko 👎 mila")
        bad = [l for l in fbs if l.get("vote") == "down"]
        if bad:
            st.dataframe([
                {"Time": l.get("time"), "Student": l.get("user"), "Sawal": l.get("q")}
                for l in bad[::-1][:30]
            ])
        else:
            st.caption("Koi nahi.")

        st.subheader("Students ki activity")
        names = sorted(set(list(load_users().keys()) + ["guest"]))
        qcount = Counter(l.get("user") for l in qs)
        rows, weak_all = [], Counter()
        for nm in names:
            pr = load_progress(nm)
            pcts = [h["pct"] for h in pr["history"] if isinstance(h.get("pct"), (int, float))]
            rows.append({
                "Student": nm,
                "Level": level_of(pr["xp"]),
                "XP": pr["xp"],
                "Streak": pr["streak"],
                "Sawal": qcount.get(nm, 0),
                "Tests": len(pr["history"]),
                "Avg %": round(sum(pcts) / len(pcts), 1) if pcts else None,
            })
            for t, c in pr["weak"].items():
                weak_all[t] += c
        st.dataframe(rows)

        st.subheader("Class ke weak topics")
        if weak_all:
            st.dataframe([{"Topic": t.title(), "Galtiyan": c} for t, c in weak_all.most_common(15)])
        else:
            st.caption("Abhi koi data nahi.")

        if logs:
            st.download_button(
                "⬇️ Poora log download karo",
                "\n".join(json.dumps(l, ensure_ascii=False) for l in logs),
                file_name="logs.jsonl",
            )

    with t_heat:
        st.subheader("🔥 Class doubt heatmap")
        st.caption("Jis page par zyada students ne sawal poochhe, wo gehra laal hai. Agla lecture wahin se shuru karo.")
        qs_h = [l for l in read_logs() if l.get("type") == "question"]
        heat = {}
        for l in qs_h:
            for s in l.get("pages") or []:
                mm = re.match(r"^(.*) \(page (\d+)\)$", s)
                if mm:
                    heat.setdefault(mm.group(1), Counter())[int(mm.group(2))] += 1
        if not heat:
            st.info("Abhi data nahi hai. Students ke sawal poochhne ke baad heatmap dikhega.")
        else:
            for fname, cnt in heat.items():
                st.markdown(f"**{fname}**")
                mx = max(cnt.values())
                maxp = max(cnt)
                cells = "".join(
                    f'<span title="page {pg}: {cnt.get(pg, 0)} sawal" style="display:inline-block;width:34px;height:34px;line-height:34px;text-align:center;margin:2px;border-radius:6px;font-size:12px;color:#111;background:rgba(239,68,68,{(0.12 + 0.88 * cnt.get(pg, 0) / mx) if cnt.get(pg, 0) else 0.06});">{pg}</span>'
                    for pg in range(1, maxp + 1)
                )
                st.markdown(cells, unsafe_allow_html=True)
            hot = Counter(s for l in qs_h for s in (l.get("pages") or [])).most_common(8)
            st.subheader("Sabse zyada confusing pages")
            st.dataframe([{"Page": p, "Sawal": c} for p, c in hot])

            weak_class = Counter()
            for nm in sorted(set(list(load_users().keys()) + ["guest"])):
                for t, c in load_progress(nm)["weak"].items():
                    weak_class[t] += c
            unans = [l.get("q") for l in qs_h if not l.get("found")][-8:]
            if st.button("🧠 Agle lecture ke liye suggestion"):
                with st.spinner(fun_spin()):
                    st.session_state["lec"] = safe_generate(
                        f"""You are advising a teacher. Based on class data, suggest what to cover in the next lecture.
Most confusing pages (file, page, number of questions): {hot}
Class weak topics (topic, mistakes): {weak_class.most_common(8)}
Questions the bot could not answer from the material: {unans}
Give: top 3 topics to re-teach, a 10-minute activity idea for each, and gaps in the study material. {ls(lang)}""")
            if st.session_state.get("lec"):
                st.markdown(st.session_state["lec"])

    with t_doubt:
        doubts = jload(DOUBTS_FILE, [])
        opend = [d for d in doubts if d["status"] == "open"]
        st.subheader(f"🙋 Open doubts ({len(opend)})")
        if not opend:
            st.caption("Koi open doubt nahi.")
        for d in opend[::-1]:
            with st.expander(f"{d['user']}: {d['q'][:80]}"):
                st.caption(f"Time: {d['time']}")
                st.markdown("**Bot ka jawab:**")
                st.write(d.get("bot", ""))
                ans_t = st.text_area("Tumhara jawab", key=f"da_{d['id']}", height=120)
                if st.button("Jawab bhejo aur verified bana do", key=f"db_{d['id']}"):
                    if ans_t.strip():
                        for x in doubts:
                            if x["id"] == d["id"]:
                                x["status"] = "answered"
                                x["answer"] = ans_t.strip()
                                x["by"] = user
                        jsave(DOUBTS_FILE, doubts)
                        save_verified(d["q"], ans_t.strip(), user)
                        st.rerun()
                    else:
                        st.error("Jawab likho.")
        answered = [d for d in doubts if d["status"] == "answered"]
        st.subheader(f"Answered ({len(answered)})")
        if answered:
            st.dataframe([{"Student": d["user"], "Sawal": d["q"], "Jawab": d["answer"]} for d in answered[::-1][:30]])

    with t_ver:
        ver = jload(VERIFIED_FILE, [])
        st.subheader(f"✅ Teacher-verified answers ({len(ver)})")
        st.caption("Jab koi student inhi sawalon ke jaise sawal poochhta hai, bot seedha ye jawab deta hai.")
        if not ver:
            st.caption("Abhi koi nahi. AI Tutor page par kisi jawab ko verify karo ya Doubt Queue mein jawab do.")
        for i, v in enumerate(ver):
            with st.expander(v["q"][:90]):
                st.write(v["a"])
                st.caption(f"By {v.get('by', '')} | {v.get('time', '')}")
                if st.button("🗑️ Hatao", key=f"vd_{i}"):
                    ver.pop(i)
                    jsave(VERIFIED_FILE, ver)
                    st.rerun()