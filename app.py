import os
import json
import time
import random
import fitz  # PyMuPDF
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ---------- API setup ----------
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None
GROQ_MODEL = os.getenv("GROQ_MODEL", "")

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

st.set_page_config(page_title="Study Chatbot", page_icon="📚", layout="wide")


@st.cache_resource
def load():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path="db").get_or_create_collection("study")
    return embedder, col


embedder, col = load()


# ---------- File reading ----------
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
        return [(i, " ".join(p.get_text().split())) for i, p in enumerate(doc, 1)]
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
    return count


def list_files():
    return sorted(os.listdir(RAW))


# ---------- Startup: database khali ho to files se bana lo ----------
if col.count() == 0:
    for fname in list_files():
        try:
            ingest_file(os.path.join(RAW, fname))
        except Exception:
            pass


# ---------- LLM call (available model apne aap chunta hai) ----------
@st.cache_resource
def get_models():
    if not groq_client:
        return []
    ids = [m.id for m in groq_client.models.list().data]
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
                model=m,
                messages=[{"role": "user", "content": prompt}],
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


# ---------- Helpers ----------
MODES = {
    "Simple": "Explain in very simple words, like to a beginner.",
    "Detailed": "Give a detailed, well-structured explanation with headings or bullet points.",
    "Exam-style": "Answer like a good exam answer: definition, key points, formula if any, short conclusion.",
    "Hinglish": "Explain in easy Hinglish (Hindi written in English letters mixed with English terms).",
}


def get_file_chunks(name):
    data = col.get(where={"file": name})
    items = list(zip(data["metadatas"], data["documents"]))
    items.sort(key=lambda x: x[0]["page"])
    return [d for _, d in items]


def ask(question, selected, mode, history):
    if col.count() == 0:
        return "Abhi koi study material upload nahi hua hai.", []
    kwargs = {}
    if selected != "Sab files":
        kwargs["where"] = {"file": selected}
    res = col.query(
        query_embeddings=[embedder.encode(question).tolist()],
        n_results=4,
        **kwargs,
    )
    docs, metas = res["documents"][0], res["metadatas"][0]
    if not docs:
        return "Ye study material mein nahi mila.", []
    context = "\n\n".join(docs)
    hist = "\n".join(
        f"{r}: {m.split('**Source:**')[0].strip()}" for r, m in history[-6:]
    )
    prompt = f"""You are a study assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I couldn't find this in the study material."
{MODES[mode]}
Reply in the same language as the question (unless the style asks for Hinglish).

Earlier conversation (for follow-up questions):
{hist}

Context:
{context}

Question: {question}"""
    return safe_generate(prompt), metas


def summarize(name, style):
    chunks = get_file_chunks(name)
    if not chunks:
        return "Is file mein text nahi mila."
    text = " ".join(chunks)
    ask_style = {
        "Short summary": "Write a short summary in 8-10 lines.",
        "Key points": "List the key points as clear bullet points.",
        "Important terms": "List important terms and formulas with one-line meanings.",
    }[style]
    parts = [text[i:i + 9000] for i in range(0, len(text), 9000)][:6]
    if len(parts) == 1:
        return safe_generate(f"{ask_style}\n\nStudy material:\n{parts[0]}")
    partial = []
    for p in parts:
        partial.append(safe_generate(f"Summarize briefly in 6 lines:\n\n{p}"))
    combined = "\n\n".join(partial)
    return safe_generate(f"{ask_style}\n\nNotes:\n{combined}")


def make_quiz(name, n, level):
    chunks = get_file_chunks(name)
    if not chunks:
        return None, "Is file mein text nahi mila."
    random.shuffle(chunks)
    text, total = [], 0
    for c in chunks:
        if total > 8000:
            break
        text.append(c)
        total += len(c)
    material = "\n".join(text)
    prompt = f"""Create {n} multiple choice questions ({level} level) ONLY from the study material below.
Return ONLY valid JSON, no extra text, in this exact format:
{{"questions":[{{"q":"question text","options":["A","B","C","D"],"answer":0,"explanation":"short reason"}}]}}
"answer" is the index (0-3) of the correct option.

Study material:
{material}"""
    raw = safe_generate(prompt)
    try:
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        qs = [
            q for q in data["questions"]
            if len(q["options"]) == 4 and 0 <= int(q["answer"]) <= 3
        ]
        if not qs:
            return None, "Quiz ban nahi paya. Dobara try karo."
        return qs, ""
    except Exception:
        return None, f"Quiz format galat aaya, dobara try karo.\n\n{raw[:300]}"


# ---------- Sidebar ----------
st.sidebar.title("📚 Study Bot")
page = st.sidebar.radio("Kya karna hai?", ["💬 Chat", "📝 Summary", "🧠 Quiz"])

st.sidebar.divider()
st.sidebar.subheader("📂 Upload")
files = st.sidebar.file_uploader(
    "PDF / TXT / DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)
if files and st.sidebar.button("Chatbot mein add karo"):
    for f in files:
        path = os.path.join(RAW, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        with st.spinner(f"{f.name} process ho rahi hai..."):
            n = ingest_file(path)
        if n == 0:
            st.sidebar.warning(f"{f.name}: text nahi mila (scanned file ho sakti hai)")
        else:
            st.sidebar.success(f"{f.name}: add ho gayi")

all_files = list_files()

st.sidebar.divider()
st.sidebar.subheader("🗑️ File delete")
if all_files:
    to_delete = st.sidebar.selectbox("File chuno", all_files, key="del_sel")
    if st.sidebar.button("Delete karo"):
        col.delete(where={"file": to_delete})
        os.remove(os.path.join(RAW, to_delete))
        st.rerun()
else:
    st.sidebar.caption("Abhi koi file nahi hai.")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Page: Chat ----------
if page == "💬 Chat":
    st.title("💬 Study Chatbot")
    c1, c2 = st.columns(2)
    selected = c1.selectbox("Kis file se jawab chahiye?", ["Sab files"] + all_files)
    mode = c2.selectbox("Jawab ka style", list(MODES.keys()))

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    q = st.chat_input("Apna sawal poochho...")
    if q:
        st.chat_message("user").write(q)
        with st.spinner("Soch raha hoon..."):
            answer, metas = ask(q, selected, mode, st.session_state.history)
        sources = sorted({f"{m['file']} (page {m['page']})" for m in metas})
        full = answer + ("\n\n**Source:** " + ", ".join(sources) if sources else "")
        st.chat_message("assistant").write(full)
        st.session_state.history += [("user", q), ("assistant", full)]

    if st.session_state.history:
        b1, b2 = st.columns(2)
        text = "\n\n".join(f"{r.upper()}: {m}" for r, m in st.session_state.history)
        b1.download_button("⬇️ Chat download karo", text, file_name="chat.txt")
        if b2.button("🧹 Chat saaf karo"):
            st.session_state.history = []
            st.rerun()

# ---------- Page: Summary ----------
elif page == "📝 Summary":
    st.title("📝 Summary")
    if not all_files:
        st.info("Pehle sidebar se koi file upload karo.")
    else:
        f = st.selectbox("File chuno", all_files)
        style = st.selectbox("Kya chahiye?", ["Short summary", "Key points", "Important terms"])
        if st.button("Summary banao"):
            with st.spinner("Summary ban rahi hai..."):
                st.session_state["summary"] = summarize(f, style)
        if st.session_state.get("summary"):
            st.markdown(st.session_state["summary"])
            st.download_button(
                "⬇️ Summary download karo", st.session_state["summary"],
                file_name="summary.txt",
            )

# ---------- Page: Quiz ----------
else:
    st.title("🧠 Quiz")
    if not all_files:
        st.info("Pehle sidebar se koi file upload karo.")
    else:
        f = st.selectbox("File chuno", all_files)
        c1, c2 = st.columns(2)
        n = c1.slider("Kitne sawal?", 3, 10, 5)
        level = c2.selectbox("Level", ["easy", "medium", "hard"])
        if st.button("Quiz banao"):
            with st.spinner("Quiz ban raha hai..."):
                qs, err = make_quiz(f, n, level)
            if qs:
                st.session_state["quiz"] = qs
                st.session_state["quiz_id"] = st.session_state.get("quiz_id", 0) + 1
                st.session_state["quiz_done"] = False
            else:
                st.error(err)

        quiz = st.session_state.get("quiz")
        if quiz:
            qid = st.session_state["quiz_id"]
            for i, item in enumerate(quiz):
                st.markdown(f"**Q{i + 1}. {item['q']}**")
                st.radio(
                    "Option chuno", item["options"], index=None,
                    key=f"quiz_{qid}_{i}", label_visibility="collapsed",
                )
            if st.button("Submit karo"):
                st.session_state["quiz_done"] = True

            if st.session_state.get("quiz_done"):
                score = 0
                for i, item in enumerate(quiz):
                    chosen = st.session_state.get(f"quiz_{qid}_{i}")
                    correct = item["options"][int(item["answer"])]
                    if chosen == correct:
                        score += 1
                        st.success(f"Q{i + 1}: Sahi ✅  {item.get('explanation', '')}")
                    else:
                        st.error(
                            f"Q{i + 1}: Galat ❌  Sahi jawab: {correct}. "
                            f"{item.get('explanation', '')}"
                        )
                st.subheader(f"Score: {score} / {len(quiz)}")