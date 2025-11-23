# AI-Powered-Chatbot
Build an intelligent chatbot using Natural Language Processing (NLP) for customer support or FAQs
# AI-Powered Chatbot

Simple AI chatbot for customer support and FAQs using:
- Semantic FAQ matching with SentenceTransformers
- Conversational fallback with DialoGPT (Transformers)
- FastAPI backend
- SQLite interaction logging
- Minimal HTML/JS frontend

## Features
- Semantic search of FAQ entries
- Contextual responses and fallback generation
- Interaction logging (SQLite)
- Easy to extend FAQ CSV
- Ready for local deployment

## If you want a demo screenshot style output, here is how it looks:
https://chatgpt.com/s/m_6922f40d2d5481919b389dbbdaa847c9

1. Create virtual env and activate
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
python app.py
# or
uvicorn app:app --reload
ai_chatbot/
├─ app.py
├─ chatbot.py
├─ db.py
├─ faqs.csv
├─ requirements.txt
├─ README.md
├─ README_GITHUB.md
├─ linkedin_post.txt
├─ demo_screenshot_instructions.txt
├─ templates/
│  └─ index.html
└─ static/
   └─ chat.js
fastapi==0.95.2
uvicorn[standard]==0.22.0
transformers==4.35.0
sentence-transformers==2.2.2
torch>=1.13.0
nltk==3.8.1
python-multipart==0.0.6
Jinja2==3.1.2
id,question,answer,category
1,How do I reset my password?,To reset your password click 'Forgot password' on login page and follow the steps.,Account
2,What are your support hours?,Our support team is available Monday to Friday, 9am to 6pm IST.,Support
3,How do I change my subscription plan?,Go to Settings > Billing > Change Plan. Follow the checkout steps.,Billing
4,Do you have an API?,Yes. We provide a REST API for paid plans. Email sales to get API keys.,Product
5,How can I contact support?,You can email support@example.com or use the chat widget.,Support
import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = "chat_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        matched_faq_id INTEGER,
        score REAL,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_interaction(user_id: Optional[str], user_message: str, bot_response: str, matched_faq_id: Optional[int], score: Optional[float]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO interactions (user_id, user_message, bot_response, matched_faq_id, score, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, user_message, bot_response, matched_faq_id if matched_faq_id is not None else None, score if score is not None else None, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
import csv
import os
from typing import List, Dict, Optional, Tuple

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# download nltk data once
nltk_data_dir = os.path.expanduser("~/.nltk_data_chatbot")
os.environ["NLTK_DATA"] = nltk_data_dir
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("omw-1.4", download_dir=nltk_data_dir)

LEM = WordNetLemmatizer()

class Chatbot:
    def __init__(self, faq_path: str = "faqs.csv", embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", gen_model: str = "microsoft/DialoGPT-small", similarity_threshold: float = 0.65):
        # 1) Load FAQs
        self.faqs = self._load_faqs(faq_path)
        # 2) sentence-transformers for semantic matching
        self.embedder = SentenceTransformer(embed_model_name)
        self.faq_texts = [f"{faq['question']} {faq['answer']}" for faq in self.faqs]
        self.faq_embeddings = self.embedder.encode(self.faq_texts, convert_to_tensor=True)
        # 3) fallback conversational generator
        self.gen = pipeline("conversational", model=gen_model, device=-1)  # cpu by default; change device if GPU
        self.similarity_threshold = similarity_threshold

    def _load_faqs(self, path: str) -> List[Dict]:
        faqs = []
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                faqs.append({
                    "id": int(row.get("id") or 0),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "category": row.get("category", "")
                })
        return faqs

    def _preprocess(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        lemmas = [LEM.lemmatize(t) for t in tokens]
        return " ".join(lemmas)

    def find_best_faq(self, user_text: str) -> Tuple[Optional[Dict], float]:
        # embed and compute similarity
        emb = self.embedder.encode(user_text, convert_to_tensor=True)
        hits = util.semantic_search(emb, self.faq_embeddings, top_k=3)
        if not hits:
            return None, 0.0
        best = hits[0][0]  # top match
        score = float(best["score"])
        idx = best["corpus_id"]
        if score >= self.similarity_threshold:
            return self.faqs[idx], score
        return None, score

    def generate_response(self, user_text: str) -> str:
        # simple conversational fallback
        try:
            convs = self.gen(user_text, max_length=150)
            if isinstance(convs, list) and len(convs) > 0:
                return convs[0]["generated_text"] if "generated_text" in convs[0] else str(convs[0])
            # transformers pipeline may return a string
            return str(convs)
        except Exception:
            # last-resort safe reply
            return "Sorry, I am having trouble forming an answer right now. Please try again later."

    def get_response(self, user_text: str) -> Tuple[str, Optional[int], float]:
        # check FAQs first
        faq, score = self.find_best_faq(user_text)
        if faq:
            return faq["answer"], faq["id"], score
        # else generate
        gen = self.generate_response(user_text)
        return gen, None, score
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import uuid

from chatbot import Chatbot
import db

app = FastAPI(title="AI-Powered Chatbot")

# initialize DB
db.init_db()

# mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# instantiate chatbot (this may download models on first start)
chat = Chatbot(faq_path="faqs.csv")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_api(message: str = Form(...), user_id: str = Form(None)):
    if not user_id:
        user_id = str(uuid.uuid4())  # anonymous ID
    user_text = message
    bot_response, matched_faq_id, score = chat.get_response(user_text)
    # log interaction
    db.log_interaction(user_id, user_text, bot_response, matched_faq_id, score)
    return JSONResponse({"response": bot_response, "matched_faq_id": matched_faq_id, "score": score, "user_id": user_id})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI Chatbot</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 0; padding: 0; display:flex; flex-direction:column; height:100vh;}
    .header { background:#0f172a; color:white; padding:16px; }
    .container { display:flex; flex-direction:column; flex:1; padding:16px; gap:12px; }
    .messages { flex:1; border:1px solid #ddd; padding:12px; overflow:auto; border-radius:8px; background:#f9fafb;}
    .msg { margin:8px 0; }
    .user { text-align:right; }
    .bot { text-align:left; color:#0b5; }
    .input-row { display:flex; gap:8px; }
    input[type="text"] { flex:1; padding:10px; border-radius:6px; border:1px solid #ccc; }
    button { padding:10px 14px; border-radius:6px; cursor:pointer; border:none; background:#0f172a; color:white; }
    .meta { font-size:12px; color:#666; }
  </style>
</head>
<body>
  <div class="header"><h3>AI-Powered Chatbot</h3><div class="meta">Contextual FAQ + fallback conversational model</div></div>
  <div class="container">
    <div id="messages" class="messages"></div>
    <form id="chatForm" class="input-row">
      <input id="message" type="text" placeholder="Type your question..." autocomplete="off" />
      <button type="submit">Send</button>
    </form>
  </div>

  <script src="/static/chat.js"></script>
</body>
</html>
const form = document.getElementById('chatForm');
const msgInput = document.getElementById('message');
const messages = document.getElementById('messages');

let userId = localStorage.getItem('chat_user_id') || null;

function appendMessage(text, who='bot') {
  const div = document.createElement('div');
  div.className = 'msg ' + (who === 'user' ? 'user' : 'bot');
  div.innerText = (who === 'user' ? 'You: ' : 'Bot: ') + text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = msgInput.value.trim();
  if (!text) return;
  appendMessage(text, 'user');
  msgInput.value = '';
  const fd = new FormData();
  fd.append('message', text);
  if (userId) fd.append('user_id', userId);
  const resp = await fetch('/api/chat', { method: 'POST', body: fd });
  const data = await resp.json();
  if (data.user_id) {
    userId = data.user_id;
    localStorage.setItem('chat_user_id', userId);
  }
  appendMessage(data.response, 'bot');
});



