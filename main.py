from fastapi import FastAPI, Request
import sqlite3
import chromadb
from google import genai
from google.genai import types
from google.api_core import retry

# ========================
# Configuración de Gemini
# ========================
# GOOGLE_API_KEY = "TU_API_KEY"  # en Render lo pondrás como variable de entorno
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ========================
# Base de datos SQLite
# ========================
DB_NAME = "chat_memory.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def save_message(session_id, role, content):
    cursor.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", 
                   (session_id, role, content))
    conn.commit()

def get_recent_history(session_id, n_turns=5):
    cursor.execute("""
        SELECT role, content FROM chat_history 
        WHERE session_id=? 
        ORDER BY id DESC LIMIT ?
    """, (session_id, n_turns*2))
    rows = cursor.fetchall()
    return list(reversed(rows))

# ========================
# ChromaDB
# ========================
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    document_mode = True
    @retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429,503})
    def __call__(self, input):
        task = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task),
        )
        return [e.values for e in response.embeddings]

embed_fn = GeminiEmbeddingFunction()
chroma_client = chromadb.Client()
knowledge_db = chroma_client.get_or_create_collection(
    name="googlecar_docs", embedding_function=embed_fn
)

documents = [
    "Operating the Climate Control System: Use the knobs to adjust temperature, airflow, fan speed, and modes (Auto, Cool, Heat, Defrost).",
    "Touchscreen Display: Access navigation, entertainment, and climate control by tapping icons.",
    "Shifting Gears: Park, Reverse, Neutral, Drive, Low (for slippery conditions)."
]
knowledge_db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# ========================
# Motor de conversación
# ========================
def build_prompt(session_id, user_query, retrieved_docs):
    history = get_recent_history(session_id, n_turns=5)
    prompt = "Eres un asistente de autos. Usa historial + docs para responder.\n\n"
    for role, content in history:
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += f"\nPregunta actual: {user_query}\n\n"
    for doc in retrieved_docs:
        prompt += f"- {doc}\n"
    return prompt

def conversational_rag(session_id, user_query):
    save_message(session_id, "user", user_query)
    embed_fn.document_mode = False
    result = knowledge_db.query(query_texts=[user_query], n_results=2)
    retrieved_docs = result["documents"][0] if result["documents"] else []
    prompt = build_prompt(session_id, user_query, retrieved_docs)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    answer = response.text
    save_message(session_id, "bot", answer)
    return answer

# ========================
# API FastAPI
# ========================
app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Chatbot RAG corriendo 🚀"}

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_user")
    answer = conversational_rag(session_id, user_message)
    return {"reply": answer}
