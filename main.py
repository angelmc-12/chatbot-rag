from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import sqlite3
import chromadb
from google import genai
from google.genai import types
from google.api_core import retry
import os

# ========================
# Configuración de Gemini
# ========================
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
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", 
        (session_id, role, content)
    )
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
# ChromaDB con información de Techy
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
    name="techy_docs", embedding_function=embed_fn
)

# Documentos de ejemplo sobre Techy (puedes ampliarlos si quieres más contexto)
documents = [
    "Techy es una ONG peruana fundada por Angel, Brisa y Max enfocada en educación en ciencia de datos y tecnologías emergentes.",
    "Misión: democratizar el acceso al conocimiento en ciencia de datos y machine learning para jóvenes de sectores menos favorecidos.",
    "Visión: formar una generación de científicos de datos peruanos que aporten soluciones innovadoras a problemas locales y globales.",
    "Programas escolares: introducción a Python, fundamentos de machine learning, proyectos aplicados, retos gamificados.",
    "Programa universitario: malla curricular en Python, Numpy, Pandas, visualización y machine learning. Duración: 8 sesiones con reto final.",
    "Programa voluntarios: formación de formadores, inducción sobre Techy, creación de FAQs y guías.",
    "Áreas de gestión: innovación educativa, alianzas estratégicas, operaciones y finanzas, evaluación y seguimiento.",
    "Estrategias pedagógicas: gamificación, aprendizaje basado en proyectos, mentoría personalizada y capacitación docente.",
    "Desafíos: financiamiento sostenible, brecha digital, retención de estudiantes y construcción de credibilidad.",
    "Fortalezas: fundadores expertos en data science, claridad en misión y visión, enfoque innovador, red de contactos educativos y tecnológicos.",
    "Próximos pasos: finalizar malla curricular del piloto, alianzas con colegios y ONGs, ejecutar piloto universitario, elaborar reportes de impacto."
]
knowledge_db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# ========================
# Motor de conversación
# ========================
def build_prompt(session_id, user_query, retrieved_docs):
    history = get_recent_history(session_id, n_turns=5)
    prompt = "Eres un asistente especializado en Techy, una ONG peruana de educación en ciencia de datos. Usa el historial y los documentos para responder en español, con un tono claro e inspirador.\n\n"
    for role, content in history:
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += f"\nPregunta actual del usuario: {user_query}\n\n"
    if retrieved_docs:
        prompt += "Pasajes relevantes recuperados:\n"
        for doc in retrieved_docs:
            prompt += f"- {doc}\n"
    return prompt

def conversational_rag(session_id, user_query):
    save_message(session_id, "user", user_query)
    embed_fn.document_mode = False
    result = knowledge_db.query(query_texts=[user_query], n_results=3)
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
# API FastAPI (adaptado a Twilio)
# ========================
app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Chatbot Techy corriendo 🚀"}

@app.post("/webhook")
async def webhook(request: Request):
    form = await request.form()
    user_message = form.get("Body", "")
    session_id = form.get("From", "default_user")
    
    if not user_message:
        return PlainTextResponse("No recibí un mensaje válido 📭")
    
    answer = conversational_rag(session_id, user_message)
    return PlainTextResponse(answer)
