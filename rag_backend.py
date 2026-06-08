"""
rag_backend.py — RAG services for Breast Cancer Diagnostic Assistant
Includes:
  - Vector Store indexing and retrieval for trusted knowledge base
  - pathology/biopsy report parser (marker extraction)
  - ML prediction explainer grounding in literature
  - RAG chat grounded in knowledge base + patient report
  - Doctor Consultation Prep Kit generator
"""

import os
import glob
import json
from typing import List, Dict, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─── Configuration ───────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_KEY", "AQ.Ab8RN6JqiFwGyIbwD5AWx8w0pf9Bn9lqHzgy6UcbYBcihcdcKQ"))

# Global variables for caching
_VECTOR_STORE = None

def get_vector_store() -> FAISS:
    """Lazily load or rebuild the FAISS vector database from knowledge base documents."""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=GEMINI_KEY
    )
    
    index_path = "faiss_index"
    
    # Try to load existing local FAISS store
    if os.path.exists(index_path):
        try:
            _VECTOR_STORE = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return _VECTOR_STORE
        except Exception as e:
            print(f"Error loading local FAISS store: {e}. Rebuilding...")

    # Rebuild from scratch
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)
        
    docs = []
    from langchain_core.documents import Document
    
    # Read all markdown files in knowledge_base directory
    for fn in os.listdir(kb_dir):
        if fn.endswith(".md"):
            path = os.path.join(kb_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append(Document(page_content=content, metadata={"source": fn}))
            except Exception as e:
                print(f"Error reading {fn}: {e}")
                
    if not docs:
        # Fallback default clinical document
        fallback_text = (
            "# Breast Cancer Basics\n"
            "Breast cancer starts when cells in the breast begin to grow out of control. "
            "These cells usually form a tumor that can often be seen on an x-ray or felt as a lump. "
            "Receptors such as Estrogen Receptor (ER), Progesterone Receptor (PR), and HER2 "
            "help classify breast cancers and determine the target treatments (e.g. endocrine therapy, chemotherapy)."
        )
        docs.append(Document(page_content=fallback_text, metadata={"source": "default.md"}))
        
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    # Generate vector store and save it locally
    _VECTOR_STORE = FAISS.from_documents(chunks, embeddings)
    _VECTOR_STORE.save_local(index_path)
    return _VECTOR_STORE

# ─── Pathology Report Text Extraction ─────────────────────────────────────────
def extract_text_from_upload(file_content: bytes, filename: str) -> str:
    """Extract text from raw file bytes based on file extension."""
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        import io
        from pypdf import PdfReader
        try:
            pdf = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text.strip()
        except Exception as e:
            return f"Failed to parse PDF report: {str(e)}"
    elif ext in ("txt", "md", "markdown"):
        try:
            return file_content.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            return f"Failed to decode text report: {str(e)}"
    else:
        # Fallback raw conversion
        try:
            return file_content.decode("utf-8", errors="ignore").strip()
        except Exception:
            return "Unsupported file type. Please upload a PDF or TXT report."

# ─── Pathology Report Parsing ─────────────────────────────────────────────────
def extract_pathology_data(report_text: str) -> Dict[str, Any]:
    """Analyze the report text and extract key markers using Gemini structured prompt."""
    system_prompt = (
        "You are an expert pathological AI parser. Analyze the breast cancer biopsy/pathology report "
        "and extract key medical details in JSON format. Do not guess; if a detail is not mentioned, "
        "use 'Not mentioned' or 'Unknown'. Return EXACTLY a JSON object matching this schema:\n"
        "{\n"
        "  \"er_status\": \"Positive / Negative / Unknown\",\n"
        "  \"pr_status\": \"Positive / Negative / Unknown\",\n"
        "  \"her2_status\": \"Positive / Negative / Borderline (Equivocal) / Unknown\",\n"
        "  \"tumor_grade\": \"Grade 1 / Grade 2 / Grade 3 / Unknown\",\n"
        "  \"margin_status\": \"Clear (Negative) / Positive / Close / Unknown\",\n"
        "  \"tumor_type\": \"e.g., Invasive Ductal Carcinoma (IDC), DCIS, Ductal, Lobular, etc.\",\n"
        "  \"summary\": \"A concise 2-3 sentence summary of the report in simple terms for a patient.\"\n"
        "}"
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_KEY,
        temperature=0.0,
        model_kwargs={"response_mime_type": "application/json"}
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Report Content:\n\n{report_text}")
        ])
        return json.loads(response.content)
    except Exception as e:
        return {
            "er_status": "Unknown",
            "pr_status": "Unknown",
            "her2_status": "Unknown",
            "tumor_grade": "Unknown",
            "margin_status": "Unknown",
            "tumor_type": "Unknown",
            "summary": "Could not parse report structure automatically. Please read the report manually.",
            "error": str(e)
        }

# ─── Prediction Explainer ─────────────────────────────────────────────────────
def explain_prediction(prediction_label: str, confidence: float, features: Dict[str, float]) -> str:
    """Retrieve relevant guidelines and generate an explanation for an ML prediction."""
    try:
        vector_store = get_vector_store()
        
        # Retrieve relevant clinical guidelines
        query = f"Breast cancer diagnosis {prediction_label} explanation next steps"
        docs = vector_store.similarity_search(query, k=3)
        context = "\n---\n".join(d.page_content for d in docs)
        
        # Format features for better readability
        features_str = ", ".join(f"{k}: {v}" for k, v in features.items())
        
        system_prompt = (
            "You are an expert clinical AI oncologist assistant. Explain the breast cancer prediction "
            "to a patient in a highly compassionate, clear, and educational manner. "
            "Ensure you remind the patient that this is a machine learning prediction and not a formal medical diagnosis, "
            "and that they must consult their doctor.\n\n"
            "Context from trusted clinical sources:\n"
            f"{context}"
        )
        
        user_prompt = (
            f"The machine learning model predicted: {prediction_label}\n"
            f"Confidence level: {confidence * 100:.2f}%\n"
            f"Patient tumor features analyzed: {features_str}\n\n"
            "Based on this, write a detailed explanation addressing:\n"
            "1. What this prediction means (benign vs malignant explanation).\n"
            "2. Key risk factors and common symptoms related to this type of lesion.\n"
            "3. Recommended next diagnostic steps (e.g., biopsy if not done, specialist consultation, follow-up imaging).\n"
            "4. Supportive, empathetic closing."
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_KEY,
            temperature=0.2
        )
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# ─── RAG Chat Q&A ─────────────────────────────────────────────────────────────
def chat_qa(messages: List[Dict[str, str]], report_text: Optional[str] = None) -> str:
    """Run a multi-turn RAG chat conversation grounded in the clinical knowledge base."""
    try:
        vector_store = get_vector_store()
        
        # Find latest user message to search vector DB
        latest_user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_query = m.get("content", "")
                break
                
        if not latest_user_query:
            latest_user_query = "breast cancer awareness and guidelines"
            
        # Similarity search in vector DB
        docs = vector_store.similarity_search(latest_user_query, k=3)
        kb_context = "\n---\n".join(d.page_content for d in docs)
        
        system_prompt = (
            "You are a compassionate, expert clinical AI assistant specializing in breast cancer awareness, "
            "diagnosis, and treatment. You help patients understand medical terms, symptoms, and treatment options in simple language.\n"
            "Always remind users that you are an AI and they should consult a qualified oncologist for medical decisions.\n"
            "Keep responses concise, empathetic, and easy to understand.\n\n"
            "Grounded medical knowledge base:\n"
            f"{kb_context}\n"
        )
        
        if report_text:
            system_prompt += (
                "\nThe patient has uploaded their pathology report. Refer to it as appropriate when answering questions. "
                "Never share patient data with unauthorized entities. Keep explanations of their markers (HER2, ER/PR, Grade) simple:\n"
                f"--- Patient Report ---\n{report_text}\n----------------------\n"
            )
            
        # Format message history for LangChain
        lc_messages = [SystemMessage(content=system_prompt)]
        
        # Include history up to last 10 messages to save context token usage
        for m in messages[-10:]:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role in ("ai", "model", "assistant"):
                lc_messages.append(AIMessage(content=content))
                
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_KEY,
            temperature=0.3
        )
        
        response = llm.invoke(lc_messages)
        return response.content
    except Exception as e:
        return f"AI Service is currently busy. Please try again. (Detail: {str(e)})"

# ─── Doctor Visit Prep Kit Generator ──────────────────────────────────────────
def generate_doctor_prep_kit(report_text: Optional[str] = None, prediction_label: Optional[str] = None, confidence: Optional[float] = None) -> str:
    """Generate a list of questions to ask the doctor based on prediction or report."""
    try:
        vector_store = get_vector_store()
        
        # Retrieve question bank guidelines
        docs = vector_store.similarity_search("questions to ask surgeon oncologist pathologist", k=2)
        context = "\n---\n".join(d.page_content for d in docs)
        
        system_prompt = (
            "You are a clinical navigator assistant. Generate a highly personalized 'Doctor Visit Prep Kit' "
            "for a breast cancer patient. Ground the questions in their specific clinical context.\n\n"
            "Context from guidelines:\n"
            f"{context}"
        )
        
        user_prompt = "Generate a list of questions for my doctors. Here is my current clinical information:\n"
        if prediction_label:
            user_prompt += f"- Machine learning prediction: {prediction_label} (Confidence: {confidence * 100:.1f}%)\n"
        if report_text:
            user_prompt += f"- Extract of my biopsy report:\n{report_text[:2000]}\n"
            
        user_prompt += (
            "\nCreate a structured guide with sections:\n"
            "1. Questions for the Surgeon (about biopsy, lumpectomy, mastectomy, reconstruction, lymph nodes)\n"
            "2. Questions for the Medical Oncologist (about chemotherapy, hormone therapy, targeted therapy, immunotherapy)\n"
            "3. Questions for the Radiation Oncologist (about radiation schedules, local side effects)\n"
            "4. Practical checklist of things to bring to the appointment (scans, reports, notebook, support person)."
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_KEY,
            temperature=0.3
        )
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content
    except Exception as e:
        return f"Error generating doctor preparation kit: {str(e)}"
