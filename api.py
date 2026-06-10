"""
api.py — FastAPI backend for Breast Cancer Diagnostic AI
Exposes:
  POST /api/predict       → ML model prediction
  GET  /api/hospitals     → Hospital search via OSM
  GET  /api/geocode       → Location geocoding
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os, pickle, pandas as pd
from typing import Optional, List
import hospital_backend as hb
# google.generativeai is replaced by langchain-groq

import rag_backend as rag

# ─── Security Setup ───────────────────────────────────────────────────────────
security = HTTPBearer()
APP_API_KEY = os.environ.get("APP_API_KEY", "breast-cancer-secret-key-2026")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies client Authorization Bearer Token against APP_API_KEY."""
    if credentials.credentials != APP_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Please check your credentials."
        )

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Breast Cancer Diagnostic AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Groq AI Setup ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a compassionate, expert clinical AI assistant specializing in breast cancer awareness, diagnosis, and treatment. 
You help patients understand medical terms, symptoms, and treatment options in simple language. 
Always remind users that you are an AI and they should consult a qualified oncologist for medical decisions. 
Keep responses concise, empathetic, and easy to understand."""


# ─── Load ML Model ────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict  # { feature_name: float, ... }

class PredictResponse(BaseModel):
    prediction: int          # 0 = Benign, 1 = Malignant
    label: str               # "BENIGN" or "MALIGNANT"
    confidence: float        # probability of the predicted class
    probabilities: List[float]  # [prob_benign, prob_malignant]

class ChatMessage(BaseModel):
    role: str       # "user" or "ai"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "api": "Breast Cancer Diagnostic AI v2.0"}


@app.post("/api/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(req: PredictRequest):
    """Run the ML model and return prediction + probabilities."""
    try:
        row = {f: req.features.get(f, 0.0) for f in FEATURES}
        df = pd.DataFrame([row])
        pred = int(MODEL.predict(df)[0])
        proba = MODEL.predict_proba(df)[0].tolist()
        return PredictResponse(
            prediction=pred,
            label="MALIGNANT" if pred == 1 else "BENIGN",
            confidence=round(proba[pred], 4),
            probabilities=[round(p, 4) for p in proba],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/geocode", dependencies=[Depends(verify_api_key)])
def geocode(q: str):
    """Geocode a location string to lat/lng."""
    result = hb.geocode_location(q)
    if not result or "error" in result:
        raise HTTPException(status_code=404, detail="Location not found")
    return result


@app.get("/api/hospitals", dependencies=[Depends(verify_api_key)])
def hospitals(
    lat: float,
    lng: float,
    radius_km: int = 10,
    filter_type: str = "All",
    filter_ownership: str = "All",
    max_results: int = 40,
):
    """Search for medical facilities near lat/lng."""
    result = hb.search_medical_facilities(
        lat, lng, radius_km,
        filter_type=filter_type,
        filter_ownership=filter_ownership,
        max_results=max_results,
    )
    if not result.get("success"):
        raise HTTPException(status_code=503, detail=result.get("error", "Search failed"))
    return result


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest):
    """Proxy chat requests to Groq with clinical context."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        from langchain_groq import ChatGroq
        
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for m in req.messages:
            if m.role == "user":
                lc_messages.append(HumanMessage(content=m.content))
            else:
                lc_messages.append(AIMessage(content=m.content))
                
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY,
            temperature=0.3
        )
        response = groq_llm.invoke(lc_messages)
        return ChatResponse(reply=response.content)
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Show error in console
        raise HTTPException(status_code=500, detail=str(e))



# ─── RAG Schemas ──────────────────────────────────────────────────────────────
class ExplainRequest(BaseModel):
    prediction_label: str
    confidence: float
    features: dict

class RAGChatRequest(BaseModel):
    messages: List[ChatMessage]
    report_text: Optional[str] = None

class DoctorPrepRequest(BaseModel):
    report_text: Optional[str] = None
    prediction_label: Optional[str] = None
    confidence: Optional[float] = None


# ─── RAG Endpoints ────────────────────────────────────────────────────────────
@app.post("/api/rag/explain", dependencies=[Depends(verify_api_key)])
def explain(req: ExplainRequest):
    """Explain ML model prediction using clinical guidelines (RAG)."""
    try:
        explanation = rag.explain_prediction(
            req.prediction_label, 
            req.confidence, 
            req.features
        )
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/analyze_report", dependencies=[Depends(verify_api_key)])
async def analyze_report(file: UploadFile = File(...)):
    """Upload pathology report (PDF or TXT) and extract clinical markers."""
    try:
        content = await file.read()
        text = rag.extract_text_from_upload(content, file.filename)
        if not text or text.startswith("Failed") or "Unsupported" in text:
            raise HTTPException(status_code=400, detail=text or "Empty report content.")
            
        parsed_data = rag.extract_pathology_data(text)
        return {
            "success": True,
            "raw_text": text,
            "parsed_data": parsed_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/chat", dependencies=[Depends(verify_api_key)])
def rag_chat(req: RAGChatRequest):
    """Chat with AI breast cancer assistant grounded in medical knowledge & report."""
    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        reply = rag.chat_qa(msgs, report_text=req.report_text)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/doctor_prep", dependencies=[Depends(verify_api_key)])
def doctor_prep(req: DoctorPrepRequest):
    """Generate customized consultation preparation checklist & question guide."""
    try:
        prep_kit = rag.generate_doctor_prep_kit(
            report_text=req.report_text,
            prediction_label=req.prediction_label,
            confidence=req.confidence
        )
        return {"prep_kit": prep_kit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
