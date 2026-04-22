"""
api.py — FastAPI backend for Breast Cancer Diagnostic AI
Exposes:
  POST /api/predict       → ML model prediction
  GET  /api/hospitals     → Hospital search via OSM
  GET  /api/geocode       → Location geocoding
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, pandas as pd
from typing import Optional, List
import hospital_backend as hb
import google.generativeai as genai
from pydantic import BaseModel

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Breast Cancer Diagnostic AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Gemini AI Setup ──────────────────────────────────────────────────────────
# Using gemini-1.5-flash
GEMINI_KEY = "AIzaSyC4pedn_3FdEDGOclYbD9E-g1bvAGyZkdc"
genai.configure(api_key=GEMINI_KEY)

SYSTEM_PROMPT = """You are a compassionate, expert clinical AI assistant specializing in breast cancer awareness, diagnosis, and treatment. 
You help patients understand medical terms, symptoms, and treatment options in simple language. 
Always remind users that you are an AI and they should consult a qualified oncologist for medical decisions. 
Keep responses concise, empathetic, and easy to understand."""

# Create model with system instruction
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=SYSTEM_PROMPT
)

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


@app.post("/api/predict", response_model=PredictResponse)
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


@app.get("/api/geocode")
def geocode(q: str):
    """Geocode a location string to lat/lng."""
    result = hb.geocode_location(q)
    if not result or "error" in result:
        raise HTTPException(status_code=404, detail="Location not found")
    return result


@app.get("/api/hospitals")
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Proxy chat requests to Gemini with clinical context."""
    try:
        # Convert local message format to Gemini format
        # Gemini history MUST alternating and MUST start with 'user'
        history = []
        for m in req.messages[:-1]:
            # Skip the initial AI greeting if it's the first message
            if not history and m.role == "ai":
                continue
            
            history.append({
                "role": "user" if m.role == "user" else "model",
                "parts": [m.content]
            })

        user_msg = req.messages[-1].content
        
        # Start a chat session
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_msg)
        
        return ChatResponse(reply=response.text)
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Show error in console
        error_str = str(e)
        if "429" in error_str:
            raise HTTPException(status_code=429, detail="AI Service is currently busy. Please try again in a minute.")
        raise HTTPException(status_code=500, detail=error_str)
