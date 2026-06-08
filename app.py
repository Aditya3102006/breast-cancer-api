"""
app.py — Streamlit frontend for AI Breast Cancer Assistant (RAG Enabled)
Provides a premium dark/light glassmorphic diagnostic dashboard.
"""

import os
import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any, List
from streamlit_folium import st_folium

# ─── Configuration ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Breast Cancer Assistant",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
APP_API_KEY = os.environ.get("APP_API_KEY", "breast-cancer-secret-key-2026")
headers = {"Authorization": f"Bearer {APP_API_KEY}"}

# ─── Custom CSS Injection ─────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }
            .main {
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: #f1f5f9;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 12px;
                background-color: rgba(255, 255, 255, 0.03);
                padding: 10px;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
            }
            .stTabs [data-baseweb="tab"] {
                height: 45px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 8px;
                color: #94a3b8;
                font-weight: 600;
                border: none;
                padding: 0 16px;
                transition: all 0.3s ease;
            }
            .stTabs [data-baseweb="tab"]:hover {
                color: #f472b6;
                background-color: rgba(244, 114, 182, 0.05);
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(90deg, #ec4899 0%, #a78bfa 100%) !important;
                color: white !important;
                box-shadow: 0 4px 15px rgba(236, 72, 153, 0.2);
            }
            .glass-card {
                background: rgba(30, 27, 75, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 24px;
                backdrop-filter: blur(16px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                margin-bottom: 20px;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 16px;
                text-align: center;
            }
            .gradient-text {
                background: linear-gradient(90deg, #f472b6 0%, #c084fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
            }
            div[data-testid="stMetricValue"] {
                font-size: 2rem;
                font-weight: 700;
                color: #f472b6;
            }
            .sidebar .sidebar-content {
                background-color: #0f172a;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# ─── Sidebar Branding ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 class='gradient-text'>Clinical Portal</h1>", unsafe_allow_html=True)
    st.markdown("🎗️ **AI Breast Cancer Assistant**")
    st.markdown("---")
    
    # Connection Health Check
    try:
        resp = requests.get(f"{API_BASE_URL}/", headers=headers)
        if resp.status_code == 200:
            st.success("🟢 API Connected")
        else:
            st.warning("🟡 API Status: Unhealthy")
    except Exception:
        st.error("🔴 API Offline. Ensure backend is running.")
        st.info("Run command: `uvicorn api:app --reload`")
        
    st.markdown("---")
    st.markdown(
        """
        ### About this Assistant
        This tool integrates:
        1. **ML Diagnostic Classifier**: Scikit-learn/XGBoost predictor trained on Wisconsion Breast Cancer dataset.
        2. **Retrieval-Augmented Generation**: LangChain and FAISS vector search, grounding the LLM in NCI & WHO clinical guidelines.
        """
    )
    st.caption("Educational tool. Consult a qualified oncologist for medical decisions.")

# ─── Navigation Tabs ──────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'><span class='gradient-text'>🎗️ Breast Cancer AI Assistant</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🩺 Diagnostic Predictor",
    "📄 Report Analyzer (Q&A)",
    "💬 AI Clinical Chatbot",
    "📚 Treatment Navigator",
    "📋 Doctor Prep Kit",
    "🏥 Nearby Oncology Finder"
])

# ─── TAB 1: Diagnostic Predictor ──────────────────────────────────────────────
with tab1:
    st.markdown("### Ductal Cell Nuclei Parameter Input")
    st.markdown("Enter the average cell nucleus measurements obtained from a Fine Needle Aspirate (FNA) biopsy.")
    
    # Main column groups
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Mean Measurements**")
        radius_mean = st.slider("Radius Mean (mm)", 6.0, 30.0, 14.1, 0.1, help="Mean of distances from center to points on the perimeter")
        texture_mean = st.slider("Texture Mean", 9.0, 40.0, 19.3, 0.1, help="Standard deviation of gray-scale values")
        perimeter_mean = st.slider("Perimeter Mean (mm)", 40.0, 200.0, 92.0, 0.5)
        area_mean = st.slider("Area Mean (mm²)", 100.0, 2500.0, 655.0, 5.0)
        smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.20, 0.10, 0.01, help="Local variation in radius lengths")
        
    with col2:
        st.markdown("**Cell Characteristic Means**")
        compactness_mean = st.slider("Compactness Mean", 0.02, 0.40, 0.10, 0.01, help="Perimeter² / Area - 1.0")
        concavity_mean = st.slider("Concavity Mean", 0.0, 0.50, 0.09, 0.01, help="Severity of concave portions of the contour")
        concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.25, 0.05, 0.01, help="Number of concave portions of the contour")
        symmetry_mean = st.slider("Symmetry Mean", 0.10, 0.35, 0.18, 0.01)
        fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.04, 0.12, 0.06, 0.01, help="Coastline approximation - 1.0")

    with col3:
        st.markdown("**Advanced Measurements**")
        with st.expander("🔬 View/Modify Standard Error (SE) & Worst Parameters"):
            st.caption("These are standard statistical measures of cell variations. They default to normal configurations matching your inputs.")
            
            radius_se = st.number_input("Radius SE", 0.1, 3.0, round(radius_mean * 0.03, 3))
            texture_se = st.number_input("Texture SE", 0.1, 5.0, round(texture_mean * 0.06, 3))
            perimeter_se = st.number_input("Perimeter SE", 0.5, 25.0, round(perimeter_mean * 0.03, 3))
            area_se = st.number_input("Area SE", 5.0, 500.0, round(area_mean * 0.06, 3))
            smoothness_se = st.number_input("Smoothness SE", 0.001, 0.03, 0.006, format="%.4f")
            compactness_se = st.number_input("Compactness SE", 0.001, 0.15, 0.025, format="%.4f")
            concavity_se = st.number_input("Concavity SE", 0.0, 0.3, 0.032, format="%.4f")
            concave_points_se = st.number_input("Concave Points SE", 0.0, 0.05, 0.012, format="%.4f")
            symmetry_se = st.number_input("Symmetry SE", 0.005, 0.1, 0.02, format="%.4f")
            fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.001, 0.03, 0.004, format="%.4f")
            
            radius_worst = st.number_input("Radius Worst", 7.0, 40.0, round(radius_mean * 1.2, 2))
            texture_worst = st.number_input("Texture Worst", 12.0, 50.0, round(texture_mean * 1.3, 2))
            perimeter_worst = st.number_input("Perimeter Worst", 50.0, 260.0, round(perimeter_mean * 1.25, 1))
            area_worst = st.number_input("Area Worst", 150.0, 4500.0, round(area_mean * 1.3, 1))
            smoothness_worst = st.number_input("Smoothness Worst", 0.05, 0.25, round(smoothness_mean * 1.3, 3))
            compactness_worst = st.number_input("Compactness Worst", 0.02, 1.2, round(compactness_mean * 2.5, 3))
            concavity_worst = st.number_input("Concavity Worst", 0.0, 1.3, round(concavity_mean * 2.5, 3))
            concave_points_worst = st.number_input("Concave Points Worst", 0.0, 0.35, round(concave_points_mean * 2.5, 3))
            symmetry_worst = st.number_input("Symmetry Worst", 0.15, 0.7, round(symmetry_mean * 1.5, 3))
            fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.05, 0.25, round(fractal_dimension_mean * 1.3, 3))

    st.markdown("---")
    if st.button("🔮 Analyze Pathology Features", type="primary", use_container_width=True):
        features = {
            "radius_mean": radius_mean, "texture_mean": texture_mean, "perimeter_mean": perimeter_mean,
            "area_mean": area_mean, "smoothness_mean": smoothness_mean, "compactness_mean": compactness_mean,
            "concavity_mean": concavity_mean, "concave points_mean": concave_points_mean,
            "symmetry_mean": symmetry_mean, "fractal_dimension_mean": fractal_dimension_mean,
            "radius_se": radius_se, "texture_se": texture_se, "perimeter_se": perimeter_se,
            "area_se": area_se, "smoothness_se": smoothness_se, "compactness_se": compactness_se,
            "concavity_se": concavity_se, "concave points_se": concave_points_se,
            "symmetry_se": symmetry_se, "fractal_dimension_se": fractal_dimension_se,
            "radius_worst": radius_worst, "texture_worst": texture_worst, "perimeter_worst": perimeter_worst,
            "area_worst": area_worst, "smoothness_worst": smoothness_worst, "compactness_worst": compactness_worst,
            "concavity_worst": concavity_worst, "concave points_worst": concave_points_worst,
            "symmetry_worst": symmetry_worst, "fractal_dimension_worst": fractal_dimension_worst
        }
        
        # 1. Trigger ML Prediction
        try:
            pred_resp = requests.post(f"{API_BASE_URL}/api/predict", json={"features": features}, headers=headers)
            if pred_resp.status_code == 200:
                data = pred_resp.json()
                label = data["label"]
                conf = data["confidence"]
                probs = data["probabilities"]
                
                # Store prediction in session state for tabs interaction
                st.session_state["last_prediction"] = {"label": label, "confidence": conf, "features": features}
                
                # Layout results
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                r_col1, r_col2 = st.columns(2)
                
                with r_col1:
                    st.markdown("#### Classifier Results")
                    if label == "MALIGNANT":
                        st.markdown(f"<h2 style='color:#ef4444;'>🎗️ MALIGNANT</h2>", unsafe_allow_html=True)
                        st.error("The ML model classifies these cell nuclei as Malignant (high correlation with cancer).")
                    else:
                        st.markdown(f"<h2 style='color:#10b981;'>🟢 BENIGN</h2>", unsafe_allow_html=True)
                        st.success("The ML model classifies these cell nuclei as Benign (non-cancerous).")
                        
                    st.metric(label="Model Confidence", value=f"{conf*100:.2f}%")
                    
                with r_col2:
                    st.markdown("#### Probability Distribution")
                    chart_data = pd.DataFrame({
                        "Class": ["Benign", "Malignant"],
                        "Probability": probs
                    })
                    st.bar_chart(chart_data, x="Class", y="Probability", color="#a78bfa")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 2. Trigger RAG Explanation
                with st.spinner("Retrieving literature and generating clinical explanation..."):
                    explain_resp = requests.post(
                        f"{API_BASE_URL}/api/rag/explain",
                        json={
                            "prediction_label": label,
                            "confidence": conf,
                            "features": {k: float(v) for k, v in features.items() if "mean" in k} # Send only core features for brevity
                        },
                        headers=headers
                    )
                    
                    if explain_resp.status_code == 200:
                        explanation = explain_resp.json()["explanation"]
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.markdown("### 📚 Clinical Explanation (RAG Grounded)")
                        st.markdown(explanation)
                        
                        # Set up redirect option
                        st.info("💡 You can export custom prep questions for your physician based on this analysis in the **Doctor Prep Kit** tab.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("RAG Explanation failed. Check API logs.")
            else:
                st.error(f"Prediction API Error: {pred_resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API server: {e}")

# ─── TAB 2: Report Analyzer (Q&A) ─────────────────────────────────────────────
with tab2:
    st.markdown("### Biopsy & Pathology Report Analyzer")
    st.markdown("Upload your biopsy or surgical pathology report (PDF or TXT) to extract key oncology markers and ask custom Q&A.")
    
    uploaded_file = st.file_uploader("Upload pathology report", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        
        # Trigger analyze API
        if "report_analyzed" not in st.session_state or st.session_state.get("report_filename") != filename:
            with st.spinner("Uploading and parsing pathology report..."):
                try:
                    files = {"file": (filename, file_bytes, "application/pdf" if filename.endswith("pdf") else "text/plain")}
                    rep_resp = requests.post(f"{API_BASE_URL}/api/rag/analyze_report", files=files, headers=headers)
                    
                    if rep_resp.status_code == 200:
                        data = rep_resp.json()
                        st.session_state["report_raw_text"] = data["raw_text"]
                        st.session_state["report_parsed_data"] = data["parsed_data"]
                        st.session_state["report_analyzed"] = True
                        st.session_state["report_filename"] = filename
                    else:
                        st.error(f"Parsing failed: {rep_resp.text}")
                except Exception as e:
                    st.error(f"Report upload API error: {e}")
                    
        if st.session_state.get("report_analyzed"):
            parsed = st.session_state["report_parsed_data"]
            raw_text = st.session_state["report_raw_text"]
            
            st.success(f"Successfully processed: {filename}")
            
            # Displays extracted markers
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 📋 Extracted Pathology Markers")
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Estrogen Receptor (ER)", parsed.get("er_status", "Unknown"))
                st.markdown("</div>", unsafe_allow_html=True)
            with m_col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Progesterone Receptor (PR)", parsed.get("pr_status", "Unknown"))
                st.markdown("</div>", unsafe_allow_html=True)
            with m_col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("HER2 Status", parsed.get("her2_status", "Unknown"))
                st.markdown("</div>", unsafe_allow_html=True)
            with m_col4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Tumor Grade", parsed.get("tumor_grade", "Unknown"))
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            st_col1, st_col2 = st.columns(2)
            with st_col1:
                st.markdown("**Tumor Type / Histology**")
                st.info(parsed.get("tumor_type", "Unknown"))
            with st_col2:
                st.markdown("**Margin Status**")
                st.info(parsed.get("margin_status", "Unknown"))
                
            st.markdown("**Biopsy Findings Summary**")
            st.write(parsed.get("summary", "No summary generated."))
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Sub-Q&A under the report
            st.markdown("### 💬 Ask Questions About This Report")
            st.caption("Ask specific questions like 'What does ER positive mean for my treatment?' or 'Are my margins clean?'")
            
            # Chat history for report QA
            if "report_chat_history" not in st.session_state:
                st.session_state["report_chat_history"] = []
                
            for m in st.session_state["report_chat_history"]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    
            if report_query := st.chat_input("Ask about your report...", key="report_chat_input"):
                with st.chat_message("user"):
                    st.markdown(report_query)
                st.session_state["report_chat_history"].append({"role": "user", "content": report_query})
                
                with st.chat_message("assistant"):
                    with st.spinner("Searching medical guidelines and analyzing report..."):
                        try:
                            messages_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state["report_chat_history"]]
                            chat_resp = requests.post(
                                f"{API_BASE_URL}/api/rag/chat",
                                json={
                                    "messages": messages_payload,
                                    "report_text": raw_text
                                },
                                headers=headers
                            )
                            if chat_resp.status_code == 200:
                                reply = chat_resp.json()["reply"]
                                st.markdown(reply)
                                st.session_state["report_chat_history"].append({"role": "ai", "content": reply})
                            else:
                                st.error("QA Chat failed.")
                        except Exception as e:
                            st.error(f"Chat API error: {e}")

# ─── TAB 3: AI Clinical Chatbot ───────────────────────────────────────────────
with tab3:
    st.markdown("### AI Clinical Assistant Chatroom")
    st.markdown("Discuss diagnosis, guidelines, and terminology. Grounded in the clinical knowledge base (NCI/WHO).")
    
    if "global_chat_history" not in st.session_state:
        st.session_state["global_chat_history"] = [
            {"role": "ai", "content": "Hello, I am your clinical AI assistant. I can help answer questions about breast cancer diagnosis, pathology findings, treatments, and medical terminology. How can I help you today?"}
        ]
        
    for m in st.session_state["global_chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
    if global_query := st.chat_input("Ask a medical question...", key="global_chat_input"):
        with st.chat_message("user"):
            st.markdown(global_query)
        st.session_state["global_chat_history"].append({"role": "user", "content": global_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting knowledge base..."):
                try:
                    messages_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state["global_chat_history"]]
                    chat_resp = requests.post(
                        f"{API_BASE_URL}/api/rag/chat",
                        json={
                            "messages": messages_payload,
                            "report_text": st.session_state.get("report_raw_text") # Include report if uploaded
                        },
                        headers=headers
                    )
                    if chat_resp.status_code == 200:
                        reply = chat_resp.json()["reply"]
                        st.markdown(reply)
                        st.session_state["global_chat_history"].append({"role": "ai", "content": reply})
                    else:
                        st.error("Chat failed.")
                except Exception as e:
                    st.error(f"Chat API error: {e}")

# ─── TAB 4: Treatment Navigator ───────────────────────────────────────────────
with tab4:
    st.markdown("### Breast Cancer Treatment Navigator")
    st.markdown("Select a treatment pathway to learn about common clinical guidelines, procedures, and side effects.")
    
    treatment_tabs = st.tabs(["🪓 Surgery", "🧪 Chemotherapy", "⚡ Radiation Therapy", "💊 Hormone Therapy"])
    
    # Simple Q&A function helper for treatments
    def get_treatment_info(query_str: str) -> str:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/rag/chat",
                json={
                    "messages": [{"role": "user", "content": query_str}],
                    "report_text": None
                },
                headers=headers
            )
            if resp.status_code == 200:
                return resp.json()["reply"]
            return "Failed to retrieve treatment information."
        except Exception as e:
            return f"Error: {e}"
            
    with treatment_tabs[0]:
        st.markdown("#### Surgical Options")
        st.markdown(
            "Surgery is the standard local treatment for non-metastatic breast cancer to remove the primary tumor. "
            "Options include Lumpectomy (preserving the breast) or Mastectomy (removing the entire breast tissue)."
        )
        if st.button("Retrieve Surgical Guidelines & Questions"):
            with st.spinner("Searching surgery vectors..."):
                info = get_treatment_info("Explain breast cancer surgeries (lumpectomy, mastectomy, sentinel node biopsy) and risks.")
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(info)
                st.markdown("</div>", unsafe_allow_html=True)
                
    with treatment_tabs[1]:
        st.markdown("#### Chemotherapy Protocols")
        st.markdown(
            "Chemotherapy uses drugs to target rapidly growing cancer cells. It can be given before surgery (neoadjuvant) "
            "to shrink the tumor, or after surgery (adjuvant) to kill microscopic cancer cells and reduce recurrence."
        )
        if st.button("Retrieve Chemotherapy Guidelines"):
            with st.spinner("Searching chemo vectors..."):
                info = get_treatment_info("Explain breast cancer chemotherapy (neoadjuvant vs adjuvant, common drugs like Taxol/Adriamycin, side effects).")
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(info)
                st.markdown("</div>", unsafe_allow_html=True)
                
    with treatment_tabs[2]:
        st.markdown("#### Radiotherapy")
        st.markdown(
            "Radiation therapy uses high-energy rays to target the breast and lymph nodes to destroy remaining cancer cells. "
            "It is highly recommended after a lumpectomy to reduce local recurrence risks."
        )
        if st.button("Retrieve Radiation Therapy Guidelines"):
            with st.spinner("Searching radiation vectors..."):
                info = get_treatment_info("Explain breast cancer radiation therapy, schedules, side effects, and when it is needed.")
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(info)
                st.markdown("</div>", unsafe_allow_html=True)
                
    with treatment_tabs[3]:
        st.markdown("#### Endocrine / Hormone Therapy")
        st.markdown(
            "For Hormone Receptor-positive (ER+ and/or PR+) cancers. It blocks estrogen or progesterone receptors "
            "on the cancer cells to starve the cancer of hormones, using SERMs (Tamoxifen) or Aromatase Inhibitors (AIs)."
        )
        if st.button("Retrieve Hormone Therapy Guidelines"):
            with st.spinner("Searching hormone vectors..."):
                info = get_treatment_info("Explain breast cancer hormone therapy (Tamoxifen vs Aromatase Inhibitors like Anastrozole/Letrozole, side effects).")
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(info)
                st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 5: Doctor Prep Kit ───────────────────────────────────────────────────
with tab5:
    st.markdown("### Customized Doctor Consultation Preparation Kit")
    st.markdown("Generate a structured checklist and tailored questions to print or save for your next medical appointment.")
    
    # Auto fill buttons if prediction/report exists
    has_report = "report_raw_text" in st.session_state
    has_pred = "last_prediction" in st.session_state
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Active Clinical Context")
    if has_pred:
        label = st.session_state["last_prediction"]["label"]
        conf = st.session_state["last_prediction"]["confidence"]
        st.write(f"✔️ Linked ML Prediction: **{label}** (Confidence: {conf*100:.1f}%)")
    else:
        st.write("❌ No active ML prediction. Go to Tab 1 to run an analysis.")
        
    if has_report:
        st.write(f"✔️ Linked pathology report: **{st.session_state['report_filename']}**")
    else:
        st.write("❌ No biopsy report uploaded. Go to Tab 2 to upload one.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("📋 Generate My Custom Consultation Prep Kit", type="primary", use_container_width=True):
        with st.spinner("Compiling patient details and formatting question guides..."):
            try:
                payload = {
                    "report_text": st.session_state.get("report_raw_text") if has_report else None,
                    "prediction_label": st.session_state["last_prediction"]["label"] if has_pred else None,
                    "confidence": st.session_state["last_prediction"]["confidence"] if has_pred else None
                }
                
                prep_resp = requests.post(f"{API_BASE_URL}/api/rag/doctor_prep", json=payload, headers=headers)
                if prep_resp.status_code == 200:
                    kit = prep_resp.json()["prep_kit"]
                    st.session_state["doctor_prep_kit"] = kit
                else:
                    st.error("Failed to generate preparation kit.")
            except Exception as e:
                st.error(f"Doctor prep API error: {e}")
                
    if "doctor_prep_kit" in st.session_state:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Your Custom Physician Preparation Guide")
        st.markdown(st.session_state["doctor_prep_kit"])
        
        # Download guide as markdown
        st.download_button(
            label="📥 Download Guide as Markdown (TXT)",
            data=st.session_state["doctor_prep_kit"],
            file_name="Breast_Cancer_Doctor_Prep_Kit.md",
            mime="text/markdown"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 6: Nearby Medical Facilities ─────────────────────────────────────────
with tab6:
    st.markdown("### Locate Specialized Oncology Centers & Hospitals")
    st.markdown("Search nearby hospitals, medical colleges, and cancer centers using OpenStreetMap geolocation details.")
    
    # Setup inputs
    map_col1, map_col2 = st.columns([1, 3])
    
    with map_col1:
        search_query = st.text_input("Enter Address / City / Location", "New Delhi, India", help="E.g., New Delhi, Mumbai, Boston, etc.")
        radius = st.slider("Search Radius (km)", 2, 30, 10)
        
        facility_type = st.selectbox(
            "Facility Specialization",
            ["All", "Cancer Center / Oncology", "Hospital", "Clinic"]
        )
        
        ownership = st.selectbox(
            "Hospital Type / Panel",
            ["All", "Government Only", "Private Only", "Panel / Empanelled"]
        )
        
        search_map_btn = st.button("🔍 Find Nearby Facilities", use_container_width=True)
        
    with map_col2:
        if search_map_btn or "last_map_query" not in st.session_state:
            st.session_state["last_map_query"] = search_query
            with st.spinner("Geocoding location and retrieving medical facilities..."):
                try:
                    # 1. Geocode
                    geo_resp = requests.get(f"{API_BASE_URL}/api/geocode", params={"q": search_query}, headers=headers)
                    if geo_resp.status_code == 200:
                        geo_data = geo_resp.json()
                        lat, lng = geo_data["lat"], geo_data["lng"]
                        display_name = geo_data["display_name"]
                        
                        st.markdown(f"**Location Found:** {display_name}")
                        st.markdown(f"Coordinates: `{lat}, {lng}`")
                        
                        # 2. Fetch hospitals
                        hosp_params = {
                            "lat": lat,
                            "lng": lng,
                            "radius_km": radius,
                            "filter_type": facility_type,
                            "filter_ownership": ownership
                        }
                        hosp_resp = requests.get(f"{API_BASE_URL}/api/hospitals", params=hosp_params, headers=headers)
                        
                        if hosp_resp.status_code == 200:
                            hosp_data = hosp_resp.json()
                            results = hosp_data.get("results", [])
                            count = hosp_data.get("count", 0)
                            
                            st.write(f"Found **{count}** facilities within {radius} km.")
                            
                            # 3. Build map
                            import hospital_backend as hb
                            fmap = hb.build_map(lat, lng, results, radius)
                            st_folium(fmap, width=900, height=500)
                            
                            # Show hospital details in a table underneath
                            if results:
                                st.markdown("#### Details of Found Facilities")
                                detail_list = []
                                for r in results:
                                    govt_status = "Government" if r["is_govt"] else "Private"
                                    panels = ", ".join(r["panels"]) if r["panels"] else "None"
                                    detail_list.append({
                                        "Name": r["name"],
                                        "Grade / Tier": r["tier"],
                                        "Ownership": govt_status,
                                        "Distance (km)": r["distance_km"],
                                        "Address": r["address"],
                                        "Panels": panels,
                                        "Phone": r["phone"]
                                    })
                                st.dataframe(pd.DataFrame(detail_list), use_container_width=True)
                        else:
                            st.error(f"Hospitals search failed: {hosp_resp.text}")
                    else:
                        st.error("Location not found. Please try a different query.")
                except Exception as e:
                    st.error(f"Facility finder error: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            # Render from session if exists
            st.info("Modify inputs on the left and click 'Find Nearby Facilities' to load results.")
