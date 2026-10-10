"""
test_rag.py — Local self-test script for verifying RAG functionalities
"""

import os
import sys

# Ensure current directory is in search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import rag_backend as rag
    print("[SUCCESS] Successfully imported rag_backend.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def test_kb_initialization():
    print("Testing Vector Database initialization...")
    try:
        vs = rag.get_vector_store()
        print(f"[SUCCESS] FAISS Vector Store successfully loaded. Type: {type(vs)}")
        
        # Test similarity search
        query = "What is HER2 positive?"
        results = vs.similarity_search(query, k=2)
        print(f"[SUCCESS] Similarity search returned {len(results)} chunks.")
        for i, r in enumerate(results):
            print(f"  Chunk {i+1} from {r.metadata.get('source')}: {r.page_content[:100]}...")
            
    except Exception as e:
        print(f"[ERROR] FAISS/Embeddings Initialization failed: {e}")
        import traceback
        traceback.print_exc()

def test_explanation():
    print("\nTesting prediction explanation...")
    features = {
        "radius_mean": 17.99,
        "texture_mean": 10.38,
        "perimeter_mean": 122.8,
        "area_mean": 1001.0,
        "smoothness_mean": 0.1184
    }
    try:
        explanation = rag.explain_prediction("MALIGNANT", 0.985, features)
        print("[SUCCESS] Explanation response generated successfully:")
        print(explanation[:300] + "...\n")
    except Exception as e:
        print(f"[ERROR] Explanation generation failed: {e}")

def test_chat():
    print("\nTesting chat QA RAG response...")
    msgs = [{"role": "user", "content": "What are the common symptoms of breast cancer?"}]
    try:
        reply = rag.chat_qa(msgs)
        print("[SUCCESS] Chat QA response generated successfully:")
        print(reply[:300] + "...\n")
    except Exception as e:
        print(f"[ERROR] Chat QA failed: {e}")

if __name__ == "__main__":
    test_kb_initialization()
    test_explanation()
    test_chat()
