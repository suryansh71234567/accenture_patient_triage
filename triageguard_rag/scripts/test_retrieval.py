import sys
sys.path.insert(0, r'c:\coding vs\aic_hackathon')

from triageguard_rag.src.embeddings.embedder import Embedder
from triageguard_rag.src.retrieval.retriever import Retriever
from pathlib import Path

embedder = Embedder('sentence-transformers/all-MiniLM-L6-v2')
vs_dir = Path(r'c:\coding vs\aic_hackathon\triageguard_rag\data\vector_store')
retriever = Retriever(vs_dir, embedder)

hist, sim = retriever.retrieve(
    query_text='chest pain shortness of breath HR 112 SpO2 94 acuity 2',
    patient_id=10016742,
    top_k_self=3,
    top_k_similar=5
)

print(f'Patient history docs : {len(hist)}')
print(f'Similar case docs    : {len(sim)}')
if sim:
    print('\nTop similar case preview:')
    print(sim[0]["document_text"][:500])

# Also test the prompt builder
from triageguard_rag.src.reasoning.llm_reasoner import build_prompt
patient_state = {
    "patient_id": 10016742,
    "chiefcomplaint": "chest pain and shortness of breath",
    "acuity": 2,
    "heartrate": 112,
    "resprate": 22,
    "o2sat": 94,
    "sbp": 148,
    "dbp": 90,
    "temperature": 98.9,
    "pain": 7
}
prompt = build_prompt(patient_state, hist, sim)
print('\n--- PROMPT PREVIEW (first 800 chars) ---')
print(prompt[:800])
