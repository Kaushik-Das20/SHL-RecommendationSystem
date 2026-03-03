
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import os
from typing import List, Optional

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Given a job description or query, returns relevant SHL assessments",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

GEMINI_API_KEY = "AIzaSyAVoa3xQz3pbGI-0eSpYSvUOKUxgruTt0U" 

DATA_PATH = "data/shl_assessments_clean.csv"
INDEX_PATH = "data/shl_index.faiss"

print("Loading data and models...")

df = pd.read_csv(DATA_PATH)
index = faiss.read_index(INDEX_PATH)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash-lite")

print(f"✅ Loaded {len(df)} assessments")
print(f"✅ FAISS index ready with {index.ntotal} vectors")
print(f"✅ Gemini AI ready")

class RecommendRequest(BaseModel):
    """What the API receives from user"""
    query: str                          
    top_k: Optional[int] = 10          

class Assessment(BaseModel):
    """Single assessment recommendation"""
    assessment_name: str
    url: str
    test_type: str
    remote_testing: str
    adaptive_irt: str
    description: str

class RecommendResponse(BaseModel):
    """What the API sends back"""
    query: str
    total_results: int
    recommended_assessments: List[Assessment]



def extract_text_from_url(url: str) -> str:
    """If user gives a URL, fetch and extract text from it"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        text = soup.get_text(separator=" ", strip=True)
        return text[:2000]
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {str(e)}")


def understand_query(query_text: str) -> dict:
    """Use Gemini to understand the query and extract key info"""
    
    prompt = f"""
    You are an expert HR assessment consultant.
    
    Analyze this job description or query and extract:
    1. Job Role
    2. Technical Skills needed
    3. Soft Skills needed
    4. Assessment Types needed (A=Ability, B=Biodata, C=Competencies, 
       K=Knowledge/Skills, P=Personality, S=Simulations)
    5. A refined search query to find relevant SHL assessments
    
    Query: {query_text}
    
    Respond in this exact format:
    JOB_ROLE: <job role>
    TECHNICAL_SKILLS: <comma separated>
    SOFT_SKILLS: <comma separated>
    ASSESSMENT_TYPES: <comma separated letters>
    SEARCH_QUERY: <refined query>
    """
    
    response = llm.generate_content(prompt)
    result = response.text
    
    parsed = {}
    for line in result.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    
    return parsed


def search_assessments(search_query: str, top_k: int = 20) -> list:
    """Search FAISS index for similar assessments"""
    

    query_embedding = embedding_model.encode([search_query]).astype("float32")
    
    distances, indices = index.search(query_embedding, top_k)
    
    candidates = []
    for i, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        candidates.append({
            "name": str(row["name"]),
            "url": str(row["url"]),
            "test_type": str(row["test_type"]),
            "remote_testing": str(row["remote_testing"]),
            "adaptive_irt": str(row["adaptive_irt"]),
            "description": str(row.get("description", ""))[:200],
            "similarity_score": float(distances[0][i])
        })
    
    return candidates


def rerank_with_gemini(query: str, candidates: list) -> list:
    """Use Gemini to re-rank candidates and pick best 5-10"""
    
    candidates_text = ""
    for i, c in enumerate(candidates):
        candidates_text += f"{i+1}. {c['name']} | Type: {c['test_type']} | {c['description'][:100]}\n"
    
    prompt = f"""
    You are an expert HR assessment consultant.
    
    Job Query: {query[:500]}
    
    From these candidate assessments, select the BEST 5 to 10 that:
    1. Are most relevant to the job query
    2. Cover both technical AND soft/behavioral skills if mentioned
    3. Are balanced mix of test types (K for technical, P for personality, A for ability)
    4. Would genuinely help screen candidates for this specific role
    
    Candidates:
    {candidates_text}
    
    Return ONLY the numbers of selected assessments as comma-separated values.
    Example: 1,3,5,7,9
    No explanation. Just numbers.
    """
    
    response = llm.generate_content(prompt)
    selected_text = response.text.strip()
    
    # Parse selected numbers
    try:
        selected = [int(x.strip()) - 1 for x in selected_text.split(",") if x.strip().isdigit()]
        selected = [n for n in selected if 0 <= n < len(candidates)]
    except:
        selected = list(range(min(10, len(candidates))))
    
    # Ensure 5-10 results
    if len(selected) < 5:
        selected = list(range(min(10, len(candidates))))
    if len(selected) > 10:
        selected = selected[:10]
    
    return [candidates[i] for i in selected]



@app.get("/health")
def health_check():
    """
    Simple health check to verify API is running
    """
    return {
        "status": "healthy",
        "message": "SHL Recommendation API is running!",
        "total_assessments": len(df)
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: RecommendRequest):
    """
    Takes a job description or natural language query
    Returns 5-10 most relevant SHL assessments
    """
    
    query = request.query.strip()
    top_k = min(max(request.top_k, 5), 10)  # Keep between 5 and 10
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if query.startswith("http://") or query.startswith("https://"):
        query = extract_text_from_url(query)
    
    analysis = understand_query(query)
    search_query = analysis.get("SEARCH_QUERY", query)
    
    candidates = search_assessments(search_query, top_k=20)
    
    final_results = rerank_with_gemini(query, candidates)
    
    assessments = []
    for r in final_results:
        assessments.append(Assessment(
            assessment_name=r["name"],
            url=r["url"],
            test_type=r["test_type"],
            remote_testing=r["remote_testing"],
            adaptive_irt=r["adaptive_irt"],
            description=r["description"]
        ))
    
    return RecommendResponse(
        query=request.query,
        total_results=len(assessments),
        recommended_assessments=assessments
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




