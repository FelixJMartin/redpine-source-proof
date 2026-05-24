# Import
import os
import json
import uuid
import requests
from bs4 import BeautifulSoup
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

REDPINE_KEY = os.getenv("REDPINE_API_KEY")
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Qdrant setup: two collections for web chunks and evaluations
qclient = QdrantClient(path="./qdrant_db")

for col in ["web_chunks", "evaluations"]:
    try:
        qclient.delete_collection(col)
    except:
        pass

qclient.create_collection("web_chunks",    vectors_config=VectorParams(size=384, distance=Distance.COSINE))
qclient.create_collection("evaluations",   vectors_config=VectorParams(size=384, distance=Distance.COSINE))
print("[Qdrant] Collections ready: web_chunks, evaluations")




def ask(prompt
        , max_tokens=600):
    """Send a prompt to the Groq chat model and return the text response."""
    res = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

# 1Redpine fetch
def fetch_redpine():
    """Fetch trending licensed media data from the Redpine API."""
    print("\n[Redpine] Fetching licensed media data...")
    res = requests.post(
        "https://api.redpine.ai/api/v1/tools/media/trending",
        headers={"Authorization": f"Bearer {REDPINE_KEY}"},
        json={"period": "7d", "metric": "mentions", "limit": 10}
    )
    data = res.json()
    print(f"[Redpine] Got {len(str(data))} bytes of licensed data")
    return data

# Web scraping and Qdrant
def scrape(url):
    """Download a URL and return cleaned text (shortened to 3k chars)."""
    try:
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:3000]
    except:
        return ""

def build_web_rag(query):
    """Discover URLs via LLM, scrape them, chunk text, embed and store in Qdrant."""
    print(f"\n[Web] Finding sources for: '{query}'")

    urls_raw = ask(f"Give me 5 real URLs of recent news articles about: {query}. Return only a JSON array of URLs, nothing else.")
    urls_raw = urls_raw.replace("```json", "").replace("```", "").strip()
    try:
        urls = json.loads(urls_raw)
    except:
        urls = []
    print(f"[Web] Found {len(urls)} sources")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    points = []

    for url in urls:
        text = scrape(url)
        if not text:
            continue
        docs = splitter.create_documents([text])
        for doc in docs:
            vec = embeddings.embed_query(doc.page_content)
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"url": url, "content": doc.page_content}
            ))

    if points:
        qclient.upsert(collection_name="web_chunks", points=points)
    print(f"[Web] {len(points)} chunks stored in Qdrant")
    return points

def retrieve_web(query, k=5):
    """Retrieve top-k web chunks from Qdrant for the given query."""
    vec = embeddings.embed_query(query)
    results = qclient.query_points(collection_name="web_chunks", query=vec, limit=k).points
    return [r.payload for r in results]

# CAAR
def caar_classify(query):
    """Use the LLM to classify the query into RECENCY, DEPTH, or AUTHORITY."""
    raw = ask(f"""Classify this query into exactly one of these context classes:
- RECENCY: needs current, real-time, or recent data
- DEPTH: needs technical detail, domain expertise, or research
- AUTHORITY: needs trusted sources, citations, or credibility

# Query: "{query}"

Return only one word: RECENCY, DEPTH, or AUTHORITY""", max_tokens=10)
    cls = raw.strip().upper()
    if cls not in ["RECENCY", "DEPTH", "AUTHORITY"]:
        cls = "RECENCY"
    print(f"\n[CAAR] Query classified as: {cls}")
    return cls

def caar_score(answer, context_class, source):
    """Score an answer 0-10 for a given context class using the LLM."""
    criteria = {
        "RECENCY":   "real-time data, specific dates, current figures, live metrics",
        "DEPTH":     "technical terms, domain specificity, detailed analysis, precise numbers",
        "AUTHORITY": "named sources, citations, licensed data, verified provenance"
    }[context_class]

    raw = ask(f"""Score this answer 0-10 for a {context_class} query.
Criteria: {criteria}

Answer: {answer[:600]}

Return only a number 0-10, nothing else.""", max_tokens=5)

    try:
        score = float(raw.strip().split()[0])
        score = min(10, max(0, score))
    except:
        score = 5.0

    print(f"[CAAR] {source} score ({context_class}): {score}/10")
    return score

# Generate answers from Redpine data and web context
def generate_answers(query, redpine_data, web_chunks):
    """Produce answers using licensed Redpine data and retrieved web context."""
    context = "\n\n".join([c["content"] for c in web_chunks])

    redpine_answer = ask(f"""Summarize this licensed real-time data to answer: {query}

Data: {json.dumps(redpine_data, indent=2)[:2000]}""")

    web_answer = ask(f"""Using only this web context, answer: {query}

Context: {context[:2000]}""")

    return redpine_answer, web_answer

# Save evaluation to Qdrant
def save_evaluation(query, context_class, redpine_answer, web_answer, r_score, w_score):
    """Store a comparative evaluation of Redpine vs web answers in Qdrant."""
    why = ask(f"""In 2 sentences, explain why {
        'Redpine' if r_score > w_score else 'the open web'
    } scored higher for this {context_class} query.

Redpine score: {r_score}/10
Web score: {w_score}/10
Query: {query}""")

    evaluation = {
        "query": query,
        "context_class": context_class,
        "redpine_score": r_score,
        "web_score": w_score,
        "redpine_answer": redpine_answer,
        "web_answer": web_answer,
        "why": why
    }

    vec = embeddings.embed_query(query)
    qclient.upsert(collection_name="evaluations", points=[
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload=evaluation)
    ])
    print(f"[Qdrant] Evaluation saved")
    return why

# Print results to console
def print_results(query, context_class, redpine_answer, web_answer, r_score, w_score, why):
    """Pretty-print query, scores, answers, and the winning source."""
    winner = "REDPINE" if r_score > w_score else "OPEN WEB"

    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print(f"CAAR CLASS: {context_class}")
    print("="*60)

    print(f"\n[REDPINE — {r_score}/10]")
    print(redpine_answer)

    print(f"\n[OPEN WEB RAG — {w_score}/10]")
    print(web_answer)

    print(f"\n{'='*60}")
    print(f"WINNER: {winner}")
    print(f"WHY: {why}")
    print(f"{'='*60}")
    print(f"\nEvaluation saved to Qdrant — queryable in workspace")


def followup_eval(query, redpine_data, context_class):
    """Generate follow-up questions from Redpine data and evaluate them against Qdrant."""
    print("\n[Followup] Generating 10 follow-up questions...")
    
    raw = ask(f"""Based on this ACTUAL data:
    {json.dumps(redpine_data)[:1000]}
Generate 10 specific follow-up questions about THIS actual data.
For example: "Why did Amgen surge 126500%?" or "What channels drove NextEra mentions?"
Return only a JSON array of strings, nothing else.""")
    raw = raw.replace("```json","").replace("```","").strip()
    try:
        questions = json.loads(raw)[:10]
    except:
        questions = []

    print(f"[Followup] Running {len(questions)} questions against Qdrant...\n")
    
    results = []
    for q in questions:
        web_chunks = retrieve_web(q, k=3)
        context = "\n\n".join([c["content"] for c in web_chunks])
        
        r_ans = ask(f"Answer using only this data: {json.dumps(redpine_data)[:1500]}\n\nQuestion: {q}")
        w_ans = ask(f"Answer using only this context: {context[:1500]}\n\nQuestion: {q}")
        
        r_score = caar_score(r_ans, context_class, "Redpine")
        w_score = caar_score(w_ans, context_class, "Web")
        
        results.append({
            "question": q,
            "redpine_score": r_score,
            "web_score": w_score
        })
        print(f"  Q: {q[:60]}")
        print(f"  Redpine: {r_score}/10  Web: {w_score}/10\n")

    avg_r = sum(r["redpine_score"] for r in results) / len(results) if results else 0
    avg_w = sum(r["web_score"] for r in results) / len(results) if results else 0
    
    print("="*60)
    print(f"FOLLOWUP SUMMARY — {len(results)} questions")
    print(f"Redpine avg: {avg_r:.1f}/10")
    print(f"Web avg:     {avg_w:.1f}/10")
    print(f"Redpine won: {sum(1 for r in results if r['redpine_score'] > r['web_score'])}/{len(results)} questions")
    print("="*60)


# Main execution
if __name__ == "__main__":
    query = "which companies are trending in media right now and what is the sentiment"

    redpine_data  = fetch_redpine()
    build_web_rag(query)
    web_chunks    = retrieve_web(query)
    context_class = caar_classify(query)
    r_ans, w_ans  = generate_answers(query, redpine_data, web_chunks)
    r_score       = caar_score(r_ans, context_class, "Redpine")
    w_score       = caar_score(w_ans, context_class, "Web RAG")
    why           = save_evaluation(query, context_class, r_ans, w_ans, r_score, w_score)

    print_results(query, context_class, r_ans, w_ans, r_score, w_score, why)
    followup_eval(query, redpine_data, context_class)