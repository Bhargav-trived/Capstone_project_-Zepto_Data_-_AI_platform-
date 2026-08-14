# -*- coding: utf-8 -*-
import os
import json
import glob
from typing import List, Optional, TypedDict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

BASE_DIR = os.getcwd()
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

DOCUMENTS = {
    "doc_01.txt": "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
    "doc_02.txt": "Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
    "doc_03.txt": "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.",
    "doc_04.txt": "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.",
    "doc_05.txt": "Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.",
    "doc_06.txt": "If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.",
    "doc_07.txt": "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.",
    "doc_08.txt": "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered."
}

def ensure_docs_exist():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        for filename, content in DOCUMENTS.items():
            with open(os.path.join(DOCS_DIR, filename), "w", encoding="utf-8") as f:
                f.write(content)

# ============================================================================
# 2. SCHEMAS
# ============================================================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="The customer support question")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The response provided to the customer")
    sources: List[str] = Field(default_factory=list, description="Document IDs used. Empty for general queries.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")

# ============================================================================
# 3. PROMPTS
# ============================================================================
RAG_SYSTEM_PROMPT = """
### ROLE
You are Zepto's official AI Support Assistant. Your duty is to provide helpful, concise, and strictly accurate customer service answers.

### CONTEXT
You will be provided with retrieved context snippets from Zepto's official policy documentation:
{context}

### TASK
Answer the customer's question directly based ONLY on the provided context:
Question: {question}

### CONSTRAINTS
1. Do NOT answer using information, assumptions, or external knowledge not explicitly present in the provided context.
2. If the context does not contain the answer, state: "I do not have enough policy information to answer that question."
3. Do NOT make promises or mention features outside the given policy text.

### FORMAT
Respond ONLY with valid JSON matching the following schema:
{{
  "answer": "<concise response grounded in context>",
  "sources": ["<doc_id_1>", "<doc_id_2>"],
  "confidence": <float between 0.0 and 1.0>
}}

### LENGTH
Keep the "answer" field under 3 sentences and under 80 words.

### FEW-SHOT EXAMPLE
Context:
[doc_01]: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation... Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.

Question: Is standard delivery free?

Response:
{{
  "answer": "Standard delivery is free on orders over INR 149. Orders below INR 149 incur a flat delivery fee of INR 25.",
  "sources": ["doc_01"],
  "confidence": 0.95
}}
"""

INTENT_SYSTEM_PROMPT = """
You are a query classifier for Zepto customer support. Classify the user query into either 'policy_question' or 'general_question'.
Return only JSON: {"intent": "policy_question" | "general_question"}
"""

# ============================================================================
# 4. VECTOR STORE INGESTION & RETRIEVAL
# ============================================================================
_embedder = None
_collection = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def get_vectorstore():
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = client.get_or_create_collection(
        name="zepto_policies", metadata={"hnsw:space": "cosine"}
    )

    if _collection.count() == 0:
        doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
        model = get_embedder()
        documents, ids, metadatas = [], [], []

        for file_path in doc_files:
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read().strip())
            ids.append(doc_id)
            metadatas.append({"source": doc_id})

        embeddings = model.encode(documents, show_progress_bar=False).tolist()
        _collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    return _collection

def query_similar_chunks(query_text: str, top_k: int = 3):
    collection = get_vectorstore()
    query_emb = get_embedder().encode([query_text], show_progress_bar=False).tolist()
    return collection.query(query_embeddings=query_emb, n_results=min(top_k, collection.count()))

# ============================================================================
# 5. LANGGRAPH PIPELINE
# ============================================================================
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_chunks: List[str]
    retrieved_ids: List[str]
    final_response: Optional[dict]

def is_mock_mode() -> bool:
    mock_env = os.environ.get("MOCK_LLM", "1").strip().lower()
    return mock_env in ("1", "true", "yes", "")

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def classify_intent(state: AgentState) -> AgentState:
    query = state["query"].lower()
    if is_mock_mode():
        matched = any(kw in query for kw in POLICY_KEYWORDS)
        state["intent"] = "policy_question" if matched else "general_question"
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
            completion = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "system", "content": INTENT_SYSTEM_PROMPT}, {"role": "user", "content": state["query"]}],
                temperature=0.0,
            )
            parsed = json.loads(completion.choices[0].message.content.strip())
            state["intent"] = parsed.get("intent", "general_question")
        except Exception:
            state["intent"] = "policy_question" if any(kw in query for kw in POLICY_KEYWORDS) else "general_question"
    return state

def call_llm_with_retry(prompt: str, source_ids: List[str], max_retries: int = 2) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "system", "content": "You are a precise JSON-only engine."}, {"role": "user", "content": current_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return QueryResponse(**json.loads(completion.choices[0].message.content.strip())).model_dump()
        except Exception as err:
            if attempt == max_retries:
                return QueryResponse(answer=f"Error: {str(err)}", sources=source_ids, confidence=0.0).model_dump()
            current_prompt += f"\n\nCorrection needed: Failed validation with error: {str(err)}. Ensure pure JSON adhering to schema."

def retrieve_and_answer(state: AgentState) -> AgentState:
    results = query_similar_chunks(state["query"], top_k=3)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    state["retrieved_chunks"] = docs
    state["retrieved_ids"] = ids

    if is_mock_mode():
        top_snippet = docs[0][:200] if docs else "No policy found."
        state["final_response"] = QueryResponse(
            answer=f"Based on the retrieved context: {top_snippet}",
            sources=ids,
            confidence=1.0,
        ).model_dump()
    else:
        context_str = "\n\n".join([f"[{doc_id}]: {text}" for doc_id, text in zip(ids, docs)])
        state["final_response"] = call_llm_with_retry(RAG_SYSTEM_PROMPT.format(context=context_str, question=state["query"]), ids)

    return state

def direct_answer(state: AgentState) -> AgentState:
    if is_mock_mode():
        state["final_response"] = QueryResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0).model_dump()
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
            completion = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "system", "content": "You are a friendly Zepto customer assistant. Answer briefly."}, {"role": "user", "content": state["query"]}],
                temperature=0.2,
            )
            state["final_response"] = QueryResponse(answer=completion.choices[0].message.content.strip(), sources=[], confidence=0.9).model_dump()
        except Exception:
            state["final_response"] = QueryResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0).model_dump()
    return state

def route_intent(state: AgentState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

graph_builder = StateGraph(AgentState)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer)

graph_builder.set_entry_point("classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent, {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"})
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)
app_graph = graph_builder.compile()

# ============================================================================
# 6. FASTAPI APPLICATION
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_docs_exist()
    get_vectorstore()  # Pre-index corpus into ChromaDB at startup
    yield

app = FastAPI(title="Zepto Support Assistant API", version="1.0.0", lifespan=lifespan)

@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(payload: QueryRequest):
    try:
        initial_state = {"query": payload.query, "intent": None, "retrieved_chunks": [], "retrieved_ids": [], "final_response": None}
        result = app_graph.invoke(initial_state)
        return QueryResponse(**result.get("final_response"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
