# Retriever Agent
# Handles indexing and retrieval from vector store

from fastapi import APIRouter, Query
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

router = APIRouter(prefix="/retriever", tags=["Retriever Agent"])

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS(embeddings.embed_query, [])

@router.post("/index")
def index_documents(docs: List[str]):
    ids = vector_store.add_texts(docs)
    return {"indexed": len(ids)}

@router.get("/retrieve")
def retrieve(query: str, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return {"results": [r.page_content for r in results]}

@router.get("/")
def root():
    return {"status": "Retriever Agent running"}
