# Retriever Agent
# Handles indexing and retrieval from vector store

from fastapi import APIRouter, Query
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
import faiss
import logging

router = APIRouter(prefix="/retriever", tags=["Retriever Agent"])

embeddings = HuggingFaceEmbeddings()
# Get embedding size correctly (HuggingFaceEmbeddings returns a list)
test_emb = embeddings.embed_query("test")
embedding_size = len(test_emb)
index = faiss.IndexFlatL2(embedding_size)
docstore = InMemoryDocstore({})
vector_store = FAISS(embeddings.embed_query, index, docstore, {})

logger = logging.getLogger("finbreaker")

@router.post("/index")
def index_documents(docs: List[str]):
    logger.info(f"Indexing {len(docs)} documents.")
    doc_objs = [Document(page_content=doc) for doc in docs]
    ids = vector_store.add_documents(doc_objs)
    logger.info(f"Indexed {len(ids)} documents.")
    return {"indexed": len(ids)}

@router.get("/retrieve")
def retrieve(query: str, k: int = 3):
    logger.info(f"Retrieving top {k} results for query: {query}")
    results = vector_store.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} results.")
    return {"results": [r.page_content for r in results]}

@router.get("/")
def root():
    return {"status": "Retriever Agent running"}
