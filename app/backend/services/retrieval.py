# Retriever Agent
# Handles indexing and retrieval from vector store

from fastapi import APIRouter
from functools import lru_cache
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
import faiss
import logging

router = APIRouter(prefix="/retriever", tags=["Retriever Agent"])

model_name = "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}
logger = logging.getLogger("finbreaker")


class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(  
            model_name=model_name,  
            model_kwargs=model_kwargs,  
            encode_kwargs=encode_kwargs  
            )
        test_emb = self.embeddings.embed_query("test")
        embedding_size = len(test_emb)
        index = faiss.IndexFlatL2(embedding_size)
        docstore = InMemoryDocstore({})
        self.vector_store = FAISS(self.embeddings.embed_query, index, docstore, {})
        
    def index_documents(self, docs: List[str]):
        logger.info(f"Indexing {len(docs)} documents.")
        doc_objs = [Document(page_content=doc) for doc in docs]
        ids = self.vector_store.add_documents(doc_objs)
        logger.info(f"Indexed {len(ids)} documents.")
        return {"indexed": len(ids)}

    def retrieve(self, query: str, k: int = 3):
        logger.info(f"Retrieving top {k} results for query: {query}")
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} results.")
        return {"results": [r.page_content for r in results]}
    

@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()