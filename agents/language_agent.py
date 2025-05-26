# Language Agent
# Handles LLM-based narrative synthesis

from fastapi import APIRouter, Query
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

router = APIRouter(prefix="/language", tags=["Language Agent"])

llm = OpenAI(temperature=0.2)

@router.post("/synthesize")
def synthesize(
    question: str,
    context: list
):
    # Use LangChain's RetrievalQA for RAG
    docs = [Document(page_content=chunk) for chunk in context]
    chain = RetrievalQA.from_chain_type(llm, retriever=None)  # retriever can be plugged in
    # For demo, just concatenate context and answer
    answer = llm("\n".join(context) + "\n" + question)
    return {"answer": answer}

@router.get("/")
def root():
    return {"status": "Language Agent running"}
