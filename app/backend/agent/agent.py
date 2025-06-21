from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from typing import List, Dict, Union
from app.backend.services.market_data import get_market_data
from app.backend.services.retrieval import get_vector_store
from app.backend.services.synthesis import get_llm_service

llm_service = get_llm_service()
vector_store_service = get_vector_store()
market_data_service = get_market_data()

# ---- Nodes ----
def entry_node(state):
    return {"prompt": state["prompt"]}

def planner_node(state):
    prompt = state["prompt"]
    # Dummy planning decision: decide tools based on keywords (replace with LLM planner)
    tools_to_use = []
    if "earnings" in prompt.lower():
        tools_to_use.append("Earnings")
    if "risk" in prompt.lower() or "exposure" in prompt.lower():
        tools_to_use.append("Market Data")
    if not tools_to_use:
        return {"question_for_clarity": "Can you clarify your request with more details?"}
    return {"tool_calls": tools_to_use, "tool_args": {"ticker": "TSMC"}}  # dummy args

def toolbox_node(state):
    tools = {
        "Market Data": market_data_service.fetch_time_series_market_data,
        "Earnings": market_data_service.fetch_earnings,
        "Company News": market_data_service.fetch_company_news,
        "Ticker Search": market_data_service.search_ticker,
        "Topic News": market_data_service.fetch_topic_news,
        "Index Data": vector_store_service.index_documents,
        "Retriever": vector_store_service.retrieve,
    }
    results = []
    for tool_name in state["tool_calls"]:
        tool_func = tools[tool_name]
        result = tool_func(**state["tool_args"])
        results.append(result)
    return {"context": results}

def rag_node(state):
    queries = state.get("prompt", "")
    context = state.get("context", [])
    docs = vector_store_service.retrieve(query=queries)
    return {"retrieved_docs": docs, "prompt": queries, "context": context + docs}

def evaluator_node(state):
    context = state["context"]
    prompt = state["prompt"]
    if len(context) < 2:  # Dummy confidence check
        return {"new_queries": True}
    return {"context_enough": True}

def synthesis_node(state):
    question = state["prompt"]
    context = state["context"]
    answer = llm_service.synthesize_with_context(question, context)
    return {"output": answer}

# ---- Graph Building ----
graph = StateGraph()

graph.add_node("entry", RunnableLambda(entry_node))
graph.add_node("planner", RunnableLambda(planner_node))
graph.add_node("toolbox", RunnableLambda(toolbox_node))
graph.add_node("rag", RunnableLambda(rag_node))
graph.add_node("evaluator", RunnableLambda(evaluator_node))
graph.add_node("synthesis", RunnableLambda(synthesis_node))

# Edges

graph.set_entry_point("entry")

graph.add_edge("entry", "planner")
graph.add_conditional_edges("planner", lambda x: "question_for_clarity" in x,
                             {True: END, False: "toolbox"})
graph.add_edge("toolbox", "rag")
graph.add_edge("rag", "evaluator")
graph.add_conditional_edges("evaluator", lambda x: x.get("context_enough", False),
                             {True: "synthesis", False: END})
graph.add_edge("synthesis", END)

# Build and return
app = graph.compile()

