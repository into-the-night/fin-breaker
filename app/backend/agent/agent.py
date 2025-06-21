import asyncio
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from app.backend.services.market_data import get_market_data
from app.backend.services.retrieval import get_vector_store
from app.backend.services.synthesis import get_llm_service
from app.backend.agent.tools import get_tools, TOOL_MAP
import google.generativeai.types as genai_types


class AgentState(TypedDict):
    prompt: str
    plan: str
    tool_calls: List[Dict[str, Any]]
    context: List[str]
    output: str
    replan_count: int

# --- Services and Tools ---
llm_service = get_llm_service()
vector_store_service = get_vector_store()
market_data_service = get_market_data()
tools = get_tools()
tool_declarations = [genai_types.FunctionDeclaration.from_callable(f) for f in tools]
gemini_tools = [genai_types.Tool(function_declarations=tool_declarations)]


# --- Nodes ---
async def planner_node(state: AgentState):
    print("---PLANNER---")
    if state.get("replan_count", 0) > 3:
        # If we've replanned too many times, we might be in a loop.
        return {"output": "I'm sorry, I'm having trouble finding the answer. Please try rephrasing your question."}

    response = await llm_service.generate_plan(state["prompt"], gemini_tools)
    
    tool_calls = []
    plan_text = ""
    for part in response.content:
        if part.type == "tool_call":
            tool_calls.append({
                "name": part.name,
                "args": part.input
            })
        elif part.type == "text":
            plan_text += part.text

    return {
        "plan": plan_text,
        "tool_calls": tool_calls,
        "replan_count": state.get("replan_count", 0) + 1
    }

def toolbox_node(state: AgentState):
    print("---TOOLBOX---")
    context = state.get("context", [])
    tool_calls = state["tool_calls"]
    
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        if tool_name in TOOL_MAP:
            result = TOOL_MAP[tool_name](**tool_args)
            context.append(f"Tool: {tool_name}\nArguments: {tool_args}\nResult: {result}\n")
    
    return {"context": context}

async def evaluator_node(state: AgentState):
    print("---EVALUATOR---")
    question = state["prompt"]
    context = state["context"]
    
    if not context:
        # No context gathered, need to replan
        return {"context_enough": "REPLAN"}
        
    evaluation = await llm_service.evaluate_context(question, context)
    
    if "CONTINUE" in evaluation:
        return {"context_enough": "CONTINUE"}
    else:
        return {"context_enough": "REPLAN"}

async def synthesis_node(state: AgentState):
    print("---SYNTHESIS---")
    question = state["prompt"]
    context = state["context"]
    answer = await llm_service.synthesize_with_context(question, context)
    return {"output": answer}

# --- Conditional Edges ---
def should_continue(state: AgentState):
    if state.get("output"):
        return END # If we have an output from planner (error) or synthesis
    if state["tool_calls"]:
        return "toolbox"
    # If no tool calls, maybe the planner answered directly or has a question.
    # For now, let's synthesize with what we have (empty context).
    return "synthesis"

def after_evaluator(state: AgentState):
    if state["context_enough"] == "CONTINUE":
        return "synthesis"
    else:
        return "planner"

# --- Graph ---
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("toolbox", toolbox_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        should_continue,
    )
    workflow.add_edge("toolbox", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        after_evaluator,
        {"CONTINUE": "synthesis", "REPLAN": "planner"},
    )
    workflow.add_edge("synthesis", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

async def run_agent(question: str):
    app = create_graph()
    config = {"configurable": {"thread_id": "1"}}
    async for event in app.astream(
        {"prompt": question, "replan_count": 0},
        config=config,
    ):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":
    asyncio.run(run_agent("What were the earnings for NVDA in the last quarter?"))

