"""
tests/run_live_web_research
===========================
Live real-world Web Research (WR) stress-test agent powered by local Ollama.

This script:
1. Wires up SotisLangGraphGuard as a state interceptor node.
2. Registers real search and scrape tools to access the live internet.
3. Guides the local LLM agent to research 2026 superconductor developments.
4. Generates live telemetry logs and intercepts any tool loops or path blockages.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any, Dict, List, Sequence
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

# Try imports
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    from bs4 import BeautifulSoup
    import httpx
except ImportError as e:
    print(f"[Sotis-WR] Required package missing: {e}. Run pip install.")
    sys.exit(1)

from sotis.core.entropy import EntropyConfig
from sotis.core.schemas import Domain
from sotis.lib.langgraph_integration import SotisLangGraphGuard


# ─────────────────────────────────────────────────────────────────────────────
# 1. State Definition
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """LangGraph conversation state wrapper."""
    messages: Annotated[List[BaseMessage], add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Live Web Research Workspace & Tools
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ExperimentLog", "web research loop", "workspace")
)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

@tool
def web_search(query: str) -> str:
    """Search the real live internet for information. Returns titles, snippets, and URLs."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n---")
        if not results:
            return "[SEARCH RESULT] No relevant results found."
        return "\n".join(results)
    except Exception as e:
        return f"[ERROR] Search failed: {e}"


@tool
def web_scrape(url: str) -> str:
    """Scrape and read the plain text content of a live URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # Perform live HTTP request
        with httpx.Client(headers=headers, follow_redirects=True, timeout=10) as client:
            resp = client.get(url)
            
        if resp.status_code != 200:
            return f"[ERROR] Scrape failed with status code {resp.status_code}"
            
        # Parse HTML to text using BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "noscript", "header", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator="\n")
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        # Return first 3000 chars to avoid swamping context
        return clean_text[:3000]
    except Exception as e:
        return f"[ERROR] Scraping failed for {url}: {e}"


@tool
def write_workspace_file(file_path: str, content: str) -> str:
    """Write research findings to the local workspace. Path must be relative (e.g. 'app/superconductor_report.txt')."""
    full_path = os.path.join(WORKSPACE_DIR, file_path)
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[SUCCESS] Wrote report to {file_path}"
    except Exception as e:
        return f"[ERROR] Failed to write report: {e}"


@tool
def list_workspace_files() -> List[str]:
    """List all file paths recursively inside the workspace."""
    files_list = []
    for root, _, files in os.walk(WORKSPACE_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, WORKSPACE_DIR)
            files_list.append(rel_path.replace("\\", "/"))
    return files_list


TOOLS = {
    "web_search": web_search,
    "web_scrape": web_scrape,
    "write_workspace_file": write_workspace_file,
    "list_workspace_files": list_workspace_files,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Live Model Execution Mode (Online Stress Test)
# ─────────────────────────────────────────────────────────────────────────────

def run_live_eval(guard: SotisLangGraphGuard, api_key: str, base_url: str, model: str) -> None:
    """Runs end-to-end execution of a real LangGraph agent via local Ollama or OpenAI."""
    print("\n" + "="*80)
    print(f"RUNNING TRACK 2: REAL WEB RESEARCH AGENT ({model.upper()} STRESS TEST)")
    print("="*80)

    # Initialize the LLM
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        openai_api_key=api_key,
        base_url=base_url,
        max_tokens=2048,
    )

    tools_list = [web_search, web_scrape, write_workspace_file, list_workspace_files]
    llm_with_tools = llm.bind_tools(tools_list, parallel_tool_calls=False)

    # Node 1: Agent thinking
    def call_agent(state: AgentState) -> Dict[str, Any]:
        resp = llm_with_tools.invoke(state["messages"])
        return {"messages": [resp]}

    # Node 2: Tool execution
    def call_tools(state: AgentState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_outputs = []
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = tc["args"]
            t_id = tc["id"]
            if t_name in TOOLS:
                print(f"Running Tool: {t_name}({t_args})")
                try:
                    res = TOOLS[t_name].invoke(t_args)
                    tool_outputs.append(ToolMessage(content=str(res), name=t_name, tool_call_id=t_id))
                except Exception as e:
                    print(f"Tool execution failed: {e}")
                    tool_outputs.append(ToolMessage(
                        content=f"[ERROR] Failed to run tool {t_name}: {e}",
                        name=t_name,
                        tool_call_id=t_id
                    ))
            else:
                tool_outputs.append(ToolMessage(content=f"[ERROR] Tool {t_name} not found.", name=t_name, tool_call_id=t_id))
        return {"messages": tool_outputs}

    # Wires our Sotis Middleware node
    def call_sotis_guard(state: AgentState) -> Dict[str, Any]:
        # Intercept and watch state trajectory
        update = guard.guard_node(state)
        if update:
            print("\n" + "#"*80)
            print("!!! [SOTIS INTERCEPTED MELTDOWN & RESUMED FRESH] !!!")
            print("#"*80)
        return update

    # Edge routing logic
    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        
        # Check if Sotis just emitted a resumption prompt
        if isinstance(last_msg, HumanMessage) and isinstance(last_msg.content, str) and "Context Reset Notice" in last_msg.content:
            return "agent"
            
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", []):
            return "tools"
            
        # If the agent output plain text without tool calls, we intercept using Smart Fallback
        if isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", []):
            # Enforce that the task is only complete when the file has actually been written to the workspace
            report_path = os.path.join(WORKSPACE_DIR, "app/superconductor_report.txt")
            if os.path.exists(report_path):
                return END
            
            print("\n>>> [Smart Fallback] Agent returned text-only response without tool calls. Prompting to use tools to complete the task.")
            # Inject correction prompt
            state["messages"].append(HumanMessage(
                content="You provided text thoughts or an explanation, but you did not perform any tool action. The task is not yet complete (report must be written to workspace). Please call a tool (such as web_search, web_scrape, or write_workspace_file) to proceed."
            ))
            return "sotis"
            
        return END

    # 4. Build LangGraph
    builder = StateGraph(AgentState)  # type: ignore[arg-type]
    builder.add_node("agent", call_agent)
    builder.add_node("tools", call_tools)
    builder.add_node("sotis", call_sotis_guard)

    builder.add_edge(START, "agent")
    builder.add_edge("tools", "sotis")
    builder.add_edge("sotis", "agent")
    
    builder.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "sotis": "sotis",
        END: END
    })

    graph = builder.compile()

    # Initial state inputs
    initial_prompt = (
        "Search the live internet for recent news regarding room-temperature superconductivity research published in 2026. "
        "Identify the names of at least 2 institutions/laboratories claiming breakthroughs, and write a detailed report named "
        "'app/superconductor_report.txt' in the workspace. Call tools in every turn until the file is written."
    )
    inputs = {"messages": [HumanMessage(content=initial_prompt)]}

    print(f"Task Initialized: {initial_prompt}\n")

    step_count = 0
    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"\n--- [Node: {node}] ---")
            if values and "messages" in values:
                for msg in values["messages"]:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"Assistant calls: {tc['name']}({tc['args']})")
                    elif isinstance(msg, ToolMessage):
                        print(f"Tool Result: {msg.content[:200]}...")
                    elif isinstance(msg, HumanMessage) and "Context Reset Notice" in msg.content:
                        print(f"Sotis Resumption Prompt: {msg.content[:400]}...")
                    else:
                        print(f"Content: {msg.content[:200]}")
            step_count += 1
            if step_count > 40:
                print("\n[ERROR] Safety Step limit exceeded. Hard stopping.")
                break

    print(f"\nLive web research finished in {step_count} transitions. Resets used: {guard.total_resets}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    goal = "Search the live internet for superconductor breakthroughs and write app/superconductor_report.txt."
    
    # Configure SotisGuard for Web Research
    guard = SotisLangGraphGuard(
        task_goal=goal,
        workspace_paths=["app/superconductor_report.txt"],
        domain=Domain.WEB_RESEARCH,
        entropy_config=EntropyConfig(hard_threshold=2.2),
    )

    # Defaults to local Ollama server running at localhost:11434
    openai_key = os.environ.get("OPENAI_API_KEY", "ollama")
    openai_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
    openai_model = os.environ.get("OPENAI_MODEL", "mistral:latest")

    # Perform a quick reachability check if running local Ollama
    ollama_offline = False
    if "localhost" in openai_base or "127.0.0.1" in openai_base:
        try:
            # Ping localhost port
            with httpx.Client(timeout=1.5) as client:
                ping_url = openai_base.replace("/v1", "")
                resp = client.get(ping_url)
                if resp.status_code not in (200, 204, 404):
                    ollama_offline = True
        except Exception:
            ollama_offline = True

    if ollama_offline and openai_key == "ollama":
        print("\n" + "!"*80)
        print("CONNECTION ERROR: Local Ollama server is offline or unreachable at " + openai_base)
        print("To run the live web crawler agent, please:")
        print("  1. Launch Ollama locally by running: `ollama serve`")
        print(f"  2. Pull the required model: `ollama pull {openai_model}`")
        print("!"*80)
        sys.exit(1)
    else:
        try:
            run_live_eval(guard, openai_key, openai_base, openai_model)
        except Exception as e:
            print(f"\n[ERROR] Connection or execution error: {e}")
            sys.exit(1)
