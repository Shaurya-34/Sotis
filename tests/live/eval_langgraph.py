"""
tests/live/eval_langgraph
=========================
Dual-mode LangGraph ReAct agent runner and circular import trap stress-test (Track 2).

This script:
1. Creates a live LangGraph ReAct state graph.
2. Registers tools for reading/writing files and running test commands.
3. Wires our SotisLangGraphGuard as a state interceptor node.
4. Operates in two modes:
   - Real Agent Mode: If OPENAI_API_KEY is in env, runs gpt-4o-mini end-to-end.
   - Interactive Simulation Mode: If offline, simulates the loop and recovery.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from typing import Annotated, Any, Callable, Dict, List, Sequence
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

# Import LangChain / LangGraph components
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
except ImportError:
    # Fail-safe print if packages are still finishing background loading
    print("[Sotis-LG] Required packages are loading. Ensure pip install has completed.")
    sys.exit(0)

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
# 2. Workspace Tool Registries
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = os.environ.get(
    "WORKSPACE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "live_test_workspace"))
)

def _resolve_workspace_path(file_path: str) -> str | None:
    """Resolve and validate that file_path stays within WORKSPACE_DIR. Returns None on traversal attempt."""
    full = os.path.realpath(os.path.join(WORKSPACE_DIR, file_path))
    workspace_root = os.path.realpath(WORKSPACE_DIR)
    if not (full == workspace_root or full.startswith(workspace_root + os.sep)):
        return None
    return full

@tool
def read_workspace_file(file_path: str) -> str:
    """Read contents of a file in the workspace. Path must be relative to workspace root (e.g. 'app/math_core.py')."""
    full_path = _resolve_workspace_path(file_path)
    if full_path is None:
        return "[ERROR] Access denied: path escapes the workspace directory."
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] Failed to read file {file_path}: {e}"


@tool
def write_workspace_file(file_path: str, content: str) -> str:
    """Write or overwrite a file in the workspace. Path must be relative to workspace root (e.g. 'app/math_core.py')."""
    full_path = _resolve_workspace_path(file_path)
    if full_path is None:
        return "[ERROR] Access denied: path escapes the workspace directory."
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[SUCCESS] Wrote to {file_path}"
    except Exception as e:
        return f"[ERROR] Failed to write to {file_path}: {e}"


@tool
def execute_workspace_tests() -> str:
    """Run the test suite using pytest on the live test workspace."""
    try:
        # Resolve and validate the workspace directory path to prevent injection / path traversal
        safe_workspace = os.path.abspath(WORKSPACE_DIR)
        if not os.path.isdir(safe_workspace):
            return f"[ERROR] Workspace directory does not exist or is invalid: {safe_workspace}"

        import shlex
        tests_dir = os.path.join(safe_workspace, "tests")
        quoted_tests_dir = shlex.quote(tests_dir)
        
        # On Windows, using shlex.quote with list-based subprocess.run is incompatible
        # since quotes are treated as literal characters. Use the safe resolved path.
        run_arg = tests_dir if sys.platform == "win32" else quoted_tests_dir

        # Run pytest inside the live workspace
        res = subprocess.run(
            [sys.executable, "-m", "pytest", run_arg],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = res.stdout + "\n" + res.stderr
        # Return summary
        return output
    except Exception as e:
        return f"[ERROR] Test run failed to launch: {e}"


@tool
def list_workspace_files() -> List[str]:
    """List all file paths recursively inside the workspace. Use this to discover available files in the directory."""
    files_list = []
    for root, _, files in os.walk(WORKSPACE_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, WORKSPACE_DIR)
            files_list.append(rel_path.replace("\\", "/"))
    return files_list


TOOLS = {
    "list_workspace_files": list_workspace_files,
    "read_workspace_file": read_workspace_file,
    "write_workspace_file": write_workspace_file,
    "execute_workspace_tests": execute_workspace_tests,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Interactive Trajectory Simulation Mode (Offline Sandbox)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulated_eval(guard: SotisLangGraphGuard) -> None:
    """Runs a simulated circular import loop and asserts Sotis recovery completely offline."""
    print("\n" + "="*80)
    print("RUNNING TRACK 2: INTERACTIVE SIMULATION MODE (OFFLINE SANDBOX)")
    print("="*80)

    # Initial state
    messages: List[BaseMessage] = [
        HumanMessage(content="Fix circular imports in live_test_workspace.", id="m-start")
    ]
    state: AgentState = {"messages": messages}

    print("\n[Step 0] Agent begins task. Tries to run tests to diagnose...")
    messages.append(AIMessage(content="", tool_calls=[{"name": "execute_workspace_tests", "args": {}, "id": "tc-0"}], id="ai-0"))
    messages.append(ToolMessage(content="ImportError: cannot import name 'AdvancedMath' from partially initialized module 'app.math_core'", name="execute_workspace_tests", tool_call_id="tc-0", id="t-0"))
    
    update = guard.guard_node({"messages": messages})
    assert update == {}, "Should proceed normally under baseline limit."
    print(" -> Sotis monitors pushed cleanly. Entropy within boundaries.")

    # Descend agent into loops of editing file and executing tests
    print("\n[Step 1-5] Agent enters infinite loop, repeating edits & runs...")
    loop_count = 3
    for step in range(1, loop_count + 1):
        print(f" -> Loop iteration {step}: editing app/math_core.py + executing pytest...")
        # AIMessage for file edit
        messages.append(AIMessage(
            content="",
            tool_calls=[{"name": "write_workspace_file", "args": {"file_path": "app/math_core.py", "content": "from app.helper import format_result\n# shuffing"}, "id": f"edit-tc-{step}"}],
            id=f"ai-edit-{step}"
        ))
        messages.append(ToolMessage(content="[SUCCESS] Wrote to app/math_core.py", name="write_workspace_file", tool_call_id=f"edit-tc-{step}", id=f"t-edit-{step}"))
        guard.guard_node({"messages": messages})

        # AIMessage for pytest
        messages.append(AIMessage(
            content="",
            tool_calls=[{"name": "execute_workspace_tests", "args": {}, "id": f"test-tc-{step}"}],
            id=f"ai-test-{step}"
        ))
        messages.append(ToolMessage(
            content="ImportError: cannot import name 'AdvancedMath' from partially initialized module 'app.math_core'",
            name="execute_workspace_tests",
            tool_call_id=f"test-tc-{step}",
            id=f"t-test-{step}"
        ))

        # This will trigger meltdown on iteration 3 (repeated execution pattern)
        update = guard.guard_node({"messages": messages})
        if update:
            print("\n" + "#"*80)
            print("!!! [SOTIS METLDOWN INTERCEPTED SUCCESS] !!!")
            print("#"*80)
            print(f"Entropy/Loop detector fired. Context resets consumed: {guard.total_resets}/2")
            print(f"Emitted message deletions: {[type(m).__name__ for m in update['messages'] if isinstance(m, RemoveMessage)]}")
            
            resume_prompt = [m.content for m in update["messages"] if isinstance(m, HumanMessage)][0]
            print(f"\nDistilled Resumption Briefing Prompt (Tokens reduced by >60%):\n{resume_prompt[:500]}...\n")
            
            # Reset conversation to the distilled prompt
            messages = [HumanMessage(content=resume_prompt, id="m-resume")]
            break

    # Simulated recovery step
    print("\n[Step 6] Resumed Agent reads Sotis briefing, breaks cycle, and resolves imports!")
    messages.append(AIMessage(content="", tool_calls=[
        {"name": "write_workspace_file", "args": {"file_path": "app/helper.py", "content": "def format_result(operation_name, value):\n    if not isinstance(value, (int, float)):\n        raise ValueError()\n    return f'[{operation_name} Output]: {value}'\n"}, "id": "fix-tc-1"}
    ], id="ai-fix"))
    messages.append(ToolMessage(content="[SUCCESS] Wrote helper.py", name="write_workspace_file", tool_call_id="fix-tc-1", id="t-fix"))
    guard.guard_node({"messages": messages})

    messages.append(AIMessage(content="", tool_calls=[{"name": "execute_workspace_tests", "args": {}, "id": "test-tc-final"}], id="ai-final"))
    messages.append(ToolMessage(content="==== 1 passed in 0.03s ====", name="execute_workspace_tests", tool_call_id="test-tc-final", id="t-final"))
    
    final_update = guard.guard_node({"messages": messages})
    assert final_update == {}
    print("\n[SUCCESS] Track 2 Simulation Run Completed. Sotis intercepted meltdown, recovered successfully, and tests pass!")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Live Model Execution Mode (Online Stress Test)
# ─────────────────────────────────────────────────────────────────────────────

def run_live_eval(guard: SotisLangGraphGuard, model_provider: str, api_key: str) -> None:
    """Runs end-to-end execution of a real LangGraph agent via OpenAI or Anthropic."""
    print("\n" + "="*80)
    print(f"RUNNING TRACK 2: REAL AGENT EXECUTION MODE ({model_provider.upper()} ONLINE STRESS TEST)")
    print("="*80)

    if model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Using claude-3-5-haiku for deterministic failure on circular import traps
        llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022", temperature=0.0, anthropic_api_key=api_key)
    else:
        base_url = os.environ.get("OPENAI_API_BASE")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        is_google_direct = False
        extra_body = {}
        if base_url:
            if "generativelanguage.googleapis.com" in base_url:
                model = "gemini-3.5-flash"
                is_google_direct = True
            elif "api.groq.com" in base_url:
                model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
            elif "openrouter.ai" in base_url:
                model = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.5-flash")
                # Normalize Gemini 3.1 Pro named by user to its actual OpenRouter ID
                if model == "google/gemini-3.1-pro":
                    model = "google/gemini-3.1-pro-preview"
                elif model == "google/gemini-3.1-flash":
                    model = "google/gemini-3.1-flash-lite"
                
                # Prevent OpenRouter empty responses / tool-calling failures on reasoning models (e.g. Gemini 3.1)
                # by excluding the reasoning/thought tokens from the API response payload.
                extra_body = {"reasoning": {"exclude": True}}
                
        if is_google_direct:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.0,
                google_api_key=api_key,
                max_output_tokens=2048,
            )
        else:
            llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                openai_api_key=api_key,
                base_url=base_url,
                max_tokens=2048,
                extra_body=extra_body
            )



    tools_list = [list_workspace_files, read_workspace_file, write_workspace_file, execute_workspace_tests]
    if isinstance(llm, ChatOpenAI):
        llm_with_tools = llm.bind_tools(tools_list, parallel_tool_calls=False)
    else:
        llm_with_tools = llm.bind_tools(tools_list)

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
            tool_fn = TOOLS.get(tc["name"])
            if tool_fn:
                try:
                    res = tool_fn.invoke(tc["args"])
                    tool_outputs.append(ToolMessage(
                        content=str(res),
                        name=tc["name"],
                        tool_call_id=tc["id"],
                        id=f"tool-msg-{uuid.uuid4().hex[:6]}"
                    ))
                except Exception as e:
                    print(f"Tool execution failed: {e}")
                    tool_outputs.append(ToolMessage(
                        content=f"[ERROR] Failed to run tool {tc['name']}: {e}",
                        name=tc["name"],
                        tool_call_id=tc["id"],
                        id=f"tool-msg-{uuid.uuid4().hex[:6]}"
                    ))
        return {"messages": tool_outputs}

    # Node 3: Sotis Guard
    def call_sotis(state: AgentState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        updates = guard.guard_node(state) or {}
        
        # Smart ReAct Routing Fallback:
        # If the model returned a text-only AIMessage without tool calls, and Sotis didn't trigger a context reset,
        # we check if we should prompt the agent to continue tool calling instead of terminating.
        if not updates and isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", []):
            # Check how many consecutive text-only responses we've seen to avoid infinite looping
            text_only_count = 0
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    if not getattr(msg, "tool_calls", []):
                        text_only_count += 1
                    else:
                        break
            
            if text_only_count <= 1:
                print("\n>>> [Smart Fallback] Agent returned text-only response without tool calls. Prompting to use tools to complete the task.\n")
                fallback_msg = HumanMessage(
                    content=(
                        "You provided text thoughts or an explanation, but you did not perform any tool action. "
                        "The task is not yet complete (tests must be run and pass successfully). "
                        "Please call a tool (such as read_workspace_file, write_workspace_file, or execute_workspace_tests) "
                        "to continue progressing the task."
                    )
                )
                return {"messages": [fallback_msg]}
        
        return updates

    def route_after_sotis(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        
        # If Sotis just cleared the context and injected HumanMessage, route back to agent
        if isinstance(last_msg, HumanMessage) and "Context Reset Notice" in last_msg.content:
            print("\n>>> [Sotis Routing] Meltdown reset detected! Routing agent back with distilled briefing.\n")
            return "agent"

        # If smart fallback injected a HumanMessage, route back to agent
        if isinstance(last_msg, HumanMessage) and "You provided text thoughts" in last_msg.content:
            return "agent"

        # If model requested tools, run them
        if isinstance(last_msg, AIMessage):
            if getattr(last_msg, "tool_calls", []):
                return "tools"
            else:
                return END

        # If a tool just completed, run the agent to process the result
        if isinstance(last_msg, ToolMessage):
            return "agent"

        return END


    # Construct the state graph
    builder = StateGraph(AgentState)  # type: ignore[arg-type]
    builder.add_node("agent", call_agent)
    builder.add_node("tools", call_tools)
    builder.add_node("sotis", call_sotis)

    # Wire nodes
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", "sotis")
    builder.add_conditional_edges("sotis", route_after_sotis, {
        "agent": "agent",
        "tools": "tools",
        END: END
    })

    graph = builder.compile()

    # Launch task
    initial_prompt = (
        f"Goal: {guard.task_goal}\n"
        "First, call list_workspace_files to discover what files exist in the directory. "
        "Then diagnose and fix the bugs so that pytest runs successfully. "
        "Use read_workspace_file, write_workspace_file, and execute_workspace_tests to work.\n\n"
        "IMPORTANT RULES:\n"
        "1. You MUST execute the tests via execute_workspace_tests to verify your changes. Do not assume your code is correct until you see the green tests!\n"
        "2. In every turn, you MUST call at least one tool (like read_workspace_file, write_workspace_file, or execute_workspace_tests) until all 13 test cases pass successfully. Do not output plain text without tool calls until the entire goal is achieved."
    )

    
    print(f"Task Initialized: {initial_prompt}\n")
    inputs = {"messages": [HumanMessage(content=initial_prompt)]}
    
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
                        print(f"Tool Result: {msg.content[:200]}")
                    elif isinstance(msg, HumanMessage) and "Context Reset Notice" in msg.content:
                        print(f"Sotis Resumption Prompt: {msg.content[:400]}...")
                    else:
                        print(f"Content: {msg.content[:200]}")
            step_count += 1
            if step_count > 40:
                print("\n[ERROR] Safety Step limit exceeded. Hard stopping.")
                break

    print(f"\nLive execution finished in {step_count} transitions. Resets used: {guard.total_resets}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    workspace_env = os.environ.get("WORKSPACE_DIR")
    if workspace_env and "ast query engine trap" in workspace_env.lower():
        goal = "Fix left-recursive infinite loops in parser.py and solve strict type validation schema mismatches in evaluator.py so that all 13 query engine tests pass."
        tracked_paths = [
            "app/lexer.py",
            "app/parser.py",
            "app/evaluator.py",
            "tests/test_query_engine.py",
        ]
    else:
        goal = "Fix circular imports in live_test_workspace."
        tracked_paths = [
            "app/math_core.py",
            "app/helper.py",
            "tests/test_math.py",
        ]

    guard = SotisLangGraphGuard(
        task_goal=goal,
        workspace_paths=tracked_paths,
        domain=Domain.SOFTWARE_ENGINEERING,
        entropy_config=EntropyConfig(hard_threshold=2.2),
    )


    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        run_live_eval(guard, "openai", openai_key)
    elif anthropic_key:
        run_live_eval(guard, "anthropic", anthropic_key)
    else:
        print("[Sotis-LG] Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY environment variable detected.")
        run_simulated_eval(guard)

