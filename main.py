import asyncio
import json
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict

from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

load_dotenv()


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=1000, tools=tools, messages=messages
        )

        has_tool_use = False
        tool_results = []
        submitted_answer = None


        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input
                    if tool_name == "python_expression":
                        if (
                            not isinstance(tool_input, dict)
                            or "expression" not in tool_input
                            or not tool_input.get("expression")
                        ):
                            result = {
                                "result": None,
                                "error": (
                                    "Invalid tool input. You must provide a JSON object "
                                    "with a non-empty 'expression' key."
                                ),
                            }
                        else:
                            if verbose:
                                print("\nInput:")
                                print("```")
                                for line in tool_input["expression"].split("\n"):
                                    print(f"{line}")
                                print("```")
                            result = handler(tool_input["expression"])
                            if verbose:
                                print("\nOutput:")
                                print("```")
                                print(result)
                                print("```")
                    elif tool_name == "submit_answer":
                        if not isinstance(tool_input, dict) or "answer" not in tool_input:
                            result = {
                                "answer": None,
                                "submitted": False,
                                "error": (
                                    "Invalid tool input. You must provide a JSON object "
                                    "with an 'answer' key."
                                ),
                            }
                        else:
                            answer_value = tool_input["answer"]
                            if isinstance(answer_value, str):
                                import ast
                                try:
                                    answer_value = ast.literal_eval(answer_value)
                                except (ValueError, SyntaxError):
                                    pass
                            result = handler(answer_value)
                            submitted_answer = result["answer"]
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer: Any,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    try:
        result = await run_agent_loop(
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            max_steps=15,
            verbose=verbose,
        )
        
        if verbose:
            print(f"\nDebug - Result type: {type(result)}, Expected type: {type(expected_answer)}")
            if result is not None:
                print(f"Debug - Result repr: {repr(result)}")
                print(f"Debug - Expected repr: {repr(expected_answer)}")

        success = result == expected_answer
        
        if success:
            print(f"✓ Run {run_id}: SUCCESS - Got {result}")
        else:
            print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")
        
        return run_id, success, result

    except Exception as e:
        print(f"✗ Run {run_id}: CRASHED - {type(e).__name__}: {e}")
        return run_id, False, f"CRASH: {e}"

async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Will be passed to exec(). Use print() to output something. "
                            "Returns stdout. This tool is STATELESS. You MUST import all "
                            "required libraries (e.g., numpy) inside the expression."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # --- Task-Specific Data and Prompt ---
    in_prompt_data = [
        {"sample_id": 0, "view1": [0.2, 0.4], "view2": [0.3, 0.5], "label": [1, 0, 0]},
        {"sample_id": 1, "view1": [0.6, 0.8], "view2": [0.7, 0.9], "label": [0, 1, 0]},
        {"sample_id": 2, "view1": [0.1, 0.3], "view2": [0.2, 0.4], "label": [0, 0, 1]},
        {"sample_id": 3, "view1": [0.4, 0.6], "view2": [0.5, 0.7], "label": [1, 0, 0]},
    ]

    expected_answer = {
        "mixed_view1": [0.53281, 0.84623],
        "mixed_view2": [0.56653, 0.82404],
        "mixed_label": [0.36, 0.36, 0.28],
    }


    prompt = f"""
        You are an ML engineer implementing **Conditional Co-Mixup** with adaptive parameters.

        **Background:**
        Standard mixup: $\\tilde{{x}} = \\lambda x_i + (1 - \\lambda) x_j$

        **Your Task - Two-Stage Conditional Mixup:**

        **Input Data:**
        `{in_prompt_data}`

        **Stage 1 - Initial Pairwise Mixing:**
        1.  **Result A:** Mix **sample 0** with **sample 1**, using $\\lambda = 0.4$.
        2.  **Result B:** Mix **sample 2** with **sample 3**, using $\\lambda = 0.7$.

        **Stage 2 - Conditional Final Mix:**
        3.  Calculate L2 norm of Result A's view1: $||view1_A||_2 = \\sqrt{{\\sum_i v_i^2}}$
        4.  **CRITICAL DECISION:**
            * **IF** $||view1_A||_2 > 0.8$: Use $\\lambda = 0.3$ for final mix
            * **ELSE**: Use $\\lambda = 0.6$ for final mix
        5.  **Final Result:** Mix **Result A** with **Result B** using the determined λ.

        **Stage 3 - Post-Processing:**
        6.  Normalize BOTH view vectors to unit L2 norm
        7.  For mixed_label: **IF** any component > 0.5:
            * Apply smoothing: $label_{{final}} = 0.9 \\times label + 0.1 \\times [0.33, 0.33, 0.34]$
            * **ELSE**: Keep original mixed_label

        **Critical Requirements:**
        * The conditional check in Stage 2 is MANDATORY
        * Round all floats to **5 decimal places**
        * Ensure label sums to 1.0 (±0.001)

        **Your Deliverable:**
        Use `python_expression` for calculations.
        Submit with `submit_answer` in the format:
        `{{"mixed_view1": [x1, x2], "mixed_view2": [x3, x4], "mixed_label": [y1, y2, y3]}}`
        """

    # --- Run the Test ---
    num_runs = 10
    execution_mode = "concurrently"
    
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print(f"Task: Implement 'Conditional Co-Mixup' (Deterministic)")
    print(f"Expected Answer: {expected_answer}")
    print("=" * 60)

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    results = []
    for task in tasks:
        result = await task
        results.append(result)

    successes = sum(1 for _, success, _ in results if success)
    pass_rate = (successes / num_runs)* 100

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main(concurrent=True))