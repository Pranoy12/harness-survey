from typing import List, Dict, Any, Optional
from harnesses.base import BaseHarness, estimate_tokens

class ConditionAHarness(BaseHarness):
    def __init__(self):
        super().__init__("condition_a")

    def build_system_prompt(self, repo: str) -> str:
        return (
            f"You are an expert software engineer resolving an issue in {repo}.\n"
            "Use the provided tools to inspect files, execute commands, and run tests.\n"
            "When the bug is identified, modify the code and verify your fix with tests.\n"
            "Call 'finish_task' once you have resolved the issue and verified the patch. \n"
            "The target repository is located at `/testbed` (already set as your working directory)."
        )

    def get_tool_definitions(self) -> Optional[List[Dict[str, Any]]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Execute a shell command inside the repository sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "description": "Signal that the issue is fixed and tested.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def apply_context_policy(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        pinned = messages[:2]
        pinned_tokens = estimate_tokens(pinned)
        available_tokens = max_tokens - pinned_tokens

        history = messages[2:]
        selected_recent = []
        accumulated_tokens = 0

        # Walk backwards
        for msg in reversed(history):
            msg_tokens = estimate_tokens([msg])
            if accumulated_tokens + msg_tokens > available_tokens:
                break
            selected_recent.insert(0, msg)
            accumulated_tokens += msg_tokens

        # SANITIZATION: If the first message in selected_recent is a dangling 'tool' message,
        # drop it until the history begins with an 'assistant' or 'user' turn
        while selected_recent and selected_recent[0].get("role") == "tool":
            selected_recent.pop(0)

        # SANITIZATION: If the last message in selected_recent has tool_calls without its tool response,
        # ensure it's valid or keep the pair intact
        # Prevent orphaned 'tool' messages at the start of the sliding window
        while selected_recent and selected_recent[0].get("role") == "tool":
            selected_recent.pop(0)

        return pinned + selected_recent

    def parse_action(self, response_msg: Any) -> List[Dict[str, Any]]:
        import json
        actions = []
        if getattr(response_msg, "tool_calls", None):
            for tc in response_msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {"command": args}
                elif not isinstance(args, dict):
                    args = {}

                actions.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args
                })
        return actions

    def format_observation(self, tool_id: str, tool_name: str, exit_code: int, raw_output: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Exit Code: {exit_code}\nOutput:\n{raw_output}"
        }