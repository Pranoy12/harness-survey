import json
from typing import List, Dict, Any, Optional
from harnesses.base import BaseHarness, estimate_tokens

class ConditionCHarness(BaseHarness):
    def __init__(self):
        super().__init__("condition_c_compacted")

    def build_system_prompt(self, repo: str) -> str:
        return (
            f"You are an expert software engineer resolving an issue in {repo}.\n"
            "Use the provided tools to inspect, modify, and verify code.\n"
            "Call 'finish_task' once the issue is completely resolved.\n"
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
        if len(messages) <= 2:
            return messages

        pinned = messages[:2]
        history = messages[2:]

        # Group history into atomic turns: each turn starts at 'assistant' (or 'user' nudge)
        # and includes all immediately following 'tool' messages.
        turn_groups: List[List[Dict[str, Any]]] = []
        for msg in history:
            if msg.get("role") == "tool" and turn_groups:
                turn_groups[-1].append(msg)
            else:
                turn_groups.append([msg])

        # Condition C Compaction: Keep only the last 2 atomic turns (e.g., last 2 assistant+tool groups)
        recent_groups = turn_groups[-2:]
        compacted_tail = [m for group in recent_groups for m in group]

        # Safety guard: drop any leading orphaned 'tool' messages if they ever occur
        while compacted_tail and compacted_tail[0].get("role") == "tool":
            compacted_tail.pop(0)

        return pinned + compacted_tail

    def parse_action(self, response_msg: Any) -> List[Dict[str, Any]]:
        actions = []
        if getattr(response_msg, "tool_calls", None):
            for tc in response_msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {"command": tc.function.arguments}
                actions.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args
                })
        return actions

    def format_observation(self, tool_id: str, tool_name: str, exit_code: int, raw_output: str) -> Dict[str, Any]:
        # Condition C: Sanitizes error output to simulate noisy/truncated feedback
        sanitized_output = raw_output
        if exit_code != 0 and raw_output.strip():
            lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
            sanitized_output = "[Sanitized Error (Last 3 lines)]:\n" + "\n".join(lines[-3:])

        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Exit Code: {exit_code}\nOutput:\n{sanitized_output}"
        }