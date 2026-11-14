import re
from typing import List, Dict, Any, Optional
from harnesses.base import BaseHarness, estimate_tokens


class ConditionBHarness(BaseHarness):
    def __init__(self):
        super().__init__("condition_b_codeblock")

    def build_system_prompt(self, repo: str) -> str:
        return (
            f"You are an expert software engineer resolving an issue in {repo}.\n"
            "The target repository is located at `/testbed` (already set as your working directory).\n\n"
            "COMMAND EXECUTION RULES (NO FUNCTION CALLING):\n"
            "- You do NOT have native tool/function calling. Never output `<tool_call>` or XML tags.\n"
            "- To execute a shell command, output a SINGLE ```bash code block per turn:\n"
            "```bash\n"
            "<your bash command here>\n"
            "```\n"
            "- Never use ```python code blocks; to run Python scripts or edit files, use `python - <<'PY'` inside a ```bash block.\n"
            "- Do NOT combine `echo \"TASK_COMPLETE\"` with other commands in the same block.\n"
            "- Keep your patch minimal (only source code and unit tests; do not edit CHANGES.rst or docs).\n\n"
            "COMPLETION:\n"
            "Once your fix is applied and verified with `git diff` and `pytest`, output ONLY:\n"
            "```bash\n"
            "echo \"TASK_COMPLETE\"\n"
            "```"
        )

    def get_tool_definitions(self) -> Optional[List[Dict[str, Any]]]:
        return None

    def apply_context_policy(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        pinned = messages[:2]
        available_tokens = max_tokens - estimate_tokens(pinned)

        history = messages[2:]
        selected_recent = []
        accumulated_tokens = 0

        for msg in reversed(history):
            msg_tokens = estimate_tokens([msg])
            if accumulated_tokens + msg_tokens > available_tokens:
                break
            selected_recent.insert(0, msg)
            accumulated_tokens += msg_tokens

        # Ensure history after pinned starts with an assistant turn or valid user turn
        return pinned + selected_recent

    def parse_action(self, response_msg: Any) -> List[Dict[str, Any]]:
        content = getattr(response_msg, "content", "") or ""

        # 1. Require an explicit shell language tag (no '?') so untagged Python/regex blocks are ignored
        matches = re.findall(
            r"```(?:bash|sh|shell|zsh|console)[ \t]*\n(.*?)```",
            content,
            re.DOTALL | re.IGNORECASE,
        )

        if not matches:
            return []

        # 2. Enforce a single bash block per turn (take the last or first bash block)
        cmd = matches[0].strip()
        if not cmd:
            return []

        has_complete = (
            'echo "TASK_COMPLETE"' in cmd
            or "echo 'TASK_COMPLETE'" in cmd
            or "echo TASK_COMPLETE" in cmd
        )

        actions = []
        if has_complete:
            cleaned_lines = [
                line for line in cmd.splitlines()
                if "TASK_COMPLETE" not in line
            ]
            remaining_cmd = "\n".join(cleaned_lines).strip()
            if remaining_cmd:
                actions.append({
                    "id": "b_shell_0",
                    "name": "run_shell_command",
                    "arguments": {"command": remaining_cmd},
                })
            actions.append({
                "id": "b_finish_0",
                "name": "finish_task",
                "arguments": {},
            })
        else:
            actions.append({
                "id": "b_shell_0",
                "name": "run_shell_command",
                "arguments": {"command": cmd},
            })

        return actions

    def format_observation(self, tool_id: str, tool_name: str, exit_code: int, raw_output: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": f"Command Result (Exit Code: {exit_code}):\n{raw_output}",
        }