from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Fast character-based token approximation (1 token ≈ 4 chars)."""
    total_chars = 0
    for m in messages:
        content = m.get("content") or ""
        total_chars += len(content)
        if "tool_calls" in m and m["tool_calls"]:
            for tc in m["tool_calls"]:
                total_chars += len(str(tc.get("function", {})))
    return total_chars // 4

class BaseHarness(ABC):
    def __init__(self, condition_name: str):
        self.condition_name = condition_name

    @abstractmethod
    def build_system_prompt(self, repo: str) -> str:
        """Returns the system instruction for the LLM."""
        pass

    @abstractmethod
    def get_tool_definitions(self) -> Optional[List[Dict[str, Any]]]:
        """Returns tool/function schemas, or None if raw text/markdown is used."""
        pass

    @abstractmethod
    def apply_context_policy(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Handles context retention, compaction, or sliding window."""
        pass

    @abstractmethod
    def parse_action(self, response_msg: Any) -> List[Dict[str, Any]]:
        """Extracts executable actions (name, arguments, id) from LLM output."""
        pass

    @abstractmethod
    def format_observation(self, tool_id: str, tool_name: str, exit_code: int, raw_output: str) -> Dict[str, Any]:
        """Formats container execution output back into the conversation."""
        pass