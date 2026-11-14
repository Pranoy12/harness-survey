from typing import Dict, Any
from config import CONFIG
from core.sandbox import DockerSandbox
from core.llm_client import LLMClient
from harnesses.base import BaseHarness


class AgentOrchestrator:
    def __init__(self, task: Dict[str, Any], harness: BaseHarness, llm_client: LLMClient):
        self.task = task
        self.harness = harness
        self.llm = llm_client
        self.instance_id = task["instance_id"]
        self.repo = task["repo"]
        self.problem_statement = task["problem_statement"]
        self.sandbox = DockerSandbox(self.instance_id)

    def run(self) -> Dict[str, Any]:
        print("=" * 100)
        print(f"Starting Task: {self.instance_id} [{self.harness.condition_name}]")
        print(f"Repository: {self.repo}")
        print(f"Problem Statement: {self.problem_statement}")
        print("=" * 100)
        print("\n\n")
        
        self.llm.tracker.reset_task()

        # 1. Start Docker container sandbox
        self.sandbox.start()

        # 2. Initialize conversation history
        system_prompt = self.harness.build_system_prompt(self.repo)
        tools = self.harness.get_tool_definitions()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Please solve this issue in repository {self.repo}:\n\n"
                    f"{self.problem_statement}\n\n"
                    "Make sure to inspect the files, locate the problem, reproduce it, and modify the code to fix it."
                )
            }
        ]

        turns_taken = 0
        finished = False
        trajectory = []
        final_patch = ""

        try:
            while turns_taken < CONFIG.MAX_AGENT_TURNS and not finished:
                turns_taken += 1
                print(f"[{self.instance_id}] Turn {turns_taken}/{CONFIG.MAX_AGENT_TURNS}...")

                # Apply context policy (sliding window vs compaction)
                messages = self.harness.apply_context_policy(messages, CONFIG.MAX_CONTEXT_TOKENS)

                # Query LLM
                raw_response = self.llm.complete(messages=messages, tools=tools)

                # Extract message choice from ModelResponse
                response_msg = raw_response.choices[0].message

                # Format assistant turn back into history
                assistant_record = {
                    "role": "assistant",
                    "content": response_msg.content or ""
                }
                if getattr(response_msg, "tool_calls", None):
                    assistant_record["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response_msg.tool_calls
                    ]
                messages.append(assistant_record)

                # Parse actions
                actions = self.harness.parse_action(response_msg)

                if not actions:
                    # Condition-aware nudge when no action was parsed
                    if tools is None:
                        nudge = (
                            "No executable bash code block was found in your response. "
                            "Output a ```bash\n<command>\n``` block to run a command, or "
                            "```bash\necho \"TASK_COMPLETE\"\n``` if the fix is applied and verified."
                        )
                    else:
                        nudge = (
                            "Please invoke `run_shell_command` to execute commands "
                            "or call `finish_task` when complete."
                        )
                    messages.append({"role": "user", "content": nudge})
                    continue

                for action in actions:
                    fn_name = action["name"]
                    fn_args = action["arguments"]
                    tool_id = action.get("id", "call_default")

                    trajectory.append({
                        "turn": turns_taken,
                        "action": fn_name,
                        "args": fn_args
                    })

                    if fn_name == "finish_task":
                        print(f"[{self.instance_id}] finish_task called by model.")
                        finished = True
                        break

                    elif fn_name == "run_shell_command":
                        cmd = fn_args.get("command", "")
                        print(f"[{self.instance_id}] $ {cmd[:80]}")
                        exit_code, raw_out = self.sandbox.exec_cmd(cmd)

                        # Truncate raw logs over 10k chars to prevent runaway payloads
                        if len(raw_out) > 10000:
                            raw_out = raw_out[:10000] + "\n...[truncated output]..."

                        obs = self.harness.format_observation(
                            tool_id=tool_id,
                            tool_name=fn_name,
                            exit_code=exit_code,
                            raw_output=raw_out
                        )

                        # Inject urgency reminder on the final 4 turns
                        if turns_taken >= CONFIG.MAX_AGENT_TURNS - 4:
                            remaining = CONFIG.MAX_AGENT_TURNS - turns_taken
                            obs["content"] += (
                                f"\n\n[System Notice: {remaining} turn(s) remaining "
                                f"({turns_taken}/{CONFIG.MAX_AGENT_TURNS}). "
                                "Do not edit docs or changelogs. Verify your fix with git diff and finish immediately.]"
                            )

                        messages.append(obs)

                    else:
                        # Guard against hallucinated tool names so tool_call_id stays paired
                        obs = self.harness.format_observation(
                            tool_id=tool_id,
                            tool_name=fn_name,
                            exit_code=1,
                            raw_output=f"Error: Unknown tool '{fn_name}'. Only use 'run_shell_command' or 'finish_task'."
                        )
                        messages.append(obs)

            # Retrieve final patch diff from repository git state
            final_patch = self.sandbox.get_git_diff()
            print(f"[{self.instance_id}] Extracted patch length: {len(final_patch)} chars")

        finally:
            print("=" * 100)
            self.sandbox.cleanup()
            print("=" * 100)
            print("\n\n")

        return {
            "instance_id": self.instance_id,
            "condition": self.harness.condition_name,
            "turns_taken": turns_taken,
            "finished_by_tool": finished,
            "model_patch": final_patch,
            "trajectory": trajectory,
            "usage_metrics": self.llm.tracker.to_dict()
        }