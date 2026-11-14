import json
import argparse
from pathlib import Path
from datasets import load_dataset

from config import CONFIG
from core.agent import AgentOrchestrator
from core.llm_client import LLMClient
from harnesses.condition_a import ConditionAHarness
from harnesses.condition_b import ConditionBHarness
from harnesses.condition_c import ConditionCHarness

# Import dataset
from dataset import load_swebench_tasks

def get_harness(condition: str):
    cond = condition.lower()
    if cond == "a":
        return ConditionAHarness()
    elif cond == "b":
        return ConditionBHarness()
    elif cond == "c":
        return ConditionCHarness()
    else:
        raise ValueError(f"Unknown condition '{condition}'. Expected 'a', 'b', or 'c'.")


def run_experiment(condition: str, limit: int = 1, offset: int = 0):
    harness = get_harness(condition)
    condition_name = harness.condition_name
    
    # Set up dedicated output directories
    output_dir = Path("results") / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "predictions.jsonl"

    tasks = load_swebench_tasks(limit=limit, offset=offset)
    total_tasks = len(tasks)

    print(f"\n==========================================")
    print(f"Running benchmark condition: {condition_name}")
    print(f"Total tasks in this batch: {total_tasks}")
    print(f"Output destination: {predictions_file}")
    print(f"==========================================\n")

    llm = LLMClient()
    
    completed_ids = set()
    if predictions_file.exists():
        with open(predictions_file, "r") as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)["instance_id"])

    for idx, task in enumerate(tasks, start=1):
        instance_id = task["instance_id"]
        if instance_id in completed_ids:
            print(f"--- Skipping already completed task: {instance_id} ({idx}/{len(tasks)}) ---")
            continue

        print(f"\n--- Progress: Task[{instance_id}] {idx}/{len(tasks)} ---")
        agent = AgentOrchestrator(task=task, harness=harness, llm_client=llm)
        result = agent.run()

        # 1. Append formatted prediction for SWE-bench evaluation harness
        prediction_record = {
            "instance_id": result["instance_id"],
            "model_patch": result["model_patch"],
            "model_name_or_path": condition_name,
        }

        with open(predictions_file, "a") as f:
            f.write(json.dumps(prediction_record) + "\n")

        # 2. Persist comprehensive turn-by-turn trajectory logs
        traj_file = output_dir / f"{instance_id}_traj.json"
        result["usage_metrics"] = llm.tracker.to_dict()
        result["usage_summary"] = llm.tracker.summary()
        with open(traj_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[{instance_id}] Saved prediction and trajectory trace.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-bench Evaluation Harness Experiments")
    parser.add_argument(
        "--condition",
        type=str,
        default="a",
        choices=["a", "b", "c"],
        help="Experimental condition: 'a' (baseline), 'b' (codeblock interface), 'c' (compacted & sanitized)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of tasks to evaluate in this run",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index offset in the benchmark slice",
    )

    args = parser.parse_args()
    run_experiment(condition=args.condition, limit=args.limit, offset=args.offset)