import json
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset

OUTPUT_FILE = Path("swe_bench_tasks.json")

# Target quotas across lightweight/stable repos in SWE-bench Lite
TARGET_QUOTAS = {
    "psf/requests": 6,          # All available in Lite
    "pallets/flask": 3,         # All available in Lite
    # "pytest-dev/pytest": 17,    # All available in Lite
    # "pylint-dev/pylint": 6,     # All available in Lite
    # "mwaskom/seaborn": 4,       # All available in Lite
    # "django/django": 14         # Lightweight slice (forms, utils, validators)
}


def build_and_cache_dataset(output_path: Path = OUTPUT_FILE) -> List[Dict[str, Any]]:
    """
    Downloads SWE-bench Lite, extracts 50 deterministic tasks matching quotas,
    and caches them to disk. Returns the extracted list of tasks.
    """
    print("Fetching SWE-bench Lite dataset metadata from Hugging Face...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Deterministic sort before quota filtering ensures reproducibility
    sorted_ds = sorted(list(ds), key=lambda x: x["instance_id"])

    selected_tasks: List[Dict[str, Any]] = []
    counts = {repo: 0 for repo in TARGET_QUOTAS}

    for item in sorted_ds:
        # print(f"Keys -> {list(item.keys())}")
        repo = item["repo"]
        if repo in TARGET_QUOTAS and counts[repo] < TARGET_QUOTAS[repo]:
            selected_tasks.append({
                "instance_id": item["instance_id"],
                "repo": item["repo"],
                "version": item["version"],
                "base_commit": item["base_commit"],
                "problem_statement": item["problem_statement"],
                "hints_text": item.get("hints_text", ""),
                "fail_to_pass": item.get("FAIL_TO_PASS", []),
                "pass_to_pass": item.get("PASS_TO_PASS", []),
                "environment_setup_commit": item.get("environment_setup_commit", "")
            })
            counts[repo] += 1

    total_selected = len(selected_tasks)
    print(f"\nExtracted {total_selected} tasks successfully:")
    for repo, count in counts.items():
        print(f"  • {repo:20s}: {count}/{TARGET_QUOTAS[repo]}")

    expected = sum(TARGET_QUOTAS.values())
    if total_selected != expected:
        print(f"\n[Warning] Collected {total_selected} tasks instead of {expected}.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_tasks, f, indent=2)

    print(f"\nSaved benchmark dataset slice to: {output_path.resolve()}")
    return selected_tasks

def load_swebench_tasks(
    limit: int = 10,
    offset: int = 0,
    force_refresh: bool = False,
    cache_path: Path = OUTPUT_FILE
) -> List[Dict[str, Any]]:
    """
    Loads a deterministic slice of SWE-bench Lite tasks.
    Reads locally from cache if available.
    Downloads, extracts quotas, and caches once on the first run.
    """
    # 1. Check if the local cache exists or if refresh requested
    if not cache_path.exists() or force_refresh:
        all_tasks = build_and_cache_dataset(output_path=cache_path)
    else:
        print(f"Loading tasks from local cache: {cache_path.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)

    # 2. Slice and return the requested subset
    return all_tasks[offset : offset + limit]

if __name__ == "__main__":
    build_and_cache_dataset()