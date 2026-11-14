import json
import re
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

# Map condition directory names to display labels and optional SWE-bench eval report paths
CONDITIONS = {
    "condition_a": {
        "label": "Condition A (JSON Tools + Sliding Window)",
        "dir": Path("condition_a"),
        "eval_report": Path("condition_a.condition_a.json"),
    },
    "condition_b_codeblock": {
        "label": "Condition B (Code-Block + Sliding Window)",
        "dir": Path("condition_b_codeblock"),
        "eval_report": Path("condition_b_codeblock.condition_b.json"),
    },
    "condition_c_compacted": {
        "label": "Condition C (JSON Tools + 2-Turn Compaction)",
        "dir": Path("condition_c_compacted"),
        "eval_report": Path("condition_c_compacted.condition_c.json"),
    },
}

USAGE_REGEX = re.compile(
    r"Calls:\s*(?P<calls>\d+)\s*\|\s*"
    r"Prompt Tokens:\s*(?P<prompt>[\d,]+)\s*\|\s*"
    r"Completion Tokens:\s*(?P<comp>[\d,]+)\s*\|\s*"
    r"Total Tokens:\s*(?P<total>[\d,]+).*?"
    r"Cache Hit:\s*(?P<hit_pct>[\d.]+)%\s*\((?P<hit_tok>[\d,]+)/(?P<in_tok>[\d,]+)\).*?"
    r"(?:Task Spend|Session Spend):\s*\$(?P<cost>[\d.]+)"
)

EDIT_PATTERNS = re.compile(
    r"(\.write_text\(|open\([^)]*['\"][wa]['\"]|sed\s+-i|cat\s+<<|patch\s+|git\s+apply)",
    re.IGNORECASE,
)


def parse_usage(traj_data: Dict[str, Any]) -> Dict[str, float]:
    """Extracts token and cost metrics from either `usage_metrics` dict or `usage_summary` string."""
    if "usage_metrics" in traj_data and isinstance(traj_data["usage_metrics"], dict):
        m = traj_data["usage_metrics"]
        return {
            "calls": int(m.get("calls", 0)),
            "prompt_tokens": int(m.get("prompt_tokens", 0)),
            "completion_tokens": int(m.get("completion_tokens", 0)),
            "cache_hit_tokens": int(m.get("cache_hit_tokens", 0)),
            "cache_hit_pct": float(m.get("cache_hit_rate_pct", 0.0)),
            "cost_usd": float(m.get("task_cost_usd", 0.0)),
        }

    summary = traj_data.get("usage_summary", "")
    match = USAGE_REGEX.search(summary)
    if not match:
        return {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cache_hit_tokens": 0, "cache_hit_pct": 0.0, "cost_usd": 0.0
        }

    return {
        "calls": int(match.group("calls")),
        "prompt_tokens": int(match.group("prompt").replace(",", "")),
        "completion_tokens": int(match.group("comp").replace(",", "")),
        "cache_hit_tokens": int(match.group("hit_tok").replace(",", "")),
        "cache_hit_pct": float(match.group("hit_pct")),
        "cost_usd": float(match.group("cost")),
    }


def analyze_trajectory_steps(trajectory: List[Dict[str, Any]], turns_taken: int) -> Dict[str, Any]:
    """Computes Time-to-First-Edit, Post-Fix Overhead, Command Count, and Amnesia Redundancy Rate."""
    shell_cmds = []
    first_edit_turn: Optional[int] = None
    last_edit_turn: Optional[int] = None

    for step in trajectory:
        if step.get("action") != "run_shell_command":
            continue
        cmd = step.get("args", {}).get("command", "").strip()
        turn = int(step.get("turn", 0))
        shell_cmds.append(cmd)

        # Detect code modification commands (excluding pure git checkout/stash resets)
        if EDIT_PATTERNS.search(cmd):
            if first_edit_turn is None:
                first_edit_turn = turn
            last_edit_turn = turn

    total_cmds = len(shell_cmds)
    # Normalize commands to detect repeated file inspections (Exploration Amnesia)
    normalized_cmds = [re.sub(r"\s+", " ", c) for c in shell_cmds]
    unique_cmds = len(set(normalized_cmds))
    duplicate_cmds = total_cmds - unique_cmds
    redundancy_pct = (duplicate_cmds / total_cmds * 100.0) if total_cmds > 0 else 0.0

    post_fix_overhead = (turns_taken - last_edit_turn) if last_edit_turn is not None else None

    return {
        "total_cmds": total_cmds,
        "first_edit_turn": first_edit_turn,
        "last_edit_turn": last_edit_turn,
        "post_fix_overhead": post_fix_overhead,
        "redundancy_pct": redundancy_pct,
    }


def analyze_patch_diff(patch: str) -> Dict[str, int]:
    """Counts files touched and total added/removed lines in model_patch."""
    if not patch or not patch.strip():
        return {"non_empty": 0, "files_touched": 0, "lines_changed": 0, "touched_docs": 0}

    files = re.findall(r"^diff --git a/(.*?) b/", patch, re.MULTILINE)
    added = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    touched_docs = int(any("CHANGES" in f or f.startswith("docs/") for f in files))

    return {
        "non_empty": 1,
        "files_touched": len(files),
        "lines_changed": added + removed,
        "touched_docs": touched_docs,
    }


def load_resolved_ids(report_path: Path) -> Optional[Set[str]]:
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text())
    return set(data.get("resolved_ids", []))


def main():
    per_task_rows = []
    summary_rows = []

    for cond_key, cfg in CONDITIONS.items():
        traj_dir: Path = cfg["dir"]
        if not traj_dir.exists():
            print(f"[Skip] Directory not found: {traj_dir}")
            continue

        resolved_set = load_resolved_ids(cfg["eval_report"])
        traj_files = sorted(traj_dir.glob("*_traj*.json"))
        if not traj_files:
            continue

        tasks = []
        for tf in traj_files:
            data = json.loads(tf.read_text())
            instance_id = data.get("instance_id", tf.stem)
            turns_taken = int(data.get("turns_taken", 0))
            finished = int(bool(data.get("finished_by_tool", False)))
            patch = data.get("model_patch", "")

            usage = parse_usage(data)
            step_stats = analyze_trajectory_steps(data.get("trajectory", []), turns_taken)
            patch_stats = analyze_patch_diff(patch)
            is_resolved = (1 if instance_id in resolved_set else 0) if resolved_set is not None else None

            row = {
                "condition": cond_key,
                "instance_id": instance_id,
                "resolved": is_resolved,
                "non_empty_patch": patch_stats["non_empty"],
                "finished_by_tool": finished,
                "turns_taken": turns_taken,
                "total_cmds": step_stats["total_cmds"],
                "first_edit_turn": step_stats["first_edit_turn"],
                "post_fix_overhead": step_stats["post_fix_overhead"],
                "redundancy_pct": round(step_stats["redundancy_pct"], 2),
                "files_touched": patch_stats["files_touched"],
                "lines_changed": patch_stats["lines_changed"],
                "touched_docs": patch_stats["touched_docs"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cache_hit_tokens": usage["cache_hit_tokens"],
                "cache_hit_pct": round(usage["cache_hit_pct"], 2),
                "cost_usd": round(usage["cost_usd"], 5),
            }
            tasks.append(row)
            per_task_rows.append(row)

        n = len(tasks)
        total_prompt = sum(t["prompt_tokens"] for t in tasks)
        total_hit = sum(t["cache_hit_tokens"] for t in tasks)
        total_cost = sum(t["cost_usd"] for t in tasks)
        edit_turns = [t["first_edit_turn"] for t in tasks if t["first_edit_turn"] is not None]
        overheads = [t["post_fix_overhead"] for t in tasks if t["post_fix_overhead"] is not None]
        resolved_count = sum(t["resolved"] for t in tasks if t["resolved"] is not None) if resolved_set is not None else None

        summary_rows.append({
            "Condition": cfg["label"],
            "N": n,
            "Resolved (%)": f"{(resolved_count / n * 100):.1f}%" if resolved_count is not None else "Pending Eval",
            "Patch Gen (%)": f"{(sum(t['non_empty_patch'] for t in tasks) / n * 100):.1f}%",
            "Graceful Finish (%)": f"{(sum(t['finished_by_tool'] for t in tasks) / n * 100):.1f}%",
            "Avg Turns": f"{(sum(t['turns_taken'] for t in tasks) / n):.1f}",
            "Avg Cmds": f"{(sum(t['total_cmds'] for t in tasks) / n):.1f}",
            "Avg T_edit": f"{(sum(edit_turns) / len(edit_turns)):.1f}" if edit_turns else "N/A",
            "Post-Fix Overhead": f"{(sum(overheads) / len(overheads)):.1f}" if overheads else "N/A",
            "Cmd Redundancy (%)": f"{(sum(t['redundancy_pct'] for t in tasks) / n):.1f}%",
            "Avg Diff Lines": f"{(sum(t['lines_changed'] for t in tasks) / n):.1f}",
            "Avg Prompt Tok": f"{int(total_prompt / n):,}",
            "Weighted Cache Hit (%)": f"{(total_hit / total_prompt * 100):.1f}%" if total_prompt > 0 else "0.0%",
            "Avg Cost/Task ($)": f"${(total_cost / n):.4f}",
            "Cost/Resolved ($)": f"${(total_cost / resolved_count):.4f}" if resolved_count else "N/A",
        })

    # Save detailed per-task CSV for paired statistical tests (Wilcoxon / McNemar)
    if per_task_rows:
        with open("benchmark_per_task_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_task_rows)
        print("Saved per-task telemetry to: benchmark_per_task_metrics.csv\n")

    # Print Markdown comparison table
    if summary_rows:
        headers = list(summary_rows[0].keys())
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in summary_rows:
            print("| " + " | ".join(str(r[h]) for h in headers) + " |")


if __name__ == "__main__":
    main()