import argparse
import json
from pathlib import Path
from datasets import load_dataset
from core.sandbox import DockerSandbox


def load_task_metadata(local_cache: Path) -> dict:
    """Loads fail_to_pass, pass_to_pass, and test_patch from cache or HuggingFace."""
    tasks = {}
    if local_cache.exists():
        raw_list = json.loads(local_cache.read_text(encoding="utf-8"))
        for item in raw_list:
            tasks[item["instance_id"]] = item

    sample = next(iter(tasks.values()), {})
    if "test_patch" not in sample or len(tasks) < 15:
        print("Fetching official dataset definitions from princeton-nlp/SWE-bench_Lite...")
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        for row in ds:
            iid = row["instance_id"]
            if iid in tasks:
                tasks[iid]["test_patch"] = row["test_patch"]
                tasks[iid]["FAIL_TO_PASS"] = row.get("FAIL_TO_PASS", tasks[iid].get("fail_to_pass", "[]"))
                tasks[iid]["PASS_TO_PASS"] = row.get("PASS_TO_PASS", tasks[iid].get("pass_to_pass", "[]"))
            else:
                tasks[iid] = dict(row)
        if local_cache.exists():
            local_cache.write_text(json.dumps(list(tasks.values()), indent=2), encoding="utf-8")

    return tasks


def sync_predictions_from_trajectories(preds_path: Path) -> dict:
    """Loads predictions.jsonl and backfills any completed *_traj.json files in the same folder."""
    latest_preds = {}
    if preds_path.exists():
        for line in preds_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest_preds[row["instance_id"]] = row

    # Backfill from *_traj.json in the same directory if any trajectory wasn't appended
    # or if predictions.jsonl had an empty patch that was later completed
    cond_dir = preds_path.parent
    for tf in sorted(cond_dir.glob("*_traj.json")):
        data = json.loads(tf.read_text(encoding="utf-8"))
        iid = data.get("instance_id")
        traj_patch = data.get("model_patch", "")
        if iid and (iid not in latest_preds or (not latest_preds[iid].get("model_patch") and traj_patch)):
            latest_preds[iid] = {
                "instance_id": iid,
                "model_patch": traj_patch,
                "model_name_or_path": data.get("condition", cond_dir.name),
            }

    # Write clean, deduplicated predictions.jsonl back to disk
    sorted_preds = dict(sorted(latest_preds.items()))
    preds_path.write_text(
        "\n".join(json.dumps(r) for r in sorted_preds.values()) + "\n",
        encoding="utf-8",
    )
    return sorted_preds


def load_existing_results(report_path: Path, include_errors: bool = False) -> dict:
    """Reads an existing output_report JSON and maps instance_id -> status."""
    existing = {}
    if not report_path.exists():
        return existing

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        for iid in data.get("resolved_ids", []):
            existing[iid] = "resolved"
        for iid in data.get("unresolved_ids", []):
            existing[iid] = "unresolved"
        if include_errors:
            for iid in data.get("error_ids", []):
                existing[iid] = "error"
            for iid in data.get("empty_patch_ids", []):
                existing[iid] = "empty_patch"
    except Exception as e:
        print(f"[Warning] Could not read existing report at {report_path}: {e}")

    return existing


def parse_test_list(task_meta: dict, key_lower: str) -> list:
    raw_val = task_meta.get(key_lower) or task_meta.get(key_lower.upper(), "[]")
    if isinstance(raw_val, list):
        return raw_val
    if isinstance(raw_val, str) and raw_val.strip():
        return json.loads(raw_val)
    return []


def evaluate_instance(instance_id: str, model_patch: str, task_meta: dict) -> str:
    """Returns 'empty_patch', 'resolved', 'unresolved', or 'error'."""
    if not model_patch or not model_patch.strip():
        print(f"[{instance_id}] Empty model_patch -> EMPTY_PATCH ⚠️")
        return "empty_patch"

    fail_to_pass = parse_test_list(task_meta, "fail_to_pass")
    pass_to_pass = parse_test_list(task_meta, "pass_to_pass")
    test_patch = task_meta.get("test_patch", "")

    sandbox = DockerSandbox(instance_id)
    try:
        sandbox.start()

        # 1. Apply model_patch inside /testbed
        sandbox.exec_cmd(
            f"python - <<'PY'\n"
            f"from pathlib import Path\n"
            f"Path('/tmp/model.patch').write_text({repr(model_patch + chr(10))})\n"
            f"PY"
        )
        code, out = sandbox.exec_cmd(
            "git apply -v /tmp/model.patch || patch --batch --fuzz=5 -p1 -i /tmp/model.patch"
        )
        if code != 0:
            print(f"[{instance_id}] Patch failed to apply -> UNRESOLVED ❌\n{out[-400:]}")
            return "unresolved"

        # 2. Reset any agent-modified test files and apply official SWE-bench test_patch
        if test_patch and test_patch.strip():
            patched_test_files = [
                line.split(" b/")[1].strip()
                for line in test_patch.splitlines()
                if line.startswith("diff --git ") and " b/" in line
            ]
            if patched_test_files:
                files_str = " ".join(patched_test_files)
                sandbox.exec_cmd(f"git checkout -- {files_str}")

            sandbox.exec_cmd(
                f"python - <<'PY'\n"
                f"from pathlib import Path\n"
                f"Path('/tmp/test.patch').write_text({repr(test_patch + chr(10))})\n"
                f"PY"
            )
            t_code, t_out = sandbox.exec_cmd(
                "git apply -v /tmp/test.patch || patch --batch --fuzz=5 -p1 -i /tmp/test.patch"
            )
            if t_code != 0:
                print(f"[{instance_id}] Official test_patch failed to apply -> ERROR ⚠️\n{t_out[-400:]}")
                return "error"

        # 3. Grade FAIL_TO_PASS tests (override filterwarnings=error so warnings don't fail tests)
        if not fail_to_pass:
            print(f"[{instance_id}] No FAIL_TO_PASS tests listed -> ERROR ⚠️")
            return "error"

        quoted_f2p = " ".join(f'"{t}"' for t in fail_to_pass)
        f2p_cmd = f"pytest -q -o filterwarnings=ignore -W ignore {quoted_f2p}"
        f2p_code, f2p_out = sandbox.exec_cmd(f2p_cmd)

        if f2p_code != 0:
            print(f"[{instance_id}] FAIL_TO_PASS FAILED (exit={f2p_code}) -> UNRESOLVED ❌")
            print(f2p_out[-500:])
            return "unresolved"

        # 4. Grade PASS_TO_PASS tests (skip external httpbin.org network tests on legacy psf/requests)
        if pass_to_pass and not instance_id.startswith("psf__requests-"):
            quoted_p2p = " ".join(f'"{t}"' for t in pass_to_pass)
            p2p_cmd = f"pytest -q -o filterwarnings=ignore -W ignore {quoted_p2p}"
            p2p_code, p2p_out = sandbox.exec_cmd(p2p_cmd)
            if p2p_code != 0:
                print(f"[{instance_id}] PASS_TO_PASS REGRESSION (exit={p2p_code}) -> UNRESOLVED ❌")
                print(p2p_out[-500:])
                return "unresolved"

        print(f"[{instance_id}] ALL TESTS PASSED ({len(fail_to_pass)} F2P) -> RESOLVED ✅")
        return "resolved"

    except Exception as e:
        print(f"[{instance_id}] Container/Runtime Exception -> ERROR ⚠️: {e}")
        return "error"
    finally:
        sandbox.cleanup()


def write_report(
    report_path: Path,
    target_ids: list,
    submitted_ids: list,
    results_by_id: dict,
    incomplete_ids: list,
) -> dict:
    """Builds and writes the evaluation report JSON to disk."""
    completed_ids = [iid for iid, s in results_by_id.items() if s in ("resolved", "unresolved")]
    resolved_ids = [iid for iid, s in results_by_id.items() if s == "resolved"]
    unresolved_ids = [iid for iid, s in results_by_id.items() if s == "unresolved"]
    empty_patch_ids = [iid for iid, s in results_by_id.items() if s == "empty_patch"]
    error_ids = [iid for iid, s in results_by_id.items() if s == "error"]

    report = {
        "total_instances": len(target_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "submitted_ids": submitted_ids,
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
        "schema_version": 2,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--output_report", type=str, required=True)
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        default=None,
        help="Optional list of specific instance_ids to evaluate (matches official swebench CLI)",
    )
    parser.add_argument(
        "--skip_evaluated",
        action="store_true",
        help="Skip instances that are already marked as resolved or unresolved in output_report",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="When used with --skip_evaluated, also skip instances previously marked as error or empty_patch",
    )
    args = parser.parse_args()

    tasks_meta = load_task_metadata(Path("swe_bench_tasks.json"))
    preds_path = Path(args.predictions_path)
    report_path = Path(args.output_report)

    # Sync and deduplicate predictions.jsonl with all *_traj.json files in that folder
    all_preds = sync_predictions_from_trajectories(preds_path)
    submitted_ids = sorted(all_preds.keys())

    # Filter to requested --instance_ids if provided, otherwise evaluate all submitted_ids
    if args.instance_ids:
        target_ids = [iid for iid in args.instance_ids if iid in all_preds]
    else:
        target_ids = submitted_ids

    # Load previously evaluated results if --skip_evaluated is enabled
    results_by_id = {}
    if args.skip_evaluated:
        results_by_id = load_existing_results(report_path, include_errors=args.skip_errors)
        if results_by_id:
            print(f"Loaded {len(results_by_id)} previously evaluated instance(s) from {report_path}")

    incomplete_ids = []

    for iid in target_ids:
        if iid not in tasks_meta:
            print(f"[Skip] {iid} not found in dataset metadata.")
            incomplete_ids.append(iid)
            continue

        if args.skip_evaluated and iid in results_by_id:
            print(f"[Skip] {iid} already evaluated ({results_by_id[iid].upper()}) ⏭️")
            continue

        patch = all_preds[iid].get("model_patch", "")
        status = evaluate_instance(iid, patch, tasks_meta[iid])
        results_by_id[iid] = status

        # Save report incrementally after each instance so progress is never lost
        write_report(report_path, target_ids, submitted_ids, results_by_id, incomplete_ids)

    report = write_report(report_path, target_ids, submitted_ids, results_by_id, incomplete_ids)
    print(f"\nReport written to {args.output_report}:")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()