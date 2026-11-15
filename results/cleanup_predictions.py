import json
from pathlib import Path

results_root = Path("./")

for cond_dir in results_root.iterdir():
    if not cond_dir.is_dir():
        continue
    
    traj_files = sorted(cond_dir.glob("*_traj.json"))
    if not traj_files:
        continue

    preds_path = cond_dir / "predictions.jsonl"
    count = 0
    with open(preds_path, "w", encoding="utf-8") as out_f:
        for tf in traj_files:
            try:
                data = json.loads(tf.read_text(encoding="utf-8"))
                pred = {
                    "instance_id": data["instance_id"],
                    "model_name_or_path": data.get("condition", cond_dir.name),
                    "model_patch": data.get("model_patch") or ""
                }
                out_f.write(json.dumps(pred) + "\n")
                count += 1
            except Exception as e:
                print(f"Skipping {tf.name}: {e}")
                
    print(f"Wrote {count} predictions to {preds_path}")