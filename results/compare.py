import json
from datasets import load_dataset

# 1. Load your trajectory file
with open("./condition_a/pallets__flask-4992_traj.json") as f:
    traj = json.load(f)

# 2. Load the official SWE-bench Lite dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
row = [r for r in ds if r["instance_id"] == "pallets__flask-4045"][0]

print("=== YOUR MODEL PATCH ===")
print(traj["model_patch"])

print("\n=== OFFICIAL GOLD PATCH ===")
print(row["patch"])

print("\n=== HIDDEN EVALUATION TEST PATCH ===")
print(row["test_patch"])

print("\n=== REQUIRED FAIL_TO_PASS TESTS ===")
print(row["FAIL_TO_PASS"])