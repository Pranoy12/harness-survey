from config import CONFIG
from core.sandbox import DockerSandbox
from harnesses.condition_a import ConditionAHarness
from harnesses.condition_b import ConditionBHarness
from harnesses.condition_c import ConditionCHarness

def test_imports():
    harnessA = ConditionAHarness()
    harnessB = ConditionBHarness()
    harnessC = ConditionCHarness()
    toolsA = harnessA.get_tool_definitions()
    toolsB = harnessB.get_tool_definitions()
    toolsC = harnessC.get_tool_definitions()
    print("-" * 100)
    print(f"Harness: {harnessA.condition_name}")
    print("Interface: Standard JSON Tool Schema")
    print("Context Management: Naive Sliding Window")
    print("Error Feedback: Raw Terminal Dump")
    print("✓ Harness tools loaded:", [t['function']['name'] for t in (toolsA or [])])
    print("-" * 100)
    print(f"Harness: {harnessB.condition_name}")
    print("Interface: Raw Shell / Code Blocks")
    print("Context Management: Naive Sliding Window")
    print("Error Feedback: Raw Terminal Dump")
    print("✓ Harness tools loaded:", [t['function']['name'] for t in (toolsB or [])])
    print("-" * 100)
    print(f"Harness: {harnessC.condition_name}")
    print("Interface: Standard JSON Tool Schema")
    print("Context Management: Naive Sliding Window")
    print("Error Feedback: Sanitized 5-line Error Filter")
    print("✓ Harness tools loaded:", [t['function']['name'] for t in (toolsC or [])])
    print("-" * 100)
    print()
    print(f"✓ Config model provider: {CONFIG.PROVIDER}")
    print(f"✓ Config model target: {CONFIG.MODEL_NAME}")
    print("✓ Environment verified.")

if __name__ == "__main__":
    test_imports()