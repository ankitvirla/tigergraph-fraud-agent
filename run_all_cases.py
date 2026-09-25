import json
import subprocess
from pathlib import Path

CASE_IDS = [f"HHG-{i:03d}" for i in range(1, 21)]

OUTPUT_DIR = Path("submission/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = []

for case_id in CASE_IDS:
    print(f"\n{'=' * 60}")
    print(f"Running {case_id}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        [
            "python",
            "fraud_investigation_agent_llm.py",
            case_id,
            "--json-only",
        ],
        capture_output=True,
        text=True,
    )

    output_file = OUTPUT_DIR / f"{case_id}.json"

    if result.returncode != 0:
        print(f"❌ {case_id} FAILED")
        print(result.stderr)

        summary.append({
            "case_id": case_id,
            "status": "failed",
            "error": result.stderr,
        })
        continue

    try:
        data = json.loads(result.stdout)

        output_file.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        risk = data.get("risk_assessment", {})
        uncertainty = data.get("uncertainty", {})
        next_action = data.get("next_action", {})

        summary.append({
            "case_id": case_id,
            "status": "success",
            "fraud_probability": risk.get("fraud_probability"),
            "uncertainty": uncertainty.get("level"),
            "independent_evidence_count": (
                uncertainty.get("independent_evidence_count")
            ),
            "next_action": next_action.get("action"),
        })

        print(f"✓ Saved: {output_file}")
        print(
            f"  Probability: "
            f"{risk.get('fraud_probability')}"
        )
        print(
            f"  Uncertainty: "
            f"{uncertainty.get('level')}"
        )
        print(
            f"  Next action: "
            f"{next_action.get('action')}"
        )

    except json.JSONDecodeError:
        print(f"❌ {case_id}: Invalid JSON returned")

        summary.append({
            "case_id": case_id,
            "status": "invalid_json",
            "raw_output": result.stdout,
        })


# Save summary
summary_file = OUTPUT_DIR / "benchmark_summary.json"

summary_file.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)

successful = sum(
    1 for x in summary
    if x["status"] == "success"
)

failed = len(summary) - successful

print(f"Successful: {successful}/20")
print(f"Failed:     {failed}/20")
print(f"Output:     {OUTPUT_DIR}")
print(f"Summary:    {summary_file}")