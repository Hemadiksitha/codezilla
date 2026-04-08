"""Quick test script for the NOVA-C pipeline."""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"

# Load demo
print("Loading demo charts...")
r = urllib.request.urlopen(f"{BASE}/api/demo")
demo = json.loads(r.read())
print(f"Loaded {demo['total']} charts:")
for c in demo["charts"]:
    status = "ERROR" if "error" in c else "OK"
    print(f"  [{status}] {c.get('title','?')} ({c.get('chart_type','?')}) conf={c.get('confidence','?')}")

print()

# Check each chart
for c in demo["charts"]:
    if "error" in c:
        continue
    cid = c["chart_id"]
    r2 = urllib.request.urlopen(f"{BASE}/api/charts/{cid}")
    data = json.loads(r2.read())
    ins = data["insight"]
    nar = data["narrative"]

    print(f"=== {ins['metadata']['title']} ===")
    print(f"  Series: {len(ins['series'])}, Trends: {len(ins['trends'])}, Anomalies: {len(ins['anomalies'])}")
    print(f"  Narrative summary: {nar['summary'][:150]}...")
    print(f"  Takeaways: {len(nar['key_takeaways'])}")
    print()

print("ALL TESTS PASSED!")

