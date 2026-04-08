"""Test single chart upload with real LLM narrative generation."""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"

# Upload a single chart
print("Uploading Chart 01 (US unemployment)...")
import http.client
import os

boundary = "----FormBoundary123456"
svg_path = os.path.join("Chart SVGs", "01. Current - 7 US unemployment rate.svg")
with open(svg_path, "rb") as f:
    svg_data = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="chart01.svg"\r\n'
    f"Content-Type: image/svg+xml\r\n\r\n"
).encode() + svg_data + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    f"{BASE}/api/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

print("Waiting for LLM response (GPT-5 needs ~15-30s for reasoning)...")
r = urllib.request.urlopen(req, timeout=180)
data = json.loads(r.read())

n = data["narrative"]
print(f"\n{'='*60}")
print(f"CHART: {data['insight']['metadata']['title']}")
print(f"{'='*60}")
print(f"\n📋 SUMMARY:\n{n['summary']}")
print(f"\n📝 DETAILED:\n{n['detailed']}")
print(f"\n🎯 KEY TAKEAWAYS:")
for i, t in enumerate(n["key_takeaways"], 1):
    print(f"  {i}. {t}")
print(f"\n🎭 TONE: {n['tone']}")
print(f"\n✅ LLM narrative generated successfully!")

